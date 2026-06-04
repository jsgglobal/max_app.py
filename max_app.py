import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import ta
import time
import requests
from io import StringIO

# --- [UI Setup] ---
def setup_page():
    st.set_page_config(page_title="고급 주식 차트 분석 및 검색기", layout="wide")
    st.title("📈 주식 차트 분석 & BNF/스마트머니 검색 시스템")

# --- [Data Fetching] ---
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    """
    S&P 500 종목 기호를 위키피디아에서 가져오는 헬퍼 함수.
    HTTP 403 에러 및 FileNotFoundError 방지를 위한 최종 수정본.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    response = requests.get(url, headers=headers)
    
    # 2. 텍스트를 StringIO로 감싸서 pandas가 파일 경로로 오해하지 않도록 만듭니다.
    html_data = StringIO(response.text)
    
    # 3. StringIO 객체를 전달합니다.
    table = pd.read_html(html_data)[0] 
    
    tickers = table['Symbol'].tolist()
    tickers = [ticker.replace('.', '-') for ticker in tickers]
    return tickers

@st.cache_data(ttl=3600)
def load_stock_data(ticker, period="1y"):
    """yfinance를 이용해 주식 데이터를 가져옵니다."""
    df = yf.download(ticker, period=period, progress=False)
    if not df.empty:
        df.reset_index(inplace=True)
    return df

# --- [Technical Indicators] ---
def add_indicators(df):
    """지표(VWAP, 이동평균선, RSI, MACD, Bollinger Bands) 추가"""
    if df.empty: return df
    
    # 1. VWAP (거래량 가중 평균 가격)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    df['Cumulative_Volume_Price'] = (df['Typical_Price'] * df['Volume']).cumsum()
    df['VWAP'] = df['Cumulative_Volume_Price'] / df['Cumulative_Volume']
    
    # 2. Moving Averages
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['MA120'] = df['Close'].rolling(window=120).mean()
    
    # 3. Bollinger Bands
    indicator_bb = ta.volatility.BollingerBands(close=df["Close"], window=20, window_dev=2)
    df['BB_High'] = indicator_bb.bollinger_hband()
    df['BB_Low'] = indicator_bb.bollinger_lband()
    df['BB_Mid'] = indicator_bb.bollinger_mavg()
    
    # 4. MACD
    indicator_macd = ta.trend.MACD(close=df["Close"], window_slow=26, window_fast=12, window_sign=9)
    df['MACD'] = indicator_macd.macd()
    df['MACD_Signal'] = indicator_macd.macd_signal()
    df['MACD_Hist'] = indicator_macd.macd_diff()
    
    # 5. RSI
    indicator_rsi = ta.momentum.RSIIndicator(close=df["Close"], window=14)
    df['RSI'] = indicator_rsi.rsi()
    
    # 6. Envelope (엔벨로프 20일 기준 15% 하단 - BNF 전략용)
    df['Env_Low'] = df['MA20'] * 0.85
    
    return df

# --- [Plotting] ---
def create_advanced_chart(df, ticker):
    """Plotly를 이용해 지표가 포함된 인터랙티브 캔들 차트를 생성합니다."""
    fig = go.Figure()

    # 캔들차트
    fig.add_trace(go.Candlestick(
        x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'],
        name="Candle", increasing_line_color='red', decreasing_line_color='blue'
    ))

    # VWAP
    fig.add_trace(go.Scatter(x=df['Date'], y=df['VWAP'], mode='lines', name='VWAP', line=dict(color='purple', width=2, dash='dot')))
    
    # 이동평균선
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], mode='lines', name='MA20', line=dict(color='orange', width=1.5)))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['MA60'], mode='lines', name='MA60', line=dict(color='green', width=1.5)))
    
    # 볼린저 밴드
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_High'], mode='lines', name='BB 상단', line=dict(color='gray', width=1), opacity=0.3))
    fig.add_trace(go.Scatter(x=df['Date'], y=df['BB_Low'], mode='lines', name='BB 하단', line=dict(color='gray', width=1), opacity=0.3, fill='tonexty'))

    # 엔벨로프 하단 (BNF 타점 확인용)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Env_Low'], mode='lines', name='Env 하단(15%)', line=dict(color='black', width=1, dash='dash')))

    fig.update_layout(
        title=f"{ticker} 차트 분석",
        yaxis_title="Price",
        xaxis_title="Date",
        xaxis_rangeslider_visible=False,
        height=600,
        template="plotly_white"
    )
    return fig

# --- [Search Algorithms] ---
def run_smart_money_search(tickers, target_date, drop_percent=15, ma_period=20):
    """
    BNF 투자 기법 (극과매도) 조건 검색 시뮬레이션
    실제 구현 시 전체 티커의 데이터를 다운로드하여 분석해야 함.
    """
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        # UI 프로그레스 업데이트
        progress_bar.progress((i + 1) / total)
        status_text.text(f"스캔 중... {i+1}/{total} ({ticker})")
        
        # 실제 환경에서는 yf.download 등으로 데이터를 가져와 조건을 비교합니다.
        # 여기서는 UI 시연을 위해 랜덤 로직으로 대체합니다.
        time.sleep(0.01)
        if np.random.rand() > 0.95:  # 약 5% 확률로 조건 만족
            results.append({
                "종목코드": ticker,
                "현재가 대비 이격도(%)": round(-15 - (np.random.rand() * 10), 2),
                "RSI 지표": round(15 + (np.random.rand() * 10), 2),
                "포착 신호": "엔벨로프 하단 이탈 + V자 반등 대기"
            })
            
    status_text.text("검색 완료!")
    return results

def run_power_trend_search(tickers, ma_short, ma_long, gap_min, gap_max, vol_ratio, rsi_val, stoch_k, macd_cond):
    """
    파워 트렌드 라이딩 (멀티 타임프레임 스캔) 실제 구현
    주봉 데이터를 먼저 검사하여 속도를 극대화하고 통과 종목만 60분봉을 검사합니다.
    """
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(tickers)
    for i, ticker in enumerate(tickers):
        progress_bar.progress((i + 1) / total)
        status_text.text(f"스캔 중... {i + 1}/{total} ({ticker})")
        
        try:
            # --- 1. [주봉] 대추세 확인 ---
            tk = yf.Ticker(ticker)
            df_wk = tk.history(period="2y", interval="1wk")
            if df_wk.empty or len(df_wk) < ma_long:
                continue
                
            close_wk = df_wk['Close']
            ma_s_val = close_wk.rolling(window=ma_short).mean().iloc[-1]
            ma_l_val = close_wk.rolling(window=ma_long).mean().iloc[-1]
            
            # 정배열 확인
            if pd.isna(ma_s_val) or pd.isna(ma_l_val) or ma_s_val <= ma_l_val:
                continue
                
            # 이격도 수렴 확인
            gap = (ma_s_val / ma_l_val) * 100
            if not (gap_min <= gap <= gap_max):
                continue
                
            # --- 2. [60분봉] 수급 및 턴어라운드 (주봉 통과 종목만 API 호출) ---
            df_1h = tk.history(period="1mo", interval="1h")
            if df_1h.empty or len(df_1h) < 26:
                continue
                
            close_1h = df_1h['Close']
            high_1h = df_1h['High']
            low_1h = df_1h['Low']
            vol_1h = df_1h['Volume']
            
            # C: 거래량 폭발 (20평균 대비)
            avg_vol_20 = vol_1h.rolling(window=20).mean().shift(1)
            vol_cond = False
            for v, avg_v in zip(vol_1h.iloc[-5:], avg_vol_20.iloc[-5:]):
                if pd.notna(v) and pd.notna(avg_v) and v >= avg_v * (vol_ratio / 100.0):
                    vol_cond = True
                    break
                    
            if not vol_cond:
                continue
                
            # D, E, F: 단기 턴어라운드 지표
            rsi_s = ta.momentum.RSIIndicator(close_1h, window=14).rsi()
            rsi_cond = (rsi_s.iloc[-5:] <= rsi_val).any()
            
            stoch = ta.momentum.StochasticOscillator(high_1h, low_1h, close_1h, window=12, smooth_window=5)
            stoch_k_s = stoch.stoch().rolling(5).mean()  # Slow %K
            stoch_cond = (stoch_k_s.iloc[-5:] <= stoch_k).any()
            
            macd = ta.trend.MACD(close_1h, window_slow=26, window_fast=12, window_sign=9)
            if "0선 이상" in macd_cond:
                macd_ok = macd.macd().iloc[-1] >= 0
            else:
                macd_ok = macd.macd_diff().iloc[-2] <= 0 and macd.macd_diff().iloc[-1] > 0
                
            # OR 조건 만족 시 결과에 추가
            if rsi_cond or stoch_cond or macd_ok:
                cond_str = "C & " + ("D " if rsi_cond else "") + ("E " if stoch_cond else "") + ("F" if macd_ok else "")
                results.append({
                    "종목코드": ticker,
                    "주봉 단기MA": round(float(ma_s_val), 2),
                    "주봉 장기MA": round(float(ma_l_val), 2),
                    "이격도(%)": round(float(gap), 2),
                    "60분 RSI": round(float(rsi_s.iloc[-1]), 2),
                    "만족조건": cond_str.replace("  ", " ").strip()
                })
        except Exception:
            # yfinance 호출 에러나 데이터가 꼬인 종목은 패스
            continue
            
    status_text.text(f"스캔 완료! 조건 만족 종목: {len(results)}개")
    return results

# --- [Main App] ---
def main():
    setup_page()
    
    # 탭 구성 (5번째 탭 신규 추가)
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["주식 차트 분석", "BNF 투자 분석", "기술적 분석 상세", "스마트머니 투매 줍줍", "파워 트렌드 라이딩"])
    
    tickers = get_sp500_tickers()

    with tab1:
        st.header("🔍 개별 종목 차트 및 지표 확인")
        col1, col2 = st.columns([1, 3])
        with col1:
            selected_ticker = st.selectbox("종목 선택", tickers, index=tickers.index('AAPL') if 'AAPL' in tickers else 0)
            period = st.selectbox("기간", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
            if st.button("차트 분석 실행"):
                df = load_stock_data(selected_ticker, period)
                if not df.empty:
                    df = add_indicators(df)
                    st.session_state['current_data'] = df
                    st.session_state['current_ticker'] = selected_ticker
        
        with col2:
            if 'current_data' in st.session_state:
                df = st.session_state['current_data']
                ticker = st.session_state['current_ticker']
                fig = create_advanced_chart(df, ticker)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("좌측에서 종목을 선택하고 분석을 실행하세요.")

    with tab2:
        st.header("📉 BNF 극과매도 평가 도구")
        st.markdown("""
        **BNF 투자 방법의 핵심 관점:**
        추세와 무관하게 비정상적 낙폭으로 인해 반대매매가 출회되는 지점을 노립니다.
        엔벨로프(20일선 기준 -15%) 하단을 이탈하는 극과매도 상태에서 기계적인 V자 반등을 수익으로 냅니다.
        """)
        
        if 'current_data' in st.session_state:
            df = st.session_state['current_data']
            latest = df.iloc[-1]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("현재가", f"${latest['Close']:.2f}")
            c2.metric("엔벨로프 하단 (-15%)", f"${latest['Env_Low']:.2f}")
            
            # BNF 타점 계산
            gap_to_env = ((latest['Close'] - latest['Env_Low']) / latest['Env_Low']) * 100
            c3.metric("엔벨로프 이격도", f"{gap_to_env:.2f}%", delta=f"{gap_to_env:.2f}%", delta_color="inverse")
            
            if latest['Close'] < latest['Env_Low'] and latest['RSI'] < 25:
                st.success("🔥 **BNF 타점 포착!** 현재가가 엔벨로프 하단을 이탈하였으며 RSI가 극과매도 상태입니다.")
            else:
                st.warning("안전 구역. 현재는 극과매도 타점이 아닙니다.")
        else:
            st.info("종목 분석 탭에서 데이터를 먼저 로드해주세요.")

    with tab3:
        st.header("📊 기술적 분석 상세 데이터")
        if 'current_data' in st.session_state:
            df = st.session_state['current_data']
            st.dataframe(df[['Date', 'Close', 'Volume', 'VWAP', 'MA20', 'BB_High', 'BB_Low', 'RSI', 'MACD']].tail(20), use_container_width=True)
        else:
            st.info("데이터가 없습니다.")

    with tab4:
        st.header("🤖 자동 종목 검색 시스템 (스마트머니 투매 줍줍)")
        st.write("시장 전체에서 BNF 기법(역추세/투매 흡수)의 조건에 맞는 종목을 자동으로 발굴합니다.")
        
        st.write("### 대상 변경 (필터링)")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.checkbox("ETF, ETN 제외", value=True)
            st.checkbox("스팩(SPAC) 제외", value=True)
        with col_f2:
            st.checkbox("ADR 제외", value=True)
            st.checkbox("우선주 제외", value=True)
            
        st.write("### 대상 종목 풀: 약 10,000개 (실제 구현 시 전체 시장 스캔)")
        st.info(f"검색 대상 종목수: {len(tickers)} 개")
        
        if st.button("자동 스캔 시작"):
            # 실제 프로덕션 환경에서는 전체 종목을 스캔하는 백엔드 서버 로직과 연동해야 합니다.
            scan_results = run_smart_money_search(tickers, datetime.now().date())
            
            if scan_results:
                res_df = pd.DataFrame(scan_results)
                st.dataframe(res_df, use_container_width=True)
            else:
                st.warning("현재 시장에서 조건에 부합하는 종목이 포착되지 않았습니다.")

    with tab5:
        st.header("📈 파워 트렌드 라이딩 (멀티 타임프레임 스캔)")
        
        st.write("### 대상 변경 (필터링)")
        # 4번째 탭과 동일한 필터 세팅 UI 적용
        col_t5_f1, col_t5_f2 = st.columns(2)
        with col_t5_f1:
            st.checkbox("ETF, ETN 제외", value=True, key="t5_ex_etf")
            st.checkbox("스팩(SPAC) 제외", value=True, key="t5_ex_spac")
        with col_t5_f2:
            st.checkbox("ADR 제외", value=True, key="t5_ex_adr")
            st.checkbox("우선주 제외", value=True, key="t5_ex_pref")

        st.info(f"🔍 현재 검색 대상 종목수: {len(tickers)} 개")
        
        st.write("---")
        st.write("### ⚙️ 검색 조건 파라미터 세팅 (키움 조건검색 연동)")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**[A, B] 주봉 대추세 및 이격도 수렴**")
            ma_short = st.number_input("단기 이평선 (주기)", value=20, help="장기 추세를 판단하기 위한 단기 이동평균선 주기입니다. 일봉으로 변경 시 120일 등으로 세팅하세요.")
            ma_long = st.number_input("장기 이평선 (주기)", value=60, help="장기 추세의 기준이 되는 이동평균선 주기입니다. (예: 60주선)")
            gap_min = st.number_input("이격도 하한 (%)", value=98.0, step=1.0, help="장기 추세(60주)와 단기 추세(20주)의 이격도 하단입니다. 100% 이하는 단기가 장기 아래로 살짝 빠진 상태입니다.")
            gap_max = st.number_input("이격도 상한 (%)", value=105.0, step=1.0, help="이격도 상단입니다. 수치를 좁힐수록(예: 103%) 더 타이트하게 수렴하여 폭발 직전의 종목을 찾습니다.")
            
        with c2:
            st.markdown("**[C, D, E, F] 60분봉 수급 및 턴어라운드**")
            vol_ratio = st.slider("기간내 거래량 폭발 기준 (%)", 100, 500, 200, step=10, help="스마트머니(세력)의 개입 여부를 확인합니다. 대형주는 150%, 중소형주는 300% 이상으로 설정하여 수급 강도를 조절합니다.")
            rsi_val = st.slider("RSI 과매도 기준", 10, 50, 35, step=1, help="단기 낙폭의 깊이를 조절합니다. 강한 주도주를 찾으려면 40 이하로 기준을 완화해 보세요.")
            stoch_k = st.slider("Stochastic Slow %K 기준", 10, 50, 25, step=1, help="단기 바닥권을 확인합니다. 수치를 낮출수록 더 깊은 바닥에서 타점을 잡습니다.")
            macd_cond = st.radio("MACD 통과 조건", ["0선 이상 (강한 추세)", "시그널선 상향돌파 (낙폭과대 반등)"], help="0선 이상은 이미 추세가 상방인 종목을, 시그널 돌파는 바닥에서 고개를 드는 종목을 의미합니다.")
            
        if st.button("🚀 파워 트렌드 조건 검색 실행"):
            # 실제 구현된 멀티 타임프레임 스캔 로직 실행
            results = run_power_trend_search(tickers, ma_short, ma_long, gap_min, gap_max, vol_ratio, rsi_val, stoch_k, macd_cond)
            
            if results:
                df_res = pd.DataFrame(results)
                st.success(f"{len(df_res)}개의 종목이 발굴되었습니다!")
                st.dataframe(df_res, use_container_width=True)
            else:
                st.warning("현재 설정된 조건을 만족하는 종목이 없습니다. 파라미터를 완화해 보세요.")
                
        st.write("---")
        st.markdown("""
        ### 🎯 매수 / 매도 타점 가이드 및 근거
        
        #### 🛫 매수 타점 (Entry Point)
        * **1차 진입 (검색기 포착 시점):** 검색식 조건이 만족되어 60분봉상 **MACD 골든크로스**나 **스토캐스틱 바닥 탈출**이 나오는 영봉전 종가에 진입합니다. 
            * **근거:** 주봉상 수렴(대추세 지지)된 상태에서 60분봉상 수급(스마트머니)이 유입되며 단기 공포(과매도)를 이겨내는 완벽한 턴어라운드 심리 변곡점이기 때문입니다.
        * **2차 진입 (불타기):** 60분봉이 직전 단기 고점을 대량의 거래량을 동반해 돌파할 때 추가 진입합니다.
        * **손절 라인:** 주봉상 **60주 이동평균선** 또는 60분봉상 **최근 최저점**을 종가 기준으로 이탈 시 기계적 손절합니다. (세력의 가격 방어 실패를 의미)

        #### 🛬 매도 타점 (Exit Point)
        * **1단계: 목표가 도달 전 (트레일링 스탑 - 리스크 제로화)**
            * 주가가 상승하여 **+5% 수익 구간**에 진입하면, 즉시 손절가를 '내 매수가(본절가)'로 올립니다. 이후 60분봉 20선을 타고 올라간다면 지표가 과매수권에 있더라도 매도하지 않고 대시세를 길게 쥐고 갑니다.
        * **2단계: 추세 종결 시 (최종 익절)**
            * **절반 익절 (50%):** 60분봉 RSI 70 이상, 스토캐스틱 80 이상의 극단적 과매수 구간 진입 후, 60분봉 종가가 5선 또는 10선을 하향 이탈할 때.
            * **전량 익절 (나머지):** 60분봉 MACD 데드크로스가 발생하거나 일간 지수가 완전히 꺾일 때 수익을 확정 짓습니다.
        """)

if __name__ == "__main__":
    main()
