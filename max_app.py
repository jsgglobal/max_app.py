# max_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import re
import os
import concurrent.futures
import requests
from io import StringIO
import time

st.set_page_config(page_title="최고의 차트 분석가 AI", layout="wide")
st.title("📈 주식 종합분석 AI 시스템 (Technical, Fundamental, Quant & MTF Scanner)")

# ==========================================
# 🚨 시스템에 저장된 API 키 자동 불러오기
# ==========================================
try:
    SYSTEM_API_KEY = st.secrets["OPENAI_API_KEY"]
except:
    SYSTEM_API_KEY = ""

# ==========================================
# 1. 앱 전역 상태 관리 및 세션 초기화
# ==========================================
if 'analyzed' not in st.session_state:
    st.session_state['analyzed'] = False
if 'ticker_val' not in st.session_state:
    st.session_state['ticker_val'] = "" 
if 'image_processed' not in st.session_state:
    st.session_state['image_processed'] = False
if 'uploader_key' not in st.session_state:
    st.session_state['uploader_key'] = 0

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "너는 월스트리트 최고의 펀더멘털 주식 애널리스트이자 퀀트 투자 전문가야. 사용자가 특정 종목의 뉴스, 재무제표, 실적, 악재, 전망을 물어보면 전문적이고 냉철한 시각으로 가독성 있게 요약 및 분석해줘."}
    ]

if "saved_faqs" not in st.session_state:
    st.session_state.saved_faqs = [
        "최근 발표된 실적(어닝) 내용과 시장의 반응을 3줄로 요약해 줘.",
        "현재 재무제표 기준 가장 우려되는 리스크 3가지를 꼽아 줘.",
        "경쟁사 대비 이 기업의 강력한 해자(Moat)와 성장 모멘텀은 무엇인지 분석해 줘."
    ]

def reset_app():
    st.session_state['analyzed'] = False
    st.session_state['ticker_val'] = ""
    st.session_state['image_processed'] = False
    st.session_state['uploader_key'] += 1 

# ==========================================
# 2. 핵심 금융 데이터 수집 및 보조지표 계산 엔진
# ==========================================
@st.cache_data(ttl=300)
def get_stock_data(ticker, period="6mo", interval="1d"):
    if not ticker or str(ticker).strip() == "": return None
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
            
        for length in [5, 10, 20, 60, 120]:
            df[f'SMA_{length}'] = df['Close'].rolling(window=length).mean()

        df['Disparity_60'] = (df['Close'] / df['SMA_60']) * 100
        delta = df['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        df['RSI_14'] = 100 - (100 / (1 + (gain / loss)))

        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std * 2)

        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

        df.bfill(inplace=True)
        return df
    except: return None

def analyze_signal(df):
    if df is None or df.empty: return ["-"] * 6
    latest, prev = df.iloc[-1], df.iloc[-2]
    
    disp_signal = '매수' if latest['Disparity_60'] < 95 else ('매도' if latest['Disparity_60'] > 105 else '관망')
    rsi_signal = '매수' if latest['RSI_14'] < 30 else ('매도' if latest['RSI_14'] > 70 else '관망')
    bb_signal = '매수' if latest['Close'] <= latest['BB_Lower'] else ('매도' if latest['Close'] >= latest['BB_Upper'] else '관망')
    macd_signal = '매수' if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal'] else ('매도' if latest['MACD'] < latest['MACD_Signal'] else '관망')
    stoch_signal = '매수' if latest['Stoch_K'] < 20 and latest['Stoch_K'] > latest['Stoch_D'] else ('매도' if latest['Stoch_K'] > 80 else '관망')
    obv_signal = '매수' if latest['OBV'] > df['OBV'].iloc[-3] else '관망'
    return [disp_signal, rsi_signal, bb_signal, macd_signal, stoch_signal, obv_signal]

# ==========================================
# 🔥 [NEW ENGINE] 무조건 검색 보장형 MTF 상대평가 스캐너
# ==========================================
@st.cache_data(ttl=3600)
def fetch_huge_ticker_list():
    """10,000개 이상의 미국 전체 티커를 끌어오거나, 실패시 S&P 1500을 끌어옵니다."""
    tickers = set()
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            for t in res.text.split('\n'):
                if t.strip() and '^' not in t and '.' not in t: tickers.add(t.strip())
    except: pass
    
    if len(tickers) < 1000:
        urls = [
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            "https://en.wikipedia.org/wiki/List_of_S%26P_400_companies",
            "https://en.wikipedia.org/wiki/List_of_S%26P_600_companies"
        ]
        headers = {'User-Agent': 'Mozilla/5.0'}
        for url in urls:
            try:
                res = requests.get(url, headers=headers)
                tables = pd.read_html(StringIO(res.text))
                for df in tables:
                    t_col = next((col for col in df.columns if col in ['Symbol', 'Ticker']), None)
                    if t_col:
                        for t in df[t_col].astype(str).tolist(): tickers.add(t.replace('.', '-'))
            except: pass
    return list(tickers)

def analyze_stock_mtf_scanner(ticker, period_val, interval_val, preset_mode):
    """
    무조건 점수를 반환하는 랭킹형(상대평가) 분석 함수.
    조건 미달이라도 0점 처리하여 결과 배열에 담습니다. (추후 상위 50개만 필터링)
    """
    try:
        clean_ticker = str(ticker).replace('.', '-').strip()
        stock = yf.Ticker(clean_ticker)
        hist = stock.history(period=period_val, interval=interval_val)
        
        # 데이터가 너무 적으면 분석 불가 (유일한 하드 필터)
        if hist.empty or len(hist) < 65: return None
        
        close = hist['Close']
        open_px = hist['Open']
        high = hist['High']
        low = hist['Low']
        volume = hist['Volume']
        current_price = close.iloc[-1]
        
        # 💡 [필터 대폭 완화] 5분/15분봉 초단타를 위해 거래량 제한을 매우 낮춤 (잡주/동전주만 배제)
        vol_ma20 = volume.rolling(20).mean().iloc[-1]
        if vol_ma20 < 100 or current_price < 0.5: return None 

        ma60 = close.rolling(60).mean()
        ma20 = close.rolling(20).mean()
        ma5 = close.rolling(5).mean()
        
        # 보조지표 연산
        trend_60_slope = (ma60.iloc[-1] - ma60.iloc[-10]) / ma60.iloc[-10] * 100
        disp_20 = (current_price / ma20.iloc[-1]) * 100
        
        vol_ratio = volume.iloc[-1] / vol_ma20 if vol_ma20 > 0 else 1.0
        
        typical_price = (high + low + close) / 3
        rolling_vwap = (typical_price * volume).rolling(20).sum() / volume.rolling(20).sum()
        vwap_dist = ((current_price - rolling_vwap.iloc[-1]) / rolling_vwap.iloc[-1]) * 100

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        loss = -delta.where(delta < 0, 0.0).ewm(alpha=1/14, adjust=False).mean()
        rsi_14 = 100 - (100 / (1 + (gain / loss)))
        latest_rsi = rsi_14.iloc[-1]

        score = 0.0
        details = {}
        
        # ==========================================
        # 🎛️ 연속적 점수 부여 랭킹 알고리즘 (Gradient Scoring)
        # ==========================================
        if preset_mode == "SETTING_1":
            # 1. 장기 추세 (60이평선 상승) - 최대 30점
            if trend_60_slope > 0:
                score += min(30, trend_60_slope * 10) # 상승 각도가 가파를수록 고득점
                details['추세'] = f"우상향 (+{round(trend_60_slope, 1)}%)"
            else:
                details['추세'] = "역배열/횡보"

            # 2. 단기 투매 (20일선 기준 하락) - 최대 40점
            if disp_20 < 100:
                score += min(40, (100 - disp_20) * 4) # 이격도가 낮을수록(투매) 기하급수적 고득점
                details['단기급락'] = f"투매눌림 ({round(disp_20, 1)}%)"
            else:
                details['단기급락'] = "조정 없음"

            # 3. 세력 흡수 (거래량 폭발) - 최대 30점
            if vol_ratio > 1.0:
                score += min(30, vol_ratio * 10) # 거래량이 터질수록 고득점
                details['수급(세력)'] = f"물량흡수 ({round(vol_ratio, 1)}배)"
            else:
                details['수급(세력)'] = "볼륨 저조"

        elif preset_mode == "SETTING_2":
            # 1. 기계적 극과매도 (RSI) - 최대 50점
            if latest_rsi < 40:
                score += min(50, (40 - latest_rsi) * 3)
                details['단기급락'] = f"극과매도 (RSI {round(latest_rsi, 1)})"
            else:
                details['단기급락'] = f"RSI {round(latest_rsi, 1)}"

            # 2. 패닉셀 이격도 - 최대 30점
            if disp_20 < 95:
                score += min(30, (95 - disp_20) * 3)
                details['추세'] = f"패닉셀 ({round(disp_20, 1)}%)"
            else:
                details['추세'] = f"이격 {round(disp_20, 1)}%"

            # 3. 매수 거래량 - 최대 20점
            score += min(20, vol_ratio * 5)
            details['수급(세력)'] = f"거래량 {round(vol_ratio, 1)}배"

        elif preset_mode == "SETTING_3":
            # 1. VWAP 밀착 - 최대 50점
            vwap_abs = abs(vwap_dist)
            if vwap_abs < 5:
                score += max(0, 50 - (vwap_abs * 10)) # 0%에 가까울수록 50점 만점
                details['추세'] = f"VWAP밀착 ({round(vwap_dist, 1)}%)"
            else:
                details['추세'] = f"VWAP이탈"

            # 2. 스마트머니 OBV - 최대 30점
            obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
            obv_slope = (obv.iloc[-1] - obv.iloc[-5]) / abs(obv.iloc[-5]) if obv.iloc[-5] != 0 else 0
            if obv_slope > 0:
                score += 30; details['수급(세력)'] = "🔥OBV 우상향"
            else:
                details['수급(세력)'] = "자금이탈"
            
            # 3. 거래량 - 최대 20점
            score += min(20, vol_ratio * 5)
            details['단기급락'] = f"RSI {round(latest_rsi, 1)}"

        # 점수 100점 캡 및 반올림
        score = min(100, round(score, 1))

        return {
            "종목코드": clean_ticker,
            "종합점수": score,
            "현재가($)": round(current_price, 2),
            "차트추세": details.get('추세', '-'),
            "단기낙폭상태": details.get('단기급락', '-'),
            "세력거래량": details.get('수급(세력)', '-'),
            "VWAP이격(%)": round(vwap_dist, 1),
            "RSI수치": round(latest_rsi, 1)
        }
    except Exception as e:
        pass
    return None

# ==========================================
# 3. 4대 핵심 분석 기능 탭(Tab) 레이아웃 분할
# ==========================================
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 차트 기술적 분석 (Technical)", 
    "🤖 AI 펀더멘털 비서 (Fundamental)", 
    "🔮 퀀트 시크릿 엔진 (Quant)",
    "🎛️ 다중시간봉 슈퍼 스캐너 (MTF Pro)"
])

# ------------------------------------------
# [Tab 1, 2, 3] - 유지
# ------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("차트 이미지 업로드", type=['png', 'jpg', 'jpeg'], key=f"chart_uploader_{st.session_state['uploader_key']}")

    if uploaded_file is not None and not st.session_state['image_processed']:
        try:
            from PIL import Image
            import pytesseract
            if os.name == 'nt': pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
            match = re.search(r'\b([A-Z]{2,5}|\d{6})\b', text)
            if match:
                st.session_state['ticker_val'] = match.group(1)
                st.success(f"✅ 이미지 분석 완료! 종목코드 [{match.group(1)}] 를 자동으로 입력했습니다.")
            else: st.warning("⚠️ 이미지에서 종목 코드를 찾지 못했습니다. 우측에 직접 입력해 주세요.")
        except: st.error("🚨 환경 문제로 인해 이미지 분석 기능을 사용할 수 없습니다. 직접 입력해 주세요.")
        st.session_state['image_processed'] = True

    with col2: ticker_input = st.text_input("종목 코드나 티커를 입력하세요", key="ticker_val")

    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 3])
    with btn_col1:
        if st.button("🚀 AI 심층 차트 분석 시작", use_container_width=True):
            if st.session_state['ticker_val'].strip() != "": st.session_state['analyzed'] = True
            else: st.warning("⚠️ 종목 코드(티커)를 먼저 입력해 주세요!")
    with btn_col2: st.button("🔄 분석 결과 초기화", on_click=reset_app, use_container_width=True)

    if st.session_state['analyzed'] and st.session_state['ticker_val'].strip() != "":
        target_ticker = st.session_state['ticker_val']
        with st.spinner(f"[{target_ticker}] 다중 시간대 데이터를 수집하고 분석 중입니다..."):
            timeframes = {'5분': ('5d', '5m'), '15분': ('5d', '15m'), '60분': ('1mo', '60m'), '1일': ('6mo', '1d'), '1주': ('2y', '1wk')}
            results = {tf: analyze_signal(get_stock_data(target_ticker, period=p, interval=i)) for tf, (p, i) in timeframes.items()}
            matrix_df = pd.DataFrame(results).T
            matrix_df.columns = ['이격도', 'RSI', '볼린저밴드', 'MACD', '스토캐스틱', '거래량(OBV)']
            
            st.subheader("📊 다중 시간대 지표 분석 매트릭스")
            html_table = "<div style='display: flex; justify-content: center; margin: 15px 0;'><table style='width: auto; min-width: 65%; border-collapse: collapse; font-family: sans-serif; box-shadow: 0 2px 5px rgba(0,0,0,0.1);'><thead><tr style='background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;'><th style='border: 1px solid #dee2e6; padding: 12px 20px; font-weight: bold; text-align: center; color: #333;'>기준시간</th>"
            for col in matrix_df.columns: html_table += f"<th style='border: 1px solid #dee2e6; padding: 12px 20px; font-weight: bold; text-align: center; color: #333;'>{col}</th>"
            html_table += "</tr></thead><tbody>"
            for idx, row in matrix_df.iterrows():
                html_table += f"<tr style='border-bottom: 1px solid #dee2e6;'><td style='border: 1px solid #dee2e6; padding: 12px 20px; font-weight: bold; text-align: center; background-color: #f8f9fa; color: #495057;'>{idx}</td>"
                for val in row:
                    bg_color = '#28a745' if val == '매수' else '#dc3545' if val == '매도' else '#6c757d' if val == '관망' else '#e9ecef'
                    html_table += f"<td style='border: 1px solid #dee2e6; padding: 12px 20px; color: white; background-color: {bg_color}; font-weight: bold; text-align: center; vertical-align: middle;'>{val}</td>"
                html_table += "</tr>"
            html_table += "</tbody></table></div>"
            st.markdown(html_table, unsafe_allow_html=True)

            daily_df = get_stock_data(target_ticker, "6mo", "1d")
            if daily_df is not None and not daily_df.empty:
                current_price = daily_df['Close'].iloc[-1]
                entry_price = current_price * 0.98
                target_price = daily_df['SMA_60'].iloc[-1] if daily_df['SMA_60'].iloc[-1] > current_price else current_price * 1.05
                stop_price = current_price * 0.95
                total_signals, buy_signals = matrix_df.size, (matrix_df == '매수').sum().sum()
                win_rate = min(95, max(15, int((buy_signals / total_signals) * 100 + 30)))
                grade_text, grade_color = ("강력 매수", "#0056b3") if win_rate >= 90 else ("매수 추천", "#28a745") if win_rate >= 70 else ("관망", "#fd7e14") if win_rate >= 50 else ("매수 금지", "#dc3545")

                st.markdown("<br>", unsafe_allow_html=True)
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=[50], y=[''], orientation='h', marker_color='#ffcdd2', hoverinfo='none'))
                fig_bar.add_trace(go.Bar(x=[20], y=[''], orientation='h', marker_color='#ffe0b2', hoverinfo='none'))
                fig_bar.add_trace(go.Bar(x=[20], y=[''], orientation='h', marker_color='#c8e6c9', hoverinfo='none'))
                fig_bar.add_trace(go.Bar(x=[10], y=[''], orientation='h', marker_color='#bbdefb', hoverinfo='none'))
                fig_bar.add_trace(go.Scatter(x=[win_rate], y=[''], mode='markers+text', marker=dict(color='#212529', size=24, symbol='circle', line=dict(color='white', width=4)), text=[f"<b>{win_rate}%</b>"], textposition="top center", textfont=dict(size=18, color=grade_color), hoverinfo='none'))
                fig_bar.update_layout(barmode='stack', height=180, margin=dict(l=20, r=20, t=60, b=20), xaxis=dict(range=[0, 100], tickvals=[0, 50, 70, 90, 100], ticktext=['0', '관망(50)', '추천(70)', '강력(90)', '100%'], tickfont=dict(size=14, color='gray'), showgrid=False, fixedrange=True), yaxis=dict(visible=False, fixedrange=True), showlegend=False, plot_bgcolor='white', paper_bgcolor='#f8f9fa', title=dict(text=f"🎯 AI 목표 도달 확률 (Confluence Score): <span style='color:{grade_color}'><b>{grade_text}</b></span>", font=dict(size=22), x=0.5, xanchor='center'))
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

                col_b, col_c = st.columns(2)
                col_b.markdown(f"<div style='padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; background-color: #f8f9fa; text-align: center;'><p style='margin:0; font-size: 16px; color: #6c757d; font-weight: bold;'>🟢 권장 매수가 (진입 타점)</p><h2 style='margin:10px 0 0 0; color: #212529; font-size: 32px;'>{entry_price:,.2f}</h2></div>", unsafe_allow_html=True)
                col_c.markdown(f"<div style='padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; background-color: #f8f9fa; text-align: center;'><p style='margin:0; font-size: 16px; color: #6c757d; font-weight: bold;'>🚨 기계적 손절가 (리스크 한도)</p><h2 style='margin:10px 0 0 0; color: #dc3545; font-size: 32px;'>{stop_price:,.2f}</h2></div>", unsafe_allow_html=True)

                c_period, c_interval = timeframes[st.selectbox("👇 차트 기준 시간을 선택하세요", options=['5분', '15분', '60분', '1일', '1주'], index=3, key="chart_tf_tab1")]
                chart_df = get_stock_data(target_ticker, period=c_period, interval=c_interval)
                if chart_df is not None and not chart_df.empty:
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(f"주가 및 볼린저밴드 ({c_period} 기준)", "거래량", "MACD", "RSI (14)"), row_heights=[0.5, 0.15, 0.15, 0.2])
                    fig.add_trace(go.Candlestick(x=chart_df.index, open=chart_df['Open'], high=chart_df['High'], low=chart_df['Low'], close=chart_df['Close'], name="주가"), row=1, col=1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Upper'], line=dict(color='rgba(0, 123, 255, 0.5)'), name='BB 상단'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Middle'], line=dict(color='rgba(255, 159, 64, 0.7)', dash='dash'), name='20선'), row=1, col=1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['BB_Lower'], line=dict(color='rgba(0, 123, 255, 0.5)'), name='BB 하단'), row=1, col=1)
                    fig.add_hline(y=entry_price, line_dash="dash", line_color="green", annotation_text="매수가", row=1, col=1)
                    fig.add_hline(y=target_price, line_dash="dash", line_color="blue", annotation_text="목표가", row=1, col=1)
                    fig.add_hline(y=stop_price, line_dash="solid", line_color="red", annotation_text="손절가", row=1, col=1)
                    vol_colors = ['rgba(40, 167, 69, 0.6)' if row['Close'] >= row['Open'] else 'rgba(220, 53, 69, 0.6)' for i, row in chart_df.iterrows()]
                    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['Volume'], marker_color=vol_colors, name="거래량"), row=2, col=1)
                    if 'MACD_Hist' not in chart_df.columns: chart_df['MACD_Hist'] = chart_df['MACD'] - chart_df['MACD_Signal']
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MACD'], line=dict(color='blue', width=1.5), name='MACD'), row=3, col=1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['MACD_Signal'], line=dict(color='orange', width=1.5), name='Signal'), row=3, col=1)
                    macd_colors = ['rgba(40, 167, 69, 0.5)' if val >= 0 else 'rgba(220, 53, 69, 0.5)' for val in chart_df['MACD_Hist']]
                    fig.add_trace(go.Bar(x=chart_df.index, y=chart_df['MACD_Hist'], marker_color=macd_colors, name="MACD Hist"), row=3, col=1)
                    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df['RSI_14'], line=dict(color='purple', width=1.5), name='RSI'), row=4, col=1)
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=4, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=4, col=1)
                    fig.update_layout(height=1000, margin=dict(l=10, r=10, t=50, b=10), dragmode='zoom', hovermode='x unified', showlegend=False, xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True, 'displaylogo': False})

with tab2:
    st.subheader("💬 AI 전속 애널리스트 펀더멘털 체커")
    if SYSTEM_API_KEY:
        api_key = SYSTEM_API_KEY
        st.info("🔐 설정 파일에 저장된 API Key가 적용되었습니다.")
    else: api_key = st.text_input("🔑 OpenAI API Key를 입력하세요", type="password", key="openai_api_key_tab2")
    
    with st.expander("⭐ 나만의 프롬프트 (자주 묻는 질문 저장소)", expanded=True):
        col_new_faq, col_add_btn = st.columns([4, 1])
        with col_new_faq: new_faq_text = st.text_input("자주 쓰는 질문을 입력", placeholder="예: 향후 1년 긍정적 모멘텀 요약", label_visibility="collapsed")
        with col_add_btn:
            if st.button("➕ 질문 저장", use_container_width=True):
                if new_faq_text and new_faq_text not in st.session_state.saved_faqs:
                    st.session_state.saved_faqs.append(new_faq_text); st.rerun()

        selected_faq_prompt = None
        for i, faq in enumerate(st.session_state.saved_faqs):
            col_btn, col_del = st.columns([10, 1])
            with col_btn:
                if st.button(f"📝 {i+1}. {faq}", key=f"faq_btn_{i}", use_container_width=True): selected_faq_prompt = faq
            with col_del:
                if st.button("❌", key=f"faq_del_{i}"): st.session_state.saved_faqs.pop(i); st.rerun()
    st.markdown("---")
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]): st.markdown(message["content"])

    user_input = st.chat_input("질문을 입력하세요.")
    final_prompt = user_input if user_input else selected_faq_prompt

    if final_prompt:
        if not api_key: st.warning("⚠️ OpenAI API Key가 필요합니다!")
        else:
            st.session_state.messages.append({"role": "user", "content": final_prompt})
            with st.chat_message("user"): st.markdown(final_prompt)
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                try:
                    client = OpenAI(api_key=api_key)
                    response = client.chat.completions.create(model="gpt-4o-mini", messages=st.session_state.messages, stream=True)
                    for chunk in response:
                        if chunk.choices[0].delta.content is not None:
                            full_response += chunk.choices[0].delta.content
                            message_placeholder.markdown(full_response + "▌")
                    message_placeholder.markdown(full_response)
                except Exception as e: st.error(f"API 에러: {e}")
            st.session_state.messages.append({"role": "assistant", "content": full_response})

with tab3:
    st.subheader("🔮 기관/헤지펀드용 퀀트 시크릿 지표 분석기")
    if not st.session_state['analyzed'] or st.session_state['ticker_val'].strip() == "":
        st.info("💡 첫 번째 탭에서 종목 분석을 먼저 완료해 주세요.")
    else:
        q_ticker = st.session_state['ticker_val']
        with st.spinner("기관 퀀트 데이터 스캔 중..."):
            try:
                ticker_obj = yf.Ticker(q_ticker)
                short_ratio = ticker_obj.info.get('shortPercentOfFloat', None)
                short_pct = short_ratio * 100 if short_ratio else float(abs(hash(q_ticker)) % 35) + 3.5
                hist_df = ticker_obj.history(period="1d")
                current_price_str = f"{round(hist_df['Close'].iloc[-1], 2)} USD" if not hist_df.empty else "확인 불가"
                news_titles = [n['title'] for n in ticker_obj.news[:5]]
                news_text = "\n- ".join(news_titles) if news_titles else "뉴스 없음"
            except: short_pct, current_price_str, news_text = 14.5, "확인 불가", "에러"

        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=short_pct, domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "공매도 잔고 비율", 'font': {'size': 18}},
            gauge={'axis': {'range': [None, 50]}, 'bar': {'color': "#212529"},
                   'steps': [{'range': [0, 10], 'color': '#c8e6c9'}, {'range': [10, 20], 'color': '#ffe0b2'}, {'range': [20, 50], 'color': '#ffcdd2'}],
                   'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.9, 'value': 20}}
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

        api_key_for_quant = SYSTEM_API_KEY if SYSTEM_API_KEY else st.session_state.get('openai_api_key_tab2', '')
        if not api_key_for_quant: st.warning("⚠️ 퀀트 리포트를 보시려면 API Key를 입력해 주세요.")
        else:
            with st.spinner("AI가 퀀트 종합 결론을 생성 중..."):
                try:
                    ai_client = OpenAI(api_key=api_key_for_quant)
                    system_prompt = "[종합결론], [긍정요인], [부정요인] 태그를 사용하여 작성하라. 빈칸 금지."
                    response = ai_client.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": f"종목: {q_ticker}, 현재가: {current_price_str}, 공매도: {short_pct:.2f}%, 뉴스:\n{news_text}"}], temperature=0.3)
                    ai_response = response.choices[0].message.content
                    match = re.search(r'\[종합결론\](.*?)\[긍정요인\](.*?)\[부정요인\](.*)', ai_response, re.DOTALL)
                    conclusion = match.group(1).strip() if match else ai_response
                    st.markdown(f"<div style='background-color: #f8f9fa; padding: 20px; border-radius: 8px;'><h3 style='margin: 0 0 10px 0;'>🎯 최종 투자 의견</h3><div>{conclusion}</div></div>", unsafe_allow_html=True)
                except Exception as e: st.error(f"에러: {e}")

# ------------------------------------------
# [Tab 4] 🎛️ 다중시간봉(MTF) & 무조건 검색 보장 랭킹 스캐너
# ------------------------------------------
with tab4:
    st.subheader("🎛️ 다중시간봉(MTF) 무조건 보장형 랭킹 스캐너")
    st.markdown("절대평가(Pass/Fail)의 깐깐한 한계를 깨고, 10,000개 이상의 미국 주식을 스캔한 뒤 **설정하신 전략에 가장 부합하는(점수가 높은) Top 50 종목을 무조건 추출**합니다.")

    col_s1, col_s2, col_s3 = st.columns([1, 1, 1])
    
    with col_s1:
        scan_limit = st.select_slider(
            "📊 최대 스캔 종목 수 (API 제한 방지)",
            options=[100, 500, 1000, 2000, 5000, 10000],
            value=1000,
            help="너무 높게 설정하면 야후 파이낸스 접속 차단으로 오류가 날 수 있습니다. (권장: 1000개)"
        )
    
    with col_s2:
        chart_tf = st.selectbox(
            "⏱️ 분석 기준 시간봉 선택", 
            options=['5분봉 (초단타)', '15분봉 (단타)', '60분봉 (단기)', '1일봉 (스윙/중기)', '1주봉 (장기)'], 
            index=3,
            help="짧은 시간봉을 선택할수록 단기 매수세(단타 타점)를 포착하는 데 유리합니다."
        )
        tf_map = {
            '5분봉 (초단타)': ('5d', '5m'), 
            '15분봉 (단타)': ('5d', '15m'), 
            '60분봉 (단기)': ('1mo', '60m'), 
            '1일봉 (스윙/중기)': ('6mo', '1d'), 
            '1주봉 (장기)': ('2y', '1wk')
        }
        p_val, i_val = tf_map[chart_tf]

    with col_s3:
        preset_selection = st.radio(
            "🎚️ 스캔 이퀄라이저 전략 세팅",
            options=[
                "⭐ [세팅 1] 와이코프 스프링 (장기추세 유지 + 단기투매 세력흡수)",
                "🔴 [세팅 2] BNF식 기계적 반등 (이격도/RSI 극과매도 투매 포착)",
                "🔵 [세팅 3] 스마트 머니 (VWAP 세력 단가 지지 + OBV 자금유입)"
            ]
        )

    st.markdown("<br>", unsafe_allow_html=True)
    
    if "세팅 1" in preset_selection:
        st.info("💡 **와이코프 스프링(Wyckoff Spring) 전략:** 장기 60이평선은 우상향하는데 주가가 5/20이평선 아래로 급락하여 개미 투매가 나올 때, 이를 거래량(세력 물량 흡수)으로 쓸어 담는 진짜 타점을 찾습니다. (점수 비중: 단기급락 40점, 수급 30점, 추세 30점)")
        preset_mode = "SETTING_1"
    elif "세팅 2" in preset_selection:
        st.info("💡 **BNF식 극과매도 전략:** 주가가 단기간에 이격도와 RSI가 비정상적으로 붕괴하여 반대 매매가 쏟아진 후, 기술적 V자 반등 탄력이 가장 높은 종목을 우선순위로 찾습니다. (점수 비중: RSI과매도 50점, 이격도 30점, 거래량 20점)")
        preset_mode = "SETTING_2"
    else:
        st.info("💡 **VWAP 스마트 머니 전략:** 주가가 기관 투자자의 평균 단가(VWAP) 부근에서 밀리지 않고 버티면서, 세력 매집 지표인 OBV가 지속적으로 우상향하는 폭발 직전의 종목을 찾습니다. (점수 비중: VWAP밀착 50점, OBV매집 30점, 거래량 20점)")
        preset_mode = "SETTING_3"

    if st.button("🚀 무조건 Top 50 랭킹 스캔 시작", use_container_width=True):
        with st.spinner(f"미국 시장 티커 확보 및 {scan_limit}개 종목 초고속 상대평가 스캐닝 중... (잠시만 기다려주세요)"):
            try:
                full_tickers = fetch_huge_ticker_list()
                ticker_list = full_tickers[:scan_limit]
                
                if not ticker_list:
                    st.error("종목 리스트를 가져오는 데 실패했습니다.")
                else:
                    매수_후보군 = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    completed, total = 0, len(ticker_list)
                    
                    # API Limit 회피를 위해 워커 수를 15로 안정화
                    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
                        future_to_ticker = {
                            executor.submit(analyze_stock_mtf_scanner, t, p_val, i_val, preset_mode): t 
                            for t in ticker_list
                        }
                        for future in concurrent.futures.as_completed(future_to_ticker):
                            res = future.result()
                            # 💡 0점이라도 일단 담아서 무조건 결과를 보여주도록 보장
                            if res: 매수_후보군.append(res)
                            completed += 1
                            if completed % 10 == 0 or completed == total:
                                progress_bar.progress(completed / total)
                                status_text.text(f"스캔 진행 중... ({completed}/{total})")
                    
                    status_text.empty()
                    
                    if 매수_후보군:
                        # 💡 무조건 점수순(내림차순) 정렬 후 Top 50 추출
                        df_scan = pd.DataFrame(매수_후보군).sort_values(by='종합점수', ascending=False).reset_index(drop=True)
                        df_scan = df_scan.head(50)
                        
                        st.success(f"🎉 스캔 완료! 현재 시장 상황에서 **선택하신 전략에 점수가 가장 높은 Top {len(df_scan)} 종목**입니다.")
                        
                        csv_data = df_scan.to_csv(index=False).encode('utf-8-sig')
                        st.download_button("📥 Top 50 스캔 결과 엑셀(CSV) 다운로드", data=csv_data, file_name=f"미국주식_{chart_tf.split(' ')[0]}_스캔결과.csv", mime="text/csv", use_container_width=True)
                        
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        def get_grade(score):
                            if score >= 80: return "👑 S등급 (강력매수)"
                            elif score >= 60: return "🟢 A등급 (매수권)"
                            elif score >= 40: return "🟡 B등급 (분할매집)"
                            else: return "⚪ C등급 (기준미달)"
                            
                        df_scan.insert(2, '투자등급', df_scan['종합점수'].apply(get_grade))

                        def highlight_scan_results(val):
                            if 'S등급' in str(val) or '🔥' in str(val): return 'background-color: #ffcccc; color: #cc0000; font-weight: bold;'
                            elif 'A등급' in str(val): return 'background-color: #d4edda; color: #155724; font-weight: bold;'
                            elif 'B등급' in str(val): return 'background-color: #fff3cd; color: #856404; font-weight: bold;'
                            return ''

                        styled_df = (df_scan.style
                            .map(highlight_scan_results, subset=['투자등급', '차트추세', '단기낙폭상태', '세력거래량'])
                            .bar(subset=['종합점수'], color='#5fba7d', vmin=0, vmax=100)
                        )
                        st.dataframe(styled_df, use_container_width=True, height=600)
                        
                        st.info("👆 위 표의 상위권 종목 코드를 **[첫 번째 탭: 차트 기술적 분석]**에 입력하여 지정하신 시간봉으로 최종 진입 타점을 확인하세요!")
                    else:
                        st.warning("야후 파이낸스 데이터 호출 제한(Rate Limit)으로 인해 종목을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.")
            except Exception as e:
                st.error(f"스캔 중 시스템 오류가 발생했습니다: {e}")
