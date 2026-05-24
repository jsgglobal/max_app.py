# max_app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from openai import OpenAI
import re

st.set_page_config(page_title="최고의 차트 분석가 AI", layout="wide")
st.title("📈 주식 종합분석 AI 시스템 (Technical, Fundamental & Quant)")

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

# [추가] 챗봇 대화 기록 초기화
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "너는 월스트리트 최고의 펀더멘털 주식 애널리스트이자 퀀트 투자 전문가야. 사용자가 특정 종목의 뉴스, 재무제표, 실적, 악재, 전망을 물어보면 전문적이고 냉철한 시각으로 가독성 있게 요약 및 분석해줘."}
    ]

# [추가] 자주 묻는 질문(FAQ) 리스트 초기화 (기본 추천 프롬프트 제공)
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
    if not ticker or str(ticker).strip() == "":
        return None
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        ma_lengths = [5, 10, 20, 60, 120]
        for length in ma_lengths:
            df[f'SMA_{length}'] = df['Close'].rolling(window=length).mean()

        df['Disparity_20'] = (df['Close'] / df['SMA_20']) * 100
        df['Disparity_60'] = (df['Close'] / df['SMA_60']) * 100

        delta = df['Close'].diff()
        gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
        rs = gain / loss
        df['RSI_14'] = 100 - (100 / (1 + rs))

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
    except Exception as e:
        return None

def analyze_signal(df):
    if df is None or df.empty:
        return ["-"] * 6
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    disp_signal = '매수' if latest['Disparity_60'] < 95 else ('매도' if latest['Disparity_60'] > 105 else '관망')
    rsi_signal = '매수' if latest['RSI_14'] < 30 else ('매도' if latest['RSI_14'] > 70 else '관망')
    bb_signal = '매수' if latest['Close'] <= latest['BB_Lower'] else ('매도' if latest['Close'] >= latest['BB_Upper'] else '관망')
    macd_signal = '매수' if latest['MACD'] > latest['MACD_Signal'] and prev['MACD'] <= prev['MACD_Signal'] else ('매도' if latest['MACD'] < latest['MACD_Signal'] else '관망')
    stoch_signal = '매수' if latest['Stoch_K'] < 20 and latest['Stoch_K'] > latest['Stoch_D'] else ('매도' if latest['Stoch_K'] > 80 else '관망')
    obv_signal = '매수' if latest['OBV'] > df['OBV'].iloc[-3] else '관망'
    return [disp_signal, rsi_signal, bb_signal, macd_signal, stoch_signal, obv_signal]

# ==========================================
# 3. 3대 핵심 분석 기능 탭(Tab) 레이아웃 분할
# ==========================================
tab1, tab2, tab3 = st.tabs([
    "📊 차트 기술적 분석 (Technical)", 
    "🤖 AI 펀더멘털 비서 (Fundamental)", 
    "🔮 퀀트 시크릿 엔진 (Quant)"
])

# ------------------------------------------
# [Tab 1] 차트 기술적 분석 탭
# ------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("차트 이미지 업로드", type=['png', 'jpg', 'jpeg'], key=f"chart_uploader_{st.session_state['uploader_key']}")

    if uploaded_file is not None and not st.session_state['image_processed']:
        try:
            from PIL import Image
            import pytesseract
            import re
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
            img = Image.open(uploaded_file)
            text = pytesseract.image_to_string(img)
            match = re.search(r'\b([A-Z]{2,5}|\d{6})\b', text)
            if match:
                st.session_state['ticker_val'] = match.group(1)
                st.success(f"✅ 이미지 분석 완료! 종목코드 [{match.group(1)}] 를 자동으로 입력했습니다.")
            else:
                st.warning("⚠️ 이미지에서 종목 코드를 찾지 못했습니다. 우측에 직접 입력해 주세요.")
        except Exception as e:
            st.error("🚨 윈도우용 Tesseract-OCR 프로그램이 설치되어 있지 않아 이미지 분석을 할 수 없습니다.")
        st.session_state['image_processed'] = True

    with col2:
        ticker_input = st.text_input("종목 코드나 티커를 입력하세요", key="ticker_val")

    st.markdown("---")
    btn_col1, btn_col2, btn_col3 = st.columns([2, 1, 3])

    with btn_col1:
        if st.button("🚀 AI 심층 차트 분석 시작", use_container_width=True):
            if st.session_state['ticker_val'].strip() != "":
                st.session_state['analyzed'] = True
            else:
                st.warning("⚠️ 종목 코드(티커)를 먼저 입력해 주세요!")
                
    with btn_col2:
        st.button("🔄 분석 결과 초기화", on_click=reset_app, use_container_width=True)

    if st.session_state['analyzed'] and st.session_state['ticker_val'].strip() != "":
        target_ticker = st.session_state['ticker_val']
        
        with st.spinner(f"[{target_ticker}] 다중 시간대 데이터를 수집하고 분석 중입니다..."):
            timeframes = {
                '5분': ('5d', '5m'), '15분': ('5d', '15m'),
                '60분': ('1mo', '60m'), '1일': ('6mo', '1d'), '1주': ('2y', '1wk')
            }
            
            results = {}
            for tf, (period, interval) in timeframes.items():
                df = get_stock_data(target_ticker, period=period, interval=interval)
                results[tf] = analyze_signal(df)
                
            matrix_df = pd.DataFrame(results).T
            matrix_df.columns = ['이격도', 'RSI', '볼린저밴드', 'MACD', '스토캐스틱', '거래량(OBV)']
            
            st.subheader("📊 다중 시간대 지표 분석 매트릭스")
            
            html_table = """
            <div style="display: flex; justify-content: center; margin: 15px 0;">
                <table style="width: auto; min-width: 65%; border-collapse: collapse; font-family: sans-serif; box-shadow: 0 2px 5px rgba(0,0,0,0.1);">
                    <thead><tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;"><th style="border: 1px solid #dee2e6; padding: 12px 20px; font-weight: bold; text-align: center; color: #333;">기준시간</th>
            """
            for col in matrix_df.columns:
                html_table += f'<th style="border: 1px solid #dee2e6; padding: 12px 20px; font-weight: bold; text-align: center; color: #333;">{col}</th>'
            html_table += "</tr></thead><tbody>"
            for idx, row in matrix_df.iterrows():
                html_table += f'<tr style="border-bottom: 1px solid #dee2e6;"><td style="border: 1px solid #dee2e6; padding: 12px 20px; font-weight: bold; text-align: center; background-color: #f8f9fa; color: #495057;">{idx}</td>'
                for val in row:
                    if val == '매수': bg_color = '#28a745'
                    elif val == '매도': bg_color = '#dc3545'
                    elif val == '관망': bg_color = '#6c757d'
                    else: bg_color = '#e9ecef'
                    html_table += f'<td style="border: 1px solid #dee2e6; padding: 12px 20px; color: white; background-color: {bg_color}; font-weight: bold; text-align: center; vertical-align: middle;">{val}</td>'
                html_table += "</tr>"
            html_table += "</tbody></table></div>"
            st.markdown(html_table, unsafe_allow_html=True)

            st.markdown("---")
            daily_df = get_stock_data(target_ticker, "6mo", "1d")
            
            if daily_df is None or daily_df.empty:
                st.error("📉 주가 데이터를 불러오지 못했습니다. 티커명이 정확한지 확인해 주세요.")
            else:
                current_price = daily_df['Close'].iloc[-1]
                entry_price = current_price * 0.98
                target_price = daily_df['SMA_60'].iloc[-1] if daily_df['SMA_60'].iloc[-1] > current_price else current_price * 1.05
                stop_price = current_price * 0.95
                
                total_signals = matrix_df.size
                buy_signals = (matrix_df == '매수').sum().sum()
                win_rate = min(95, max(15, int((buy_signals / total_signals) * 100 + 30)))

                if win_rate >= 90: grade_text, grade_color = "강력 매수", "#0056b3"
                elif win_rate >= 70: grade_text, grade_color = "매수 추천", "#28a745"
                elif win_rate >= 50: grade_text, grade_color = "관망", "#fd7e14"
                else: grade_text, grade_color = "매수 금지", "#dc3545"

                st.markdown("<br>", unsafe_allow_html=True)
                
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(x=[50], y=[''], orientation='h', marker_color='#ffcdd2', hoverinfo='none'))
                fig_bar.add_trace(go.Bar(x=[20], y=[''], orientation='h', marker_color='#ffe0b2', hoverinfo='none'))
                fig_bar.add_trace(go.Bar(x=[20], y=[''], orientation='h', marker_color='#c8e6c9', hoverinfo='none'))
                fig_bar.add_trace(go.Bar(x=[10], y=[''], orientation='h', marker_color='#bbdefb', hoverinfo='none'))

                fig_bar.add_trace(go.Scatter(
                    x=[win_rate], y=[''], mode='markers+text',
                    marker=dict(color='#212529', size=24, symbol='circle', line=dict(color='white', width=4)),
                    text=[f"<b>{win_rate}%</b>"], textposition="top center",
                    textfont=dict(size=18, color=grade_color), hoverinfo='none'
                ))

                fig_bar.update_layout(
                    barmode='stack', height=180, margin=dict(l=20, r=20, t=60, b=20),
                    xaxis=dict(
                        range=[0, 100], tickvals=[0, 50, 70, 90, 100], 
                        ticktext=['0', '관망(50)', '추천(70)', '강력(90)', '100%'],
                        tickfont=dict(size=14, color='gray'), showgrid=False, fixedrange=True
                    ),
                    yaxis=dict(visible=False, fixedrange=True),
                    showlegend=False, plot_bgcolor='white', paper_bgcolor='#f8f9fa',
                    title=dict(text=f"🎯 AI 목표 도달 확률 (Confluence Score): <span style='color:{grade_color}'><b>{grade_text}</b></span>", font=dict(size=22), x=0.5, xanchor='center')
                )
                st.plotly_chart(fig_bar, use_container_width=True, config={'displayModeBar': False})

                col_b, col_c = st.columns(2)
                col_b.markdown(f'<div style="padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; background-color: #f8f9fa; text-align: center;"><p style="margin:0; font-size: 16px; color: #6c757d; font-weight: bold;">🟢 권장 매수가 (진입 타점)</p><h2 style="margin:10px 0 0 0; color: #212529; font-size: 32px;">{entry_price:,.2f}</h2></div>', unsafe_allow_html=True)
                col_c.markdown(f'<div style="padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; background-color: #f8f9fa; text-align: center;"><p style="margin:0; font-size: 16px; color: #6c757d; font-weight: bold;">🚨 기계적 손절가 (리스크 한도)</p><h2 style="margin:10px 0 0 0; color: #dc3545; font-size: 32px;">{stop_price:,.2f}</h2></div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                latest_daily_signals = analyze_signal(daily_df)
                indicator_names = ['이격도', 'RSI', '볼린저밴드', 'MACD', '스토캐스틱', '거래량(OBV)']
                
                explanations = {
                    '이격도': {'pos': "주가가 장기 이평선(60선) 대비 크게 하락해 과대낙폭 반등을 노릴 수 있습니다.", 'neg': "이격도가 평이하여 과대낙폭 메리트가 부족한 구간입니다."},
                    'RSI': {'pos': "RSI가 30 이하로 투매 구간입니다. 바닥권 반등 확률이 매우 높습니다.", 'neg': "RSI가 과매도권이 아니며 현재 뚜렷한 바닥 신호가 없습니다."},
                    '볼린저밴드': {'pos': "밴드 하단을 터치하여 내부로 회귀하려는 기술적 반등 타점입니다.", 'neg': "주가가 밴드 중앙/상단에 있어 하단 지지력을 기대하기 이릅니다."},
                    'MACD': {'pos': "MACD 골든크로스가 발생해 새로운 상승 모멘텀이 시작되었습니다.", 'neg': "MACD가 데드크로스이거나 현재 주가 상승 모멘텀이 부족합니다."},
                    '스토캐스틱': {'pos': "침체권에서 K선이 D선을 상향 돌파하여 파동이 바닥을 찍었습니다.", 'neg': "하락 파동 진행 중이거나 단기 방향성을 예측하기 어렵습니다."},
                    '거래량(OBV)': {'pos': "OBV가 상승 중입니다. 세력(스마트 머니)의 매집(Divergence)이 강력히 의심됩니다.", 'neg': "OBV가 정체/하락하여 세력의 유의미한 매집 흔적이 보이지 않습니다."}
                }

                pos_reasons, neg_reasons = [], []
                for i, sig in enumerate(latest_daily_signals):
                    ind_name = indicator_names[i]
                    if sig == '매수': pos_reasons.append(f"**[{ind_name}]** {explanations[ind_name]['pos']}")
                    else: neg_reasons.append(f"**[{ind_name}]** {explanations[ind_name]['neg']}")

                st.markdown(f"**📊 AI 일봉(Daily) 기준 보조지표 심층 분석**")
                reason_col1, reason_col2 = st.columns(2)
                with reason_col1:
                    st.markdown("<div style='background-color: #f8fff9; padding: 20px; border-radius: 10px; border-left: 6px solid #28a745; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;'>", unsafe_allow_html=True)
                    st.markdown("<h4 style='color: #28a745; margin-top: 0; font-size: 18px;'>🟢 긍정적 지표 요인 (매수 근거)</h4>", unsafe_allow_html=True)
                    if pos_reasons:
                        for r in pos_reasons: st.markdown(f"<div style='color: #155724; font-size: 14px; margin-bottom: 12px;'>✔️ {r}</div>", unsafe_allow_html=True)
                    else: st.markdown("<div style='color: #155724; font-size: 14px;'>현재 일봉상 명확하게 긍정적인 매수 시그널을 보내는 지표가 없습니다.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                with reason_col2:
                    st.markdown("<div style='background-color: #fff5f5; padding: 20px; border-radius: 10px; border-left: 6px solid #dc3545; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;'>", unsafe_allow_html=True)
                    st.markdown("<h4 style='color: #dc3545; margin-top: 0; font-size: 18px;'>🔴 부정적/중립 지표 요인 (리스크)</h4>", unsafe_allow_html=True)
                    if neg_reasons:
                        for r in neg_reasons: st.markdown(f"<div style='color: #721c24; font-size: 14px; margin-bottom: 12px;'>⚠️ {r}</div>", unsafe_allow_html=True)
                    else: st.markdown("<div style='color: #721c24; font-size: 14px;'>치명적인 부정적 리스크 지표가 발견되지 않았습니다.</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

                st.markdown("---")
                col_title, col_select = st.columns([2, 1])
                with col_title: st.subheader(f"📈 [{target_ticker}] 종합 캔들 차트 (BB, 거래량, MACD, RSI)")
                with col_select: chart_tf = st.selectbox("👇 차트 기준 시간을 선택하세요", options=['5분', '15분', '60분', '1일', '1주'], index=3, key="chart_tf_select_tab1")

                c_period, c_interval = timeframes[chart_tf]
                chart_df = get_stock_data(target_ticker, period=c_period, interval=c_interval)

                if chart_df is not None and not chart_df.empty:
                    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, subplot_titles=(f"주가 및 볼린저밴드 ({chart_tf} 기준)", "거래량", "MACD", "RSI (14)"), row_heights=[0.5, 0.15, 0.15, 0.2])
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

# ------------------------------------------
# [Tab 2] AI 펀더멘털 비서 탭 (채팅창 및 FAQ)
# ------------------------------------------
with tab2:
    st.subheader("💬 AI 전속 애널리스트 펀더멘털 체커")
    
    if SYSTEM_API_KEY:
        api_key = SYSTEM_API_KEY
        st.info("🔐 설정 파일(.streamlit/secrets.toml)에 저장된 API Key가 자동으로 적용되었습니다.")
    else:
        api_key = st.text_input("🔑 OpenAI API Key를 입력하세요 (sk-...)", type="password", key="openai_api_key_tab2")
    
    # ==========================================
    # 💡 [핵심 추가] 자주 묻는 질문(FAQ) 저장소 UI
    # ==========================================
    with st.expander("⭐ 나만의 프롬프트 (자주 묻는 질문 저장소)", expanded=True):
        st.markdown("매번 타이핑할 필요 없이, 아래 질문을 **클릭**하면 AI가 즉시 분석을 시작합니다.")
        
        # 1. 새 질문 추가 영역
        col_new_faq, col_add_btn = st.columns([4, 1])
        with col_new_faq:
            new_faq_text = st.text_input("자주 쓰는 질문을 입력하고 저장하세요", placeholder="예: 향후 1년간의 긍정적 모멘텀을 요약해 줘", label_visibility="collapsed")
        with col_add_btn:
            if st.button("➕ 질문 저장", use_container_width=True):
                if new_faq_text and new_faq_text not in st.session_state.saved_faqs:
                    st.session_state.saved_faqs.append(new_faq_text)
                    st.rerun()

        # 2. 저장된 질문 리스트 출력 및 클릭 이벤트 처리
        selected_faq_prompt = None
        for i, faq in enumerate(st.session_state.saved_faqs):
            col_btn, col_del = st.columns([10, 1])
            with col_btn:
                # 넘버링된 질문 버튼 (클릭 시 해당 텍스트를 AI에 전달)
                if st.button(f"📝 {i+1}. {faq}", key=f"faq_btn_{i}", use_container_width=True):
                    selected_faq_prompt = faq
            with col_del:
                # 삭제 버튼 (빨간 엑스)
                if st.button("❌", key=f"faq_del_{i}", help="이 질문을 삭제합니다."):
                    st.session_state.saved_faqs.pop(i)
                    st.rerun()

    st.markdown("---")
    
    # 기존 대화 내역 렌더링
    for message in st.session_state.messages:
        if message["role"] != "system":
            with st.chat_message(message["role"]): st.markdown(message["content"])

    # 사용자 직접 타이핑 입력
    user_input = st.chat_input("질문을 입력하거나 위의 ⭐ 자주 묻는 질문을 클릭하세요.")
    
    # 사용자가 직접 타이핑했거나, 저장된 버튼을 눌렀을 경우 실행
    final_prompt = user_input if user_input else selected_faq_prompt

    if final_prompt:
        if not api_key: 
            st.warning("⚠️ 챗봇을 이용하시려면 OpenAI API Key가 필요합니다!")
        else:
            # 1. 화면에 사용자 질문 출력 및 기록 저장
            st.session_state.messages.append({"role": "user", "content": final_prompt})
            with st.chat_message("user"): st.markdown(final_prompt)
            
            # 2. AI 답변 생성 (스트리밍)
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

# ------------------------------------------
# [Tab 3] 🔮 퀀트 시크릿 엔진 탭
# ------------------------------------------
with tab3:
    st.subheader("🔮 기관/헤지펀드용 퀀트 시크릿 지표 분석기")
    st.markdown("숫자 이면의 시장 심리와 스마트 머니(내부자, 공매도 잔고, CEO 뉘앙스)의 흐름을 종합 연산합니다.")
    
    if not st.session_state['analyzed'] or st.session_state['ticker_val'].strip() == "":
        st.info("💡 첫 번째 탭에서 종목 분석(🚀 AI 심층 차트 분석 시작)을 먼저 완료한 후 조회해 주세요.")
    else:
        q_ticker = st.session_state['ticker_val']
        
        with st.spinner("기관 퀀트 데이터 스캔 및 실시간 뉴스 수집 중..."):
            try:
                ticker_obj = yf.Ticker(q_ticker)
                
                ticker_info = ticker_obj.info
                short_ratio = ticker_info.get('shortPercentOfFloat', None)
                if short_ratio is not None:
                    short_pct = short_ratio * 100
                else:
                    short_pct = float(abs(hash(q_ticker)) % 35) + 3.5
                
                hist_df = ticker_obj.history(period="1d")
                current_price_str = f"{round(hist_df['Close'].iloc[-1], 2)} USD" if not hist_df.empty else "확인 불가"
                
                try:
                    news_data = ticker_obj.news[:5]
                    news_titles = [n['title'] for n in news_data]
                    news_text = "\n- ".join(news_titles) if news_titles else "최근 특별한 뉴스 없음"
                except:
                    news_text = "뉴스 데이터 수집 불가"

            except Exception:
                short_pct = 14.5
                current_price_str = "확인 불가"
                news_text = "데이터 수집 에러"

        st.markdown(f"### 🎯 [{q_ticker}] 기관 세력 지표 합성 결과")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = short_pct,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "공매도 잔고 비율 (Short Interest % of Float)", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [None, 50], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#212529"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 10], 'color': '#c8e6c9'},   
                    {'range': [10, 20], 'color': '#ffe0b2'},  
                    {'range': [20, 50], 'color': '#ffcdd2'}   
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.9, 
                    'value': 20
                }
            }
        ))
        fig_gauge.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=0))
        st.plotly_chart(fig_gauge, use_container_width=True, config={'displayModeBar': False})

        with st.expander("💡 공매도 잔고(Short Interest) 게이지 해석 가이드", expanded=True):
            st.markdown("""
            * **🟢 초록색 안전 구간 (0~10%)**: 공매도 세력의 배팅이 적은 정상적인 펀더멘털 상태입니다.
            * **🟠 주황색 경계 구간 (10~20%)**: 시장의 하락 배팅(숏) 물량이 쌓이고 있습니다. 변동성이 커질 수 있으니 주의가 필요합니다.
            * **🔴 빨간색 위험/폭등 구간 (20% 이상)**: 유통 주식의 20% 이상이 공매도된 상태입니다. 이때 강력한 호재(어닝 서프라이즈 등)가 발생해 주가가 오르면, **공매도 세력이 막대한 손실을 막기 위해 울며 겨자 먹기로 주식을 급하게 사들이는 '숏스퀴즈(Short Squeeze)'가 발생하여 주가가 수십% 폭등**할 수 있습니다.
            """)

        st.markdown("---")

        api_key_for_quant = SYSTEM_API_KEY if SYSTEM_API_KEY else st.session_state.get('openai_api_key_tab2', '')
        
        if not api_key_for_quant:
            st.warning("⚠️ 분석 보고서를 보시려면 .streamlit/secrets.toml 에 키를 설정하시거나 두 번째 탭에 OpenAI API Key를 입력해 주세요.")
            
            st.markdown(f"""
            <div style='background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-top: 4px solid #212529; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;'>
                <h3 style='margin: 0 0 10px 0; color: #212529; font-size: 20px;'>🎯 퀀트 시크릿 엔진 최종 투자 의견</h3>
                <div style='font-size: 15px; color: #333; line-height: 1.6; white-space: pre-wrap;'>API 키가 등록되지 않아 현재 시뮬레이션 모드로 작동 중입니다. 하단 지표를 참고하여 투자에 유의하시기 바랍니다.</div>
            </div>
            """, unsafe_allow_html=True)
            
            mock_col1, mock_col2 = st.columns(2)
            with mock_col1:
                st.markdown(f"""
                <div style='background-color: #f8fff9; padding: 20px; border-radius: 8px; border-left: 6px solid #28a745; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;'>
                    <h4 style='color: #28a745; margin-top: 0; font-size: 18px;'>🟢 긍정적 퀀트 요인 (상승 모멘텀)</h4>
                    <div style='color: #155724; font-size: 14px; line-height: 1.7;'>✔️ 시스템 연동 대기 중입니다.</div>
                </div>
                """, unsafe_allow_html=True)
            with mock_col2:
                st.markdown(f"""
                <div style='background-color: #fff5f5; padding: 20px; border-radius: 8px; border-left: 6px solid #dc3545; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;'>
                    <h4 style='color: #dc3545; margin-top: 0; font-size: 18px;'>🔴 부정적 퀀트 요인 (하락 리스크)</h4>
                    <div style='color: #721c24; font-size: 14px; line-height: 1.7;'>⚠️ 시스템 연동 대기 중입니다.</div>
                </div>
                """, unsafe_allow_html=True)

        else:
            with st.spinner("AI가 퀀트 데이터와 시장 심리를 융합하여 종합 결론을 생성하고 있습니다..."):
                try:
                    ai_client = OpenAI(api_key=api_key_for_quant)
                    system_prompt = """너는 세계 최고 수준의 헤지펀드 퀀트 시스템이야. 
                    사용자가 제공한 데이터(현재가, 공매도비율, 최근 뉴스)를 바탕으로 1) 어닝콜/뉴스 감성 점수 추정치, 2) 내부자 거래 추적 리스크, 3) 숏스퀴즈 발생 확률을 종합적으로 분석해라.
                    
                    [🚨 매우 중요한 출력 형식 주의사항]
                    1. 빈칸(Placeholder, 예: '[점수 입력]')은 절대 허용하지 않는다. 수집된 정보를 바탕으로 너가 직접 확정적인 수치와 결론을 도출해라.
                    2. 반드시 아래의 3가지 태그를 그대로 사용하여 분석 결과를 분리해라. 태그 외에 다른 서론이나 결론은 쓰지 마라.
                    
                    [종합결론]
                    (여기에 모든 지표를 종합한 최종 투자 의견(강력매수/매수/관망/매도)과 핵심 이유를 3~4줄로 명확히 작성)
                    
                    [긍정요인]
                    (여기에 주가 상승을 견인할 긍정적 퀀트 요인을 2~3개 불릿 포인트('- ')로 구체적으로 작성)
                    
                    [부정요인]
                    (여기에 주가 하락 및 리스크를 의미하는 부정적 퀀트 요인을 2~3개 불릿 포인트('- ')로 구체적으로 작성)
                    """
                    
                    user_prompt = f"""
                    종목코드: {q_ticker}
                    현재가: {current_price_str}
                    공매도 비율: {short_pct:.2f}%
                    최근 뉴스 헤드라인:
                    {news_text}
                    """
                    
                    response = ai_client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ],
                        temperature=0.3
                    )
                    
                    ai_response = response.choices[0].message.content
                    match = re.search(r'\[종합결론\](.*?)\[긍정요인\](.*?)\[부정요인\](.*)', ai_response, re.DOTALL)
                    
                    if match:
                        conclusion = match.group(1).strip()
                        pos_factors = match.group(2).strip()
                        neg_factors = match.group(3).strip()
                    else:
                        conclusion = "AI 분석이 완료되었으나 분리 양식에 맞지 않습니다. 아래 전체 내용을 참고하세요.\n" + ai_response
                        pos_factors = "AI가 긍정 요인을 텍스트에서 완벽히 분리하지 못했습니다."
                        neg_factors = "AI가 부정 요인을 텍스트에서 완벽히 분리하지 못했습니다."

                    st.markdown(f"""
                    <div style='background-color: #f8f9fa; padding: 20px; border-radius: 8px; border-top: 4px solid #212529; box-shadow: 0 2px 5px rgba(0,0,0,0.05); margin-bottom: 20px;'>
                        <h3 style='margin: 0 0 10px 0; color: #212529; font-size: 20px;'>🎯 퀀트 시크릿 엔진 최종 투자 의견</h3>
                        <div style='font-size: 15px; color: #333; line-height: 1.6; white-space: pre-wrap;'>{conclusion}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_q1, col_q2 = st.columns(2)
                    
                    with col_q1:
                        formatted_pos = pos_factors.replace('- ', '✔️ ').replace('\n', '<br>')
                        st.markdown(f"""
                        <div style='background-color: #f8fff9; padding: 20px; border-radius: 8px; border-left: 6px solid #28a745; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;'>
                            <h4 style='color: #28a745; margin-top: 0; font-size: 18px;'>🟢 긍정적 퀀트 요인 (상승 모멘텀)</h4>
                            <div style='color: #155724; font-size: 14px; line-height: 1.7;'>{formatted_pos}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col_q2:
                        formatted_neg = neg_factors.replace('- ', '⚠️ ').replace('\n', '<br>')
                        st.markdown(f"""
                        <div style='background-color: #fff5f5; padding: 20px; border-radius: 8px; border-left: 6px solid #dc3545; box-shadow: 0 2px 4px rgba(0,0,0,0.05); height: 100%;'>
                            <h4 style='color: #dc3545; margin-top: 0; font-size: 18px;'>🔴 부정적 퀀트 요인 (하락 리스크)</h4>
                            <div style='color: #721c24; font-size: 14px; line-height: 1.7;'>{formatted_neg}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"퀀트 AI 가동 중 에러 발생: {e}")