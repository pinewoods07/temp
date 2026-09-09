import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 기본 설정
st.set_page_config(
    page_title="서울 기온 예측기", 
    page_icon="🌡️", 
    layout="wide"
)

# 2. 커스텀 CSS 스타일 적용
st.markdown("""
    <style>
    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #FF4B4B;
        text-align: center;
        padding-bottom: 10px;
    }
    .sub-title {
        text-align: center;
        color: #555555;
        font-size: 18px;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #EAEAEA;
        border-radius: 15px;
        padding: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
        text-align: center;
    }
    .big-temp {
        font-size: 60px;
        font-weight: 900;
        color: #E63946;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F8F9FA;
        border-radius: 10px 10px 0px 0px;
        padding: 10px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #FF4B4B;
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# 3. 제목 영역
st.markdown('<p class="main-title">🌡️ 서울 연평균 기온 예측기</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">서울 관측 데이터를 기반으로 온난화 추세를 분석하고 미래 기온을 예측합니다.</p>', unsafe_allow_html=True)

# 4. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data():
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    df.columns = df.columns.str.strip()
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    df = df.dropna(subset=["평균기온"])
    
    yearly_df = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    filtered_df = yearly_df[(yearly_df["연도"] <= 2025) & (yearly_df["관측일수"] >= 300)].copy()
    return filtered_df

data = load_and_process_data()

# 5. 전체 기간 회귀 분석
X_all = data["연도"].values
y_all = data["연평균기온"].values

slope_all, intercept_all = np.polyfit(X_all, y_all, 1)
corr_all = np.corrcoef(X_all, y_all)[0, 1]

start_year_all = int(X_all.min())
end_year_all = int(X_all.max())
num_years_all = len(X_all)
rate_per_century_all = slope_all * 100

# 6. 최근 20년 데이터 필터링 및 회귀 분석
recent_20_df = data.tail(20)
X_recent = recent_20_df["연도"].values
y_recent = recent_20_df["연평균기온"].values

slope_recent, intercept_recent = np.polyfit(X_recent, y_recent, 1)
corr_recent = np.corrcoef(X_recent, y_recent)[0, 1]

start_year_recent = int(X_recent.min())
end_year_recent = int(X_recent.max())
num_years_recent = len(X_recent)
rate_per_century_recent = slope_recent * 100

# 7. 탭 구성
tab1, tab2, tab3 = st.tabs(["🔥 추세 비교", "📈 그래프 분석", "🔮 미래 예측"])

# --- TAB 1: 추세 비교 ---
with tab1:
    st.subheader("100년당 기온 상승 폭 비교")
    st.caption("기울기(1년당 변화량)에 100을 곱하여, 해당 기간의 추세가 지속될 경우 100년간 오르는 기온을 비교합니다.")
    
    col_rate1, col_rate2 = st.columns(2)

    with col_rate1:
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size:16px; color:gray;">🌐 전체 기간 ({start_year_all}~{end_year_all}년, {num_years_all}개 해)</p>
            <p class="big-temp">{rate_per_century_all:+.2f} °C</p>
            <p style="color:gray;">/ 100년 기준 · 상관계수(r): {corr_all:.3f}</p>
        </div>
        """, unsafe_allow_html=True)

    with col_rate2:
        difference = rate_per_century_recent - rate_per_century_all
        color = "#E63946" if difference > 0 else "#457B9D"
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size:16px; color:gray;">⚡ 최근 20년 ({start_year_recent}~{end_year_recent}년, {num_years_recent}개 해)</p>
            <p class="big-temp" style="color:{color};">{rate_per_century_recent:+.2f} °C</p>
            <p style="color:gray;">/ 100년 기준 · 상관계수(r): {corr_recent:.3f}</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    if difference > 0:
        st.warning(f"⚠️ 최근 20년의 기온 상승 속도가 전체 기간 평균보다 **{difference:.2f}°C** 더 빠릅니다. 온난화가 가속화되고 있을 가능성이 있습니다.")
    else:
        st.info(f"ℹ️ 최근 20년의 기온 상승 속도가 전체 기간 평균보다 **{abs(difference):.2f}°C** 더 완만합니다.")

# --- TAB 2: 그래프 분석 ---
with tab2:
    st.subheader("연평균 기온 산점도 및 회귀 직선 비교")
    
    line_x_all = np.linspace(start_year_all, end_year_all, 100)
    line_y_all = slope_all * line_x_all + intercept_all

    line_x_recent = np.linspace(start_year_recent, end_year_recent, 100)
    line_y_recent = slope_recent * line_x_recent + intercept_recent

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=data["연도"], y=data["연평균기온"],
        mode="markers", name="연평균 기온 (관측치)",
        marker=dict(color="#457B9D", size=8, opacity=0.6, line=dict(width=1, color='white'))
    ))

    fig.add_trace(go.Scatter(
        x=line_x_all, y=line_y_all,
        mode="lines", name=f"전체 회귀선 ({rate_per_century_all:+.2f}°C/100년)",
        line=dict(color="#E63946", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=line_x_recent, y=line_y_recent,
        mode="lines", name=f"최근 20년 회귀선 ({rate_per_century_recent:+.2f}°C/100년)",
        line=dict(color="#F4A261", width=4, dash="dash")
    ))

    fig.update_layout(
        xaxis_title="연도",
        yaxis_title="평균기온 (°C)",
        hovermode="x unified",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor='rgba(255,255,255,0.8)'),
        template="plotly_white",
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Arial, sans-serif", size=13),
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig, use_container_width=True)

# --- TAB 3: 미래 예측 ---
with tab3:
    st.subheader("연도를 선택해 예상 기온을 확인하세요")

    selected_year = st.slider(
        "예측할 연도 (1900년 ~ 2100년)", 
        min_value=1900, max_value=2100, value=2026, step=1
    )

    pred_temp_all = slope_all * selected_year + intercept_all
    pred_temp_recent = slope_recent * selected_year + intercept_recent

    st.markdown(f"<h3 style='text-align:center;'>🗓️ {selected_year}년 서울의 예상 평균기온</h3>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    pred_col1, pred_col2 = st.columns(2)

    with pred_col1:
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size:16px; color:gray;">전체 기간 추세 기준</p>
            <p class="big-temp">{pred_temp_all:.2f} °C</p>
            <p style="color:gray;">연간 {slope_all:+.3f}°C 변화 반영</p>
        </div>
        """, unsafe_allow_html=True)

    with pred_col2:
        st.markdown(f"""
        <div class="metric-card">
            <p style="font-size:16px; color:gray;">최근 20년 추세 기준</p>
            <p class="big-temp" style="color:#F4A261;">{pred_temp_recent:.2f} °C</p>
            <p style="color:gray;">연간 {slope_recent:+.3f}°C 변화 반영</p>
        </div>
        """, unsafe_allow_html=True)
