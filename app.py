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
    .stApp {
        background-color: #F7F9FC;
    }
    .main-title {
        font-size: 46px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF512F, #F09819);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-top: 10px;
        padding-bottom: 0px;
    }
    .sub-title {
        text-align: center;
        color: #6c757d;
        font-size: 17px;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 24px;
        font-weight: 700;
        color: #2b2d42;
        border-left: 6px solid #FF512F;
        padding-left: 12px;
        margin-top: 40px;
        margin-bottom: 20px;
    }
    .metric-card {
        background: white;
        border-radius: 20px;
        padding: 25px 20px;
        box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.06);
        text-align: center;
        transition: transform 0.2s;
        border: 1px solid #f0f0f0;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .card-label {
        font-size: 15px;
        color: #8d99ae;
        font-weight: 600;
        margin-bottom: 5px;
    }
    .big-temp {
        font-size: 52px;
        font-weight: 900;
        margin: 5px 0;
    }
    .mid-temp {
        font-size: 36px;
        font-weight: 800;
        margin: 5px 0;
    }
    .card-sub {
        font-size: 13px;
        color: #adb5bd;
    }
    .prediction-box {
        background: linear-gradient(135deg, #FF512F 0%, #F09819 100%);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        color: white;
        box-shadow: 0px 10px 25px rgba(255, 81, 47, 0.3);
    }
    .prediction-box .big-temp {
        color: white;
        font-size: 65px;
    }
    .info-box {
        background: #EDF2FB;
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid #D6E0F5;
    }
    hr {
        margin-top: 30px;
        margin-bottom: 30px;
        border: none;
        border-top: 1px solid #e9ecef;
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
        연평균기온=("평균기온", "mean"),
        연평균최고기온=("최고기온", "mean"),
        연평균최저기온=("최저기온", "mean")
    ).reset_index()
    
    # 일교차 계산
    yearly_df["일교차"] = yearly_df["연평균최고기온"] - yearly_df["연평균최저기온"]
    
    # 기준 기간(2025년까지) 및 관측일 300일 미만 제외
    filtered_df = yearly_df[(yearly_df["연도"] <= 2025) & (yearly_df["관측일수"] >= 300)].copy()
    return filtered_df

data = load_and_process_data()

# 5. 전체 기간 회귀 분석 (평균기온 기준)
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

difference = rate_per_century_recent - rate_per_century_all

# 7. 최고/최저기온/일교차 회귀 분석 (전체 기간 기준)
slope_high, intercept_high = np.polyfit(X_all, data["연평균최고기온"].values, 1)
slope_low, intercept_low = np.polyfit(X_all, data["연평균최저기온"].values, 1)
slope_range, intercept_range = np.polyfit(X_all, data["일교차"].values, 1)

rate_high_century = slope_high * 100
rate_low_century = slope_low * 100
rate_range_century = slope_range * 100

# =========================================================
# 섹션 1: 100년당 기온 상승 폭 비교
# =========================================================
st.markdown('<p class="section-title">🔥 100년당 기온 상승 폭 비교</p>', unsafe_allow_html=True)
st.caption("기울기(1년당 변화량)에 100을 곱하여, 해당 기간의 추세가 지속될 경우 100년간 오르는 기온을 비교합니다.")

col_rate1, col_rate2 = st.columns(2)

with col_rate1:
    st.markdown(f"""
    <div class="metric-card">
        <p class="card-label">🌐 전체 기간 ({start_year_all}~{end_year_all}년)</p>
        <p class="big-temp" style="color:#457B9D;">{rate_per_century_all:+.2f}°C</p>
        <p class="card-sub">분석 연도 수: {num_years_all}개 해 · 상관계수(r): {corr_all:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

with col_rate2:
    color = "#E63946" if difference > 0 else "#2A9D8F"
    st.markdown(f"""
    <div class="metric-card">
        <p class="card-label">⚡ 최근 20년 ({start_year_recent}~{end_year_recent}년)</p>
        <p class="big-temp" style="color:{color};">{rate_per_century_recent:+.2f}°C</p>
        <p class="card-sub">분석 연도 수: {num_years_recent}개 해 · 상관계수(r): {corr_recent:.3f}</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if difference > 0:
    st.warning(f"⚠️ 최근 20년의 기온 상승 속도가 전체 기간 평균보다 **{difference:.2f}°C** 더 빠릅니다. 온난화가 가속화되고 있을 가능성이 있습니다.")
else:
    st.info(f"ℹ️ 최근 20년의 기온 상승 속도가 전체 기간 평균보다 **{abs(difference):.2f}°C** 더 완만합니다.")

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# 섹션 2: 최고/최저기온 & 일교차 분석 (신규 추가)
# =========================================================
st.markdown('<p class="section-title">🌡️ 최고·최저기온 & 일교차 추세 분석</p>', unsafe_allow_html=True)
st.caption(f"전체 기간({start_year_all}~{end_year_all}년) 기준, 최고기온·최저기온·일교차 각각의 100년당 변화 폭입니다.")

col_high, col_low, col_range = st.columns(3)

with col_high:
    st.markdown(f"""
    <div class="metric-card">
        <p class="card-label">🔴 최고기온</p>
        <p class="mid-temp" style="color:#E63946;">{rate_high_century:+.2f}°C</p>
        <p class="card-sub">/ 100년 기준</p>
    </div>
    """, unsafe_allow_html=True)

with col_low:
    st.markdown(f"""
    <div class="metric-card">
        <p class="card-label">🔵 최저기온</p>
        <p class="mid-temp" style="color:#457B9D;">{rate_low_century:+.2f}°C</p>
        <p class="card-sub">/ 100년 기준</p>
    </div>
    """, unsafe_allow_html=True)

with col_range:
    range_color = "#2A9D8F" if rate_range_century < 0 else "#F4A261"
    st.markdown(f"""
    <div class="metric-card">
        <p class="card-label">↔️ 일교차 (최고-최저)</p>
        <p class="mid-temp" style="color:{range_color};">{rate_range_century:+.2f}°C</p>
        <p class="card-sub">/ 100년 기준</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
if rate_range_century < 0:
    st.info(f"ℹ️ 일교차가 100년당 **{abs(rate_range_century):.2f}°C** 줄어드는 추세입니다. 최저기온이 최고기온보다 더 빠르게 오르고 있을 수 있습니다.")
else:
    st.info(f"ℹ️ 일교차가 100년당 **{rate_range_century:.2f}°C** 늘어나는 추세입니다. 최고기온이 최저기온보다 더 빠르게 오르고 있을 수 있습니다.")

# 최고/최저/일교차 그래프
fig_hl = go.Figure()

fig_hl.add_trace(go.Scatter(
    x=data["연도"], y=data["연평균최고기온"],
    mode="lines+markers", name="연평균 최고기온",
    line=dict(color="#E63946", width=2), marker=dict(size=5)
))
fig_hl.add_trace(go.Scatter(
    x=data["연도"], y=data["연평균기온"],
    mode="lines+markers", name="연평균 기온",
    line=dict(color="#8d99ae", width=2, dash="dot"), marker=dict(size=5)
))
fig_hl.add_trace(go.Scatter(
    x=data["연도"], y=data["연평균최저기온"],
    mode="lines+markers", name="연평균 최저기온",
    line=dict(color="#457B9D", width=2), marker=dict(size=5)
))

fig_hl.update_layout(
    xaxis_title="연도",
    yaxis_title="기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    template="plotly_white",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", size=13, color="#2b2d42"),
    margin=dict(l=10, r=10, t=60, b=10),
    height=450
)
fig_hl.update_xaxes(showgrid=True, gridcolor='#eeeeee')
fig_hl.update_yaxes(showgrid=True, gridcolor='#eeeeee')

st.plotly_chart(fig_hl, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# 섹션 3: 그래프 분석 (+ 무지개 테마 버튼)
# =========================================================
header_col, button_col = st.columns([4, 1])
with header_col:
    st.markdown('<p class="section-title">📈 연평균 기온 산점도 및 회귀 직선 비교</p>', unsafe_allow_html=True)
with button_col:
    st.write("")
    if "rainbow_mode" not in st.session_state:
        st.session_state.rainbow_mode = False
    if st.button("🌈 무지개 테마 ON/OFF", use_container_width=True):
        st.session_state.rainbow_mode = not st.session_state.rainbow_mode

rainbow_mode = st.session_state.rainbow_mode

if rainbow_mode:
    st.caption("🌈 무지개 모드: 점 색깔이 연도 순서를 나타냅니다. **보라색(과거) → 빨간색(최근)** 순서로 변합니다!")

line_x_all = np.linspace(start_year_all, end_year_all, 100)
line_y_all = slope_all * line_x_all + intercept_all

line_x_recent = np.linspace(start_year_recent, end_year_recent, 100)
line_y_recent = slope_recent * line_x_recent + intercept_recent

fig = go.Figure()

if rainbow_mode:
    fig.add_trace(go.Scatter(
        x=data["연도"], y=data["연평균기온"],
        mode="markers", name="연평균 기온 (관측치)",
        marker=dict(
            size=11,
            color=data["연도"],
            colorscale="Rainbow",
            showscale=True,
            colorbar=dict(title="연도"),
            line=dict(width=1, color='white'),
            opacity=0.85
        )
    ))
    line_color_all = "#00C2D1"
    line_color_recent = "#FF00E4"
else:
    fig.add_trace(go.Scatter(
        x=data["연도"], y=data["연평균기온"],
        mode="markers", name="연평균 기온 (관측치)",
        marker=dict(color="#457B9D", size=9, opacity=0.55, line=dict(width=1, color='white'))
    ))
    line_color_all = "#E63946"
    line_color_recent = "#F4A261"

fig.add_trace(go.Scatter(
    x=line_x_all, y=line_y_all,
    mode="lines", name=f"전체 회귀선 ({rate_per_century_all:+.2f}°C/100년)",
    line=dict(color=line_color_all, width=3)
))

fig.add_trace(go.Scatter(
    x=line_x_recent, y=line_y_recent,
    mode="lines", name=f"최근 20년 회귀선 ({rate_per_century_recent:+.2f}°C/100년)",
    line=dict(color=line_color_recent, width=4, dash="dash")
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, bgcolor='rgba(255,255,255,0)'),
    template="plotly_white",
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(family="Arial, sans-serif", size=13, color="#2b2d42"),
    margin=dict(l=10, r=10, t=60, b=10),
    height=500
)
fig.update_xaxes(showgrid=True, gridcolor='#eeeeee')
fig.update_yaxes(showgrid=True, gridcolor='#eeeeee')

st.plotly_chart(fig, use_container_width=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# 섹션 4: 상세 통계 정보 (전체 vs 최근 20년)
# =========================================================
st.markdown('<p class="section-title">📊 회귀 분석 상세 정보</p>', unsafe_allow_html=True)

info_col1, info_col2 = st.columns(2)
with info_col1:
    st.markdown(f"""
    <div class="metric-card" style="text-align:left;">
        <p class="card-label" style="text-align:center;">🌐 전체 기간 모델</p>
        <p>📅 분석 기간: <b>{start_year_all}년 ~ {end_year_all}년</b></p>
        <p>🔢 분석 연도 수: <b>{num_years_all}개</b></p>
        <p>📐 연간 기온 변화율: <b>{slope_all:+.4f} °C/년</b></p>
        <p>🔗 상관계수(r): <b>{corr_all:.3f}</b></p>
    </div>
    """, unsafe_allow_html=True)

with info_col2:
    st.markdown(f"""
    <div class="metric-card" style="text-align:left;">
        <p class="card-label" style="text-align:center;">⚡ 최근 20년 모델</p>
        <p>📅 분석 기간: <b>{start_year_recent}년 ~ {end_year_recent}년</b></p>
        <p>🔢 분석 연도 수: <b>{num_years_recent}개</b></p>
        <p>📐 연간 기온 변화율: <b>{slope_recent:+.4f} °C/년</b></p>
        <p>🔗 상관계수(r): <b>{corr_recent:.3f}</b></p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# 섹션 5: 미래 예측 (+ 실제 관측치 비교 기능 추가)
# =========================================================
st.markdown('<p class="section-title">🔮 연도별 예상 기온 확인하기</p>', unsafe_allow_html=True)

selected_year = st.slider(
    "예측할 연도를 선택하세요 (1900년 ~ 2100년)", 
    min_value=1900, max_value=2100, value=2026, step=1
)

pred_temp_all = slope_all * selected_year + intercept_all
pred_temp_recent = slope_recent * selected_year + intercept_recent

st.markdown("<br>", unsafe_allow_html=True)

pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.markdown(f"""
    <div class="prediction-box">
        <p style="font-size:18px; opacity:0.9;">🌐 전체 기간 추세 기준</p>
        <p class="big-temp">{pred_temp_all:.2f}°C</p>
        <p style="font-size:14px; opacity:0.85;">{selected_year}년 예상 평균기온 · 연간 {slope_all:+.3f}°C 변화 반영</p>
    </div>
    """, unsafe_allow_html=True)

with pred_col2:
    st.markdown(f"""
    <div class="prediction-box" style="background: linear-gradient(135deg, #F4A261 0%, #E9C46A 100%); box-shadow: 0px 10px 25px rgba(244, 162, 97, 0.4);">
        <p style="font-size:18px; opacity:0.9;">⚡ 최근 20년 추세 기준</p>
        <p class="big-temp">{pred_temp_recent:.2f}°C</p>
        <p style="font-size:14px; opacity:0.85;">{selected_year}년 예상 평균기온 · 연간 {slope_recent:+.3f}°C 변화 반영</p>
    </div>
    """, unsafe_allow_html=True)

# --- 실제 관측치와 비교 ---
st.markdown("<br>", unsafe_allow_html=True)

actual_row = data[data["연도"] == selected_year]

if not actual_row.empty:
    actual_temp = actual_row["연평균기온"].values[0]
    error_all = pred_temp_all - actual_temp
    error_recent = pred_temp_recent - actual_temp

    st.markdown(f"""
    <div class="info-box">
        <p class="card-label">📌 {selected_year}년 실제 관측된 연평균기온</p>
        <p class="mid-temp" style="color:#2b2d42;">{actual_temp:.2f}°C</p>
        <p class="card-sub">
            전체 기간 모델 오차: <b>{error_all:+.2f}°C</b> &nbsp;|&nbsp; 
            최근 20년 모델 오차: <b>{error_recent:+.2f}°C</b>
        </p>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="info-box">
        <p class="card-label">📌 {selected_year}년은(는) 실제 관측 데이터 범위({start_year_all}~{end_year_all}년) 밖입니다.</p>
        <p class="card-sub">위에 표시된 값은 회귀직선을 이용한 <b>예측치</b>입니다.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# =========================================================
# 섹션 6: 데이터 다운로드
# =========================================================
st.markdown('<p class="section-title">💾 전처리된 데이터 다운로드</p>', unsafe_allow_html=True)
st.write("분석에 사용된 연도별 평균·최고·최저기온 및 일교차 데이터를 CSV 파일로 내려받아 직접 검산해볼 수 있습니다.")

download_df = data[["연도", "관측일수", "연평균기온", "연평균최고기온", "연평균최저기온", "일교차"]].copy()
csv_data = download_df.to_csv(index=False, encoding="utf-8-sig")

st.download_button(
    label="📥 연도별 기온 데이터 CSV 다운로드",
    data=csv_data,
    file_name="seoul_yearly_temperature.csv",
    mime="text/csv",
    use_container_width=True
)

st.markdown("<br><br>", unsafe_allow_html=True)
