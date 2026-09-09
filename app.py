import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측 및 온난화 추세 분석기")
st.write("기상청 서울 기온 데이터를 바탕으로 전체 기간과 최근 20년의 기온 상승 추세를 비교하고 미래 기온을 예측합니다.")

# 2. 데이터 불러오기 및 전처리
DATA_URL = "https://raw.githubusercontent.com/greatsong/modudata/bb860932644270ad1199f10d3e7670e30231bce4/data/seoul.csv"

@st.cache_data
def load_and_process_data():
    # 데이터 로드
    df = pd.read_csv(DATA_URL, encoding="utf-8")
    
    # 열 이름 공백 제거
    df.columns = df.columns.str.strip()
    
    # 날짜 데이터 변환 및 연도 추출
    df["날짜"] = pd.to_datetime(df["날짜"])
    df["연도"] = df["날짜"].dt.year
    
    # 평균기온 결측치 제거
    df = df.dropna(subset=["평균기온"])
    
    # 연도별 관측일 수 및 평균기온 계산
    yearly_df = df.groupby("연도").agg(
        관측일수=("평균기온", "count"),
        연평균기온=("평균기온", "mean")
    ).reset_index()
    
    # 기준 기간(2025년까지) 및 관측일 300일 미만 제외
    filtered_df = yearly_df[(yearly_df["연도"] <= 2025) & (yearly_df["관측일수"] >= 300)].copy()
    
    return filtered_df

data = load_and_process_data()

# 3. 전체 기간 회귀 분석
X_all = data["연도"].values
y_all = data["연평균기온"].values

slope_all, intercept_all = np.polyfit(X_all, y_all, 1)
corr_all = np.corrcoef(X_all, y_all)[0, 1]

start_year_all = int(X_all.min())
end_year_all = int(X_all.max())
num_years_all = len(X_all)

# 100년당 기온 상승량 (전체 기간)
rate_per_century_all = slope_all * 100

# 4. 최근 20년 데이터 필터링 및 회귀 분석
recent_20_df = data.tail(20)
X_recent = recent_20_df["연도"].values
y_recent = recent_20_df["연평균기온"].values

slope_recent, intercept_recent = np.polyfit(X_recent, y_recent, 1)
corr_recent = np.corrcoef(X_recent, y_recent)[0, 1]

start_year_recent = int(X_recent.min())
end_year_recent = int(X_recent.max())
num_years_recent = len(X_recent)

# 100년당 기온 상승량 (최근 20년 기준)
rate_per_century_recent = slope_recent * 100

# 5. 상승 속도 비교 (화면에 크게 표시)
st.markdown("---")
st.subheader("🔥 100년당 기온 상승 폭 비교")
st.caption("기울기(1년당 변화량)에 100을 곱하여, 해당 기간의 추세가 지속될 경우 100년간 오르는 기온을 비교합니다.")

col_rate1, col_rate2 = st.columns(2)

with col_rate1:
    st.metric(
        label=f"전체 기간 ({start_year_all}~{end_year_all}년)",
        value=f"{rate_per_century_all:+.2f} °C / 100년",
        help="전체 관측 데이터 기준 연간 기온 상승 폭에 100을 곱한 수치입니다."
    )

with col_rate2:
    difference = rate_per_century_recent - rate_per_century_all
    st.metric(
        label=f"최근 20년 ({start_year_recent}~{end_year_recent}년)",
        value=f"{rate_per_century_recent:+.2f} °C / 100년",
        delta=f"전체 기간 대비 {difference:+.2f} °C 더 빠름",
        delta_color="inverse",
        help="최근 20년간의 온난화 추세가 계속된다고 가정했을 때의 100년당 상승 폭입니다."
    )

st.markdown("---")

# 6. 상세 통계 정보 요약
st.subheader("📊 데이터 및 회귀 분석 요약")
col1, col2 = st.columns(2)

with col1:
    st.markdown(f"#### 🌐 전체 기간 모델 ({num_years_all}개 연도)")
    st.write(f"- **분석 기간**: {start_year_all}년 ~ {end_year_all}년")
    st.write(f"- **상관계수 (r)**: `{corr_all:.3f}`")
    st.write(f"- **연간 기온 변화율**: `{slope_all:+.4f} °C/년`")

with col2:
    st.markdown(f"#### ⚡ 최근 20년 모델 ({num_years_recent}개 연도)")
    st.write(f"- **분석 기간**: {start_year_recent}년 ~ {end_year_recent}년")
    st.write(f"- **상관계수 (r)**: `{corr_recent:.3f}`")
    st.write(f"- **연간 기온 변화율**: `{slope_recent:+.4f} °C/년`")

st.markdown("---")

# 7. 시각화 (Plotly)
st.subheader("📈 연평균 기온 산점도 및 두 회귀 직선 비교")

# 회귀선 생성용 좌표
line_x_all = np.linspace(start_year_all, end_year_all, 100)
line_y_all = slope_all * line_x_all + intercept_all

line_x_recent = np.linspace(start_year_recent, end_year_recent, 100)
line_y_recent = slope_recent * line_x_recent + intercept_recent

fig = go.Figure()

# 전체 관측 데이터 산점도
fig.add_trace(go.Scatter(
    x=data["연도"],
    y=data["연평균기온"],
    mode="markers",
    name="연평균 기온 (실제 관측치)",
    marker=dict(color="steelblue", size=7, opacity=0.6)
))

# 전체 기간 회귀선
fig.add_trace(go.Scatter(
    x=line_x_all,
    y=line_y_all,
    mode="lines",
    name=f"전체 회귀선 ({rate_per_century_all:+.2f}°C/100년)",
    line=dict(color="crimson", width=2.5)
))

# 최근 20년 회귀선
fig.add_trace(go.Scatter(
    x=line_x_recent,
    y=line_y_recent,
    mode="lines",
    name=f"최근 20년 회귀선 ({rate_per_century_recent:+.2f}°C/100년)",
    line=dict(color="orange", width=3, dash="dash")
))

fig.update_layout(
    xaxis_title="연도",
    yaxis_title="평균기온 (°C)",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# 8. 연도 선택 슬라이더 및 예상 기온 표시
st.subheader("🔮 미래 기온 예측하기")

selected_year = st.slider(
    "예측할 연도를 선택하세요 (범위: 1900년 ~ 2100년):",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 두 모델의 예측치 계산
pred_temp_all = slope_all * selected_year + intercept_all
pred_temp_recent = slope_recent * selected_year + intercept_recent

st.markdown(f"### 🗓️ **{selected_year}년** 서울의 예상 기온 비교")
pred_col1, pred_col2 = st.columns(2)

with pred_col1:
    st.metric(
        label="전체 기간 추세 기준 예상 기온",
        value=f"{pred_temp_all:.2f} °C",
        delta=f"연간 약 {slope_all:+.3f} °C 상승 반영"
    )

with pred_col2:
    st.metric(
        label="최근 20년 추세 기준 예상 기온",
        value=f"{pred_temp_recent:.2f} °C",
        delta=f"연간 약 {slope_recent:+.3f} °C 상승 반영"
    )
