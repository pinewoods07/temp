import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 기본 설정
st.set_page_config(page_title="서울 기온 예측기", layout="wide")
st.title("🌡️ 서울 연평균 기온 예측기")
st.write("기상청 서울 기온 데이터를 바탕으로 과거의 기온 변화 추세를 분석하고 미래 기온을 예측합니다.")

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

# 3. 회귀 직선 및 상관계수 계산
X = data["연도"].values
y = data["연평균기온"].values

# 1차 선형 회귀 (기울기와 절편 계산)
slope, intercept = np.polyfit(X, y, 1)

# 상관계수 계산
correlation = np.corrcoef(X, y)[0, 1]

# 회귀 모델의 시작 연도, 끝 연도, 총 연도 개수
start_year = int(X.min())
end_year = int(X.max())
num_years = len(X)

# 4. 분석 요약 정보 표시
st.subheader("📊 데이터 분석 요약")
col1, col2, col3, col4 = st.columns(4)
col1.metric("분석 대상 연도 수", f"{num_years}개 해")
col2.metric("시작 연도", f"{start_year}년")
col3.metric("끝 연도", f"{end_year}년")
col4.metric("연도-기온 상관계수(r)", f"{correlation:.3f}")

st.markdown("---")

# 5. 시각화 (Plotly)
st.subheader("📈 연평균 기온 산점도 및 선형 회귀 직선")

# 회귀선 x 좌표 (시작 연도부터 끝 연도까지)
line_x = np.linspace(start_year, end_year, 100)
line_y = slope * line_x + intercept

fig = go.Figure()

# 실제 데이터 산점도 추가
fig.add_trace(go.Scatter(
    x=data["연도"],
    y=data["연평균기온"],
    mode="markers",
    name="연평균 기온 (관측치)",
    marker=dict(color="royalblue", size=7, opacity=0.7)
))

# 회귀 직선 추가
fig.add_trace(go.Scatter(
    x=line_x,
    y=line_y,
    mode="lines",
    name=f"회귀 직선 (기울기: {slope:.4f})",
    line=dict(color="crimson", width=2.5)
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

# 6. 연도 선택 슬라이더 및 예상 기온 표시
st.subheader("🔮 미래 기온 예측하기")

selected_year = st.slider(
    "예측할 연도를 선택하세요 (범위: 1900년 ~ 2100년):",
    min_value=1900,
    max_value=2100,
    value=2026,
    step=1
)

# 예측값 계산 (y = ax + b)
predicted_temp = slope * selected_year + intercept

st.markdown(f"### 🗓️ **{selected_year}년** 서울의 예상 평균기온")
st.metric(
    label="예상 기온",
    value=f"{predicted_temp:.2f} °C",
    delta=f"회귀식 추세: 연간 약 {slope:.3f} °C 상승"
)
