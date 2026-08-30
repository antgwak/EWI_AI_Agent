"""
카드사 EWI 대시보드 (Streamlit)
================================
실행:
    streamlit run app.py

구성:
  - 사이드바: 카드사 / 분기 선택
  - KPI: 현재 EWI 지표값, 경고 임계값, 판정 결과
  - 차트1: 해당 회사 EWI 시계열 (임계값 라인 + 경고 분기 강조)
  - 차트2: EWI를 구성한 4대 핵심지표 위험기여도(z-score) 분해
  - AI 리포트: 버튼 클릭 시 LangGraph 에이전트를 실시간 호출해 3단락 리포트 생성.
    같은 (회사, 분기) 조합은 세션 동안 결과를 캐싱해 재생성 없이 재사용
    (Groq 무료 티어 RPM/TPM 절약 + 반응 속도 개선 목적)
"""
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from agent_graph import build_graph
from config import EWI_STD_MULTIPLIER
from ewi import EWI_COLUMN, get_ewi_level, load_panel_data

st.set_page_config(page_title="카드사 리스크조기경보(EWI) 대시보드", layout="wide")

# ----------------------------------------------------------------
# selectbox 스타일 조정
#   - 기본 selectbox는 클릭 시 텍스트 입력(타이핑 필터링)이 가능하고 포커스 시
#     빨간 테두리가 뜨는데, 이를 막고 "클릭 → 목록에서 고르기"만 되도록 고정한다.
# ----------------------------------------------------------------
st.markdown(
    """
    <style>
    /* 포커스 시 빨간 테두리/그림자 제거 */
    div[data-baseweb="select"] > div {
        border-color: #d0d0d0 !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] > div:focus-within {
        border-color: #d0d0d0 !important;
        box-shadow: none !important;
    }
    /* 내부 입력창을 통한 타이핑(검색 필터링) 비활성화 - 클릭으로 목록만 열리게 함 */
    div[data-baseweb="select"] input {
        caret-color: transparent !important;
        pointer-events: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _clean_report_text(text: str) -> str:
    """LLM이 간혹 실제 줄바꿈 대신 '<br>' 같은 HTML 태그를 그대로 출력하는 경우가 있어,
    st.markdown()이 렌더링 전에 마크다운 줄바꿈(두 칸 공백 + 개행)으로 치환한다."""
    return re.sub(r"<br\s*/?>", "  \n", text, flags=re.IGNORECASE)


def _quarter_sort_key(q: str) -> tuple[int, int]:
    """'2025/Q3' -> (2025, 3) — 분기 문자열을 시간순 정렬하기 위한 키."""
    year, q_num = q.split("/Q")
    return int(year), int(q_num)


# ----------------------------------------------------------------
# 무거운 리소스 캐싱
#   - get_panel_data: 엑셀 로딩 (파일이 안 바뀌는 한 재사용)
#   - get_agent_graph: LangGraph 컴파일 + 임베딩모델/ChromaDB 연결 (앱 실행 중 1회만)
# ----------------------------------------------------------------
@st.cache_data
def get_panel_data() -> pd.DataFrame:
    return load_panel_data()


@st.cache_resource
def get_agent_graph():
    return build_graph()


df = get_panel_data()
graph = get_agent_graph()

if "report_cache" not in st.session_state:
    st.session_state.report_cache = {}  # {(company, quarter): report_text}

# ----------------------------------------------------------------
# 사이드바 : 회사 / 분기 선택
# ----------------------------------------------------------------
st.sidebar.header("조회 조건")
companies = sorted(df["회사"].unique())
company = st.sidebar.selectbox("카드사", companies)

quarters = sorted(
    df[df["회사"] == company]["분기"].unique(),
    key=_quarter_sort_key,
    reverse=True,  # 최신순(내림차순) — 전체 분기 옵션이 다 보이도록 필터 없이 정렬만 적용
)
quarter = st.sidebar.selectbox("분기", quarters, index=0)  # 목록 첫 번째 = 최신 분기

# ----------------------------------------------------------------
# EWI 판정 (결정론적 계산, 빠르므로 매 조회마다 다시 계산)
# ----------------------------------------------------------------
ewi_info = get_ewi_level(df, company, quarter)
level = ewi_info["level"]

st.title(f"📊 {company} 리스크조기경보(EWI) 대시보드")
st.caption(f"{quarter} 기준 · 경고 임계값 = 회사 자체 평균 - {EWI_STD_MULTIPLIER}×표준편차")

# ----------------------------------------------------------------
# KPI 카드
# ----------------------------------------------------------------
col1, col2, col3 = st.columns(3)
col1.metric("EWI 지표값", f"{ewi_info['current_value']:.4f}")
col2.metric("경고 임계값", f"{ewi_info['threshold']:.4f}")
col3.metric("판정", "🔴 경고" if level == "경고" else "🟢 비경고")

st.divider()  # 이후 모든 섹션 사이 간격을 동일한 구분선으로 통일

# ----------------------------------------------------------------
# 차트 1 : EWI 시계열
# ----------------------------------------------------------------
st.subheader("EWI 시계열 추이")

company_df = df[df["회사"] == company].copy()
company_df = company_df.sort_values(
    by="분기", key=lambda s: s.map(_quarter_sort_key)
)

point_colors = [
    "crimson" if v <= ewi_info["threshold"] else "royalblue"
    for v in company_df[EWI_COLUMN]
]

fig_ts = go.Figure()
fig_ts.add_trace(go.Scatter(
    x=company_df["분기"], y=company_df[EWI_COLUMN],
    mode="lines+markers", name="EWI",
    marker=dict(color=point_colors, size=9),
    line=dict(color="lightslategray"),
))
fig_ts.add_hline(
    y=ewi_info["threshold"], line_dash="dash", line_color="crimson",
    annotation_text="경고 임계값", annotation_position="bottom right",
)
fig_ts.update_layout(height=350, margin=dict(t=20, b=20), yaxis_title="EWI 지표값")
st.plotly_chart(fig_ts, use_container_width=True)

st.divider()

# ----------------------------------------------------------------
# 차트 2 : 핵심지표 분해 (위험기여 z-score)
# ----------------------------------------------------------------
st.subheader("핵심지표 분해 — 이번 분기 위험기여도")

diag_df = pd.DataFrame(ewi_info["diagnosis"])
bar_colors = ["crimson" if z > 0 else "seagreen" for z in diag_df["위험기여_z"]]

fig_bar = go.Figure(go.Bar(
    x=diag_df["위험기여_z"], y=diag_df["지표"], orientation="h",
    marker_color=bar_colors,
    text=diag_df["위험기여_z"].map(lambda v: f"{v:+.2f}"),
    textposition="outside",
))
fig_bar.update_layout(
    height=280, margin=dict(t=20, b=20),
    xaxis_title="위험기여 z-score ( + 위험 기여 / - 안전 기여 )",
)
st.plotly_chart(fig_bar, use_container_width=True)

with st.expander("지표 상세 수치 보기"):
    st.dataframe(diag_df, use_container_width=True, hide_index=True)

st.divider()

# ----------------------------------------------------------------
# AI 리포트 : 실시간 생성 + 세션 캐싱
# ----------------------------------------------------------------
st.subheader("🤖 AI 분석 리포트")

cache_key = (company, quarter)
cached_report = st.session_state.report_cache.get(cache_key)

col_a, col_b = st.columns([1, 4])
with col_a:
    generate_clicked = st.button("리포트 생성", type="primary")
with col_b:
    if cached_report:
        st.caption("✅ 이미 생성된 리포트가 있습니다 (세션 캐시). 다시 만들려면 버튼을 눌러주세요.")

if generate_clicked:
    with st.spinner("뉴스·법령 조사 및 리포트 생성 중... (수십 초 소요될 수 있습니다)"):
        result = graph.invoke({"company": company, "quarter": quarter})
        cached_report = result["report"]
        st.session_state.report_cache[cache_key] = cached_report

if cached_report:
    st.markdown(_clean_report_text(cached_report))
else:
    st.info("아직 생성된 리포트가 없습니다. 위 [리포트 생성] 버튼을 눌러주세요.")
    
    
##    
import streamlit as st

groq_key = st.secrets["GROQ_API_KEY"]
law_oc = st.secrets["LAW_API_OC"]