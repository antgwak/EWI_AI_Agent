"""
LangGraph 기반 EWI 분석 에이전트 (단순화 버전).

흐름:
1. determine_ewi   : 엑셀 패널데이터로 EWI 판정(경고/비경고) + 핵심지표 분해(diagnosis)를 결정
                      (결정론적 계산, LLM 미사용)
2. search_context  : search_news / search_law를 결정론적으로 직접 호출해 뉴스·법령 컨텍스트 확보
                      (LLM 미사용 — 매 회사/분기마다 항상 조사하는 것이 유용하므로
                       "조사할지 말지"를 LLM이 판단하는 단계 자체를 없앴다.
                       bind_tools로 도구 스키마를 매번 프롬프트에 실어 보내면
                       Groq 무료 티어의 TPM(분당 토큰) 한도를 쉽게 넘기기 때문)
3. report          : EWI 판정 결과에 따라 프롬프트를 분기해 최종 3단락 보고서 생성 (LLM 호출 1회뿐)
"""
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END

from config import GROQ_API_KEY, GROQ_MODEL_NAME
from tools import search_news, search_law
from ewi import load_panel_data, get_ewi_level

llm = ChatGroq(model=GROQ_MODEL_NAME, api_key=GROQ_API_KEY, temperature=0)


class AgentState(TypedDict):
    company: str
    quarter: str
    ewi_info: dict
    tool_context: str
    report: str


WARN_PROMPT = """당신은 카드업권 리스크 분석가입니다.
다음은 {company}의 {quarter} 리스크조기경보(EWI) 판정 결과입니다.

[EWI 판정]
- 지표값: {value} (경고 임계값: {threshold} = 평균 {mean} - 1.5*표준편차 {std})
- 판정: 경고 수준

[EWI를 구성한 핵심지표 전체 분해 - 위험기여도(z-score) 높은 순, 1단락 서술 전용 근거]
(회사 자기 자신의 과거 분기들과 비교한 값입니다. 위험기여_z가 클수록
 이번 EWI 판정에 크게 기여했다는 뜻입니다. 이 지표들은 1단락에서만 사용하고,
 2단락 표에는 절대 다시 쓰지 마세요.)
{diagnosis}

[2단락에 그대로 삽입할 표 - EWI 4대 핵심지표가 아닌, 회사 자체 평균 대비 이번 분기
 변동폭(|z-score|)이 가장 큰 보조 재무데이터 2개를 미리 정리함]
{core_table}

[참고: 수익성/외형/리볼빙·대환대출/거시경제 원자료 전체]
{supplementary}

[조사된 뉴스/법령 정보]
{context}

[작성 지침]
- 마크다운 형식으로 작성하세요. 줄바꿈이 필요하면 실제 줄바꿈 문자를 사용하고,
  <br> 등 HTML 태그는 절대 사용하지 마세요.
- 세 단락의 소제목은 모두 같은 마크다운 헤딩 레벨(###)을 사용해 글자 크기를 통일하세요.
- 각 단락 소제목 맨 앞에 순서대로 1️⃣, 2️⃣, 3️⃣ 이모지를 붙이세요.
  (예: "### 1️⃣ 경고 판단 근거", "### 2️⃣ 핵심 원인 2가지", "### 3️⃣ 구체적인 대응 권고사항")
- [조사된 뉴스/법령 정보]에 있는 항목을 근거로 쓸 때는 최대한 빠짐없이 인용하고,
  그 항목에 "(출처: URL)"이 있으면 반드시 마크다운 링크 형식
  [기사 제목 또는 법령명](URL)으로 표기해 출처를 명시하세요. 링크는 생략하지 말고
  본문에서 언급하는 모든 뉴스·법령 근거에 대해 가능한 한 출처 URL을 남기세요.
  출처 URL이 없는 항목만 링크 없이 서술하세요.

위 정보를 종합해 총 3개 단락으로 보고서를 작성하세요. 출력 시 문체는 일관되게 존댓말(격식체)로 작성합니다.

1단락 (제목: "1️⃣ 경고 판단 근거"): [EWI를 구성한 핵심지표 전체 분해]의 위험기여_z를 근거로,
       이 회사가 왜 EWI 경고 수준인지 설명하는 서술형 문단.

2단락 (제목: "2️⃣ 핵심 원인 2가지"): 소제목 다음에 [2단락에 그대로 삽입할 표]를 행/열 변경
       없이 그대로 넣으세요. 이 표는 1단락의 z-score 4대 핵심지표가 아니라 보조 재무데이터이므로
       그 지표를 다시 나열하지 말고, 표에 없는 3번째·4번째 지표도 추가하지 마세요.
       표 아래에는 그 2개 지표와 관련된 뉴스·법령 근거(출처 URL 포함)나 추가 해설을
       불릿을 사용하지 말고 자유로운 문단으로 덧붙이세요.

3단락 (제목: "3️⃣ 구체적인 대응 권고사항"): 표를 쓰지 말고 불릿포인트 리스트로, 최대 3개까지만
       작성하세요 (3개를 넘기지 마세요). 각 불릿은 "**리스크/권고 이름**: 설명 → 대응 방안"
       형식의 한 문단으로 작성하고, 불릿 안에 하위 불릿이나 표를 추가로 만들지 마세요.
       마지막에는 "핵심 메시지" 같은 볼드체 소제목을 달지 말고, 자연스러운 요약 문단 하나로
       전체 내용을 마무리하세요 (예: "이와 같이 현재는 ...로 시작하는 문단").
"""

SAFE_PROMPT = """당신은 카드업권 리스크 분석가입니다.
다음은 {company}의 {quarter} 리스크조기경보(EWI) 판정 결과입니다.

[EWI 판정]
- 지표값: {value} (경고 임계값: {threshold} = 평균 {mean} - 1.5*표준편차 {std})
- 판정: 비경고 수준

[EWI를 구성한 핵심지표 전체 분해 - 위험기여도(z-score) 높은 순, 1단락 서술 전용 근거]
(회사 자기 자신의 과거 분기들과 비교한 값입니다. 값이 음수에 가까울수록
 그 지표가 안전한 방향으로 기여했다는 뜻입니다. 이 지표들은 1단락에서만 사용하고,
 2단락 표에는 절대 다시 쓰지 마세요.)
{diagnosis}

[2단락에 그대로 삽입할 표 - EWI 4대 핵심지표가 아닌, 회사 자체 평균 대비 이번 분기
 변동폭(|z-score|)이 가장 큰 보조 재무데이터 2개를 미리 정리함]
{core_table}

[참고: 수익성/외형/리볼빙·대환대출/거시경제 원자료 전체]
{supplementary}

[조사된 뉴스/법령 정보]
{context}

[작성 지침]
- 마크다운 형식으로 작성하세요. 줄바꿈이 필요하면 실제 줄바꿈 문자를 사용하고,
  <br> 등 HTML 태그는 절대 사용하지 마세요.
- 세 단락의 소제목은 모두 같은 마크다운 헤딩 레벨(###)을 사용해 글자 크기를 통일하세요.
- 각 단락 소제목 맨 앞에 순서대로 1️⃣, 2️⃣, 3️⃣ 이모지를 붙이세요.
  (예: "### 1️⃣ 비경고 판단 근거", "### 2️⃣ 핵심 원인 2가지",
   "### 3️⃣ 향후 예상되는 리스크 요인과 이에 대한 준비 방안")
- [조사된 뉴스/법령 정보]에 있는 항목을 근거로 쓸 때는 최대한 빠짐없이 인용하고,
  그 항목에 "(출처: URL)"이 있으면 반드시 마크다운 링크 형식
  [기사 제목 또는 법령명](URL)으로 표기해 출처를 명시하세요. 링크는 생략하지 말고
  본문에서 언급하는 모든 뉴스·법령 근거에 대해 가능한 한 출처 URL을 남기세요.
  출처 URL이 없는 항목만 링크 없이 서술하세요.

위 정보를 종합해 총 3개 단락으로 보고서를 작성하세요. 출력 시 문체는 일관되게 존댓말(격식체)로 작성합니다.

1단락 (제목: "1️⃣ 비경고 판단 근거"): [EWI를 구성한 핵심지표 전체 분해]의 위험기여_z를 근거로,
       이 회사가 왜 EWI 경고 수준이 아닌지 설명하는 서술형 문단.

2단락 (제목: "2️⃣ 핵심 원인 2가지"): 소제목 다음에 [2단락에 그대로 삽입할 표]를 행/열 변경
       없이 그대로 넣으세요. 이 표는 1단락의 z-score 4대 핵심지표가 아니라 보조 재무데이터이므로
       그 지표를 다시 나열하지 말고, 표에 없는 3번째·4번째 지표도 추가하지 마세요.
       표 아래에는 그 2개 지표와 관련된 뉴스·법령 근거(출처 URL 포함)나 추가 해설을
       불릿을 사용하지 말고 자유로운 문단으로 덧붙이세요.

3단락 (제목: "3️⃣ 향후 예상되는 리스크 요인과 이에 대한 준비 방안"): 표를 쓰지 말고
       불릿포인트 리스트로, 최대 3개까지만 작성하세요 (3개를 넘기지 마세요).
       각 불릿은 "**리스크 이름**: 근거 → 준비·완화 방안" 형식의 한 문단으로 작성하고,
       불릿 안에 하위 불릿이나 표를 추가로 만들지 마세요.
       마지막에는 "핵심 메시지" 같은 볼드체 소제목을 달지 말고, 자연스러운 요약 문단 하나로
       전체 내용을 마무리하세요 (예: "이와 같이 현재는 ...로 시작하는 문단").
"""


def _format_diagnosis(diagnosis: list[dict]) -> str:
    lines = []
    for d in diagnosis:
        qoq = f"{d['전분기대비_변화']:+.3f}" if d["전분기대비_변화"] is not None else "N/A(직전분기 없음)"
        lines.append(
            f"- {d['지표']}: 값 {d['값']} (회사 자체 평균 {d['회사평균']}) "
            f"| 위험기여 z-score {d['위험기여_z']:+.2f} | 전분기 대비 변화 {qoq}"
        )
    return "\n".join(lines)


def _format_supplementary(supplementary: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in supplementary.items())


def _select_top2_for_table(supplementary_diagnosis: list[dict]) -> list[dict]:
    """2단락 표에 넣을 2개 지표를 고른다.
    ⚠️ EWI를 구성하는 4대 핵심지표(z-score, 1단락 전용)는 여기서 다시 쓰지 않는다.
    supplementary_diagnosis는 보조 재무데이터를 |z-score| 내림차순으로 정렬한 리스트이므로,
    회사 자체 평균 대비 이번 분기 변동폭이 가장 큰 상위 2개를 그대로 사용한다.
    """
    return supplementary_diagnosis[:2]


def _build_core_table(top2: list[dict]) -> str:
    """2단락에 그대로 삽입할, 정확히 2개의 데이터 행을 가진 마크다운 표를 만든다.
    (표 형식/행 개수를 LLM에 맡기면 들쭉날쭉해지는 문제가 있어 파이썬에서 결정론적으로 생성)
    ※ 이 표는 EWI 4대 핵심지표(z-score)가 아니라 보조 재무데이터 기준이므로
      "위험기여" 대신 "회사평균 대비 편차(z)"로 라벨링한다.
    """
    header = "| 핵심지표(재무) | 값 | 회사평균 대비 편차(z) | 전분기 대비 변화 |\n|---|---|---|---|"
    rows = []
    for d in top2:
        qoq = f"{d['전분기대비_변화']:+.3f}" if d["전분기대비_변화"] is not None else "N/A"
        rows.append(f"| {d['지표']} | {d['값']} | {d['편차_z']:+.2f} | {qoq} |")
    return "\n".join([header, *rows])


def determine_ewi(state: AgentState) -> AgentState:
    df = load_panel_data()
    ewi_info = get_ewi_level(df, state["company"], state["quarter"])
    return {**state, "ewi_info": ewi_info}


def search_context(state: AgentState) -> AgentState:
    """LLM 판단 없이 결정론적으로 뉴스/법령을 조회한다.
    쿼리는 회사명 + 위험기여도가 가장 큰 핵심지표로 구성해, 이번 판정과
    관련성이 높은 결과가 우선 검색되도록 한다.
    """
    ewi_info = state["ewi_info"]
    company = state["company"]
    top_indicator = ewi_info["diagnosis"][0]["지표"]

    # 출처 URL을 최대한 많이 확보하기 위해 k를 3 -> 5로 상향
    news_result = search_news.invoke({"query": f"{company} {top_indicator}", "k": 5})
    law_result = search_law.invoke({"query": f"카드사 {top_indicator} 감독규정", "k": 5})

    tool_context = f"[뉴스]\n{news_result}\n\n[법령/행정규칙]\n{law_result}"
    return {**state, "tool_context": tool_context}


def report_node(state: AgentState) -> AgentState:
    ewi_info = state["ewi_info"]
    level = ewi_info["level"]
    template = WARN_PROMPT if level == "경고" else SAFE_PROMPT

    top2 = _select_top2_for_table(ewi_info["supplementary_diagnosis"])
    core_table = _build_core_table(top2)

    prompt = template.format(
        company=ewi_info["company"],
        quarter=ewi_info["quarter"],
        value=ewi_info["current_value"],
        threshold=ewi_info["threshold"],
        mean=ewi_info["mean"],
        std=ewi_info["std"],
        diagnosis=_format_diagnosis(ewi_info["diagnosis"]),
        core_table=core_table,
        supplementary=_format_supplementary(ewi_info["supplementary"]),
        context=state["tool_context"],
    )

    final = llm.invoke(prompt)
    return {**state, "report": final.content}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("determine_ewi", determine_ewi)
    graph.add_node("search_context", search_context)
    graph.add_node("report", report_node)

    graph.set_entry_point("determine_ewi")
    graph.add_edge("determine_ewi", "search_context")
    graph.add_edge("search_context", "report")
    graph.add_edge("report", END)

    return graph.compile()