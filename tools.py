"""
에이전트가 자율 호출(Tool Calling)할 LangChain Tool 정의.
search_news / search_law 는 build_vectorstore.py로 미리 구축한 ChromaDB를 검색만 합니다.
결과에는 신뢰도를 위해 출처 링크(뉴스 기사 URL / 법령정보센터 상세페이지 URL)를 함께 반환합니다.
"""
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL_NAME
from ewi import load_panel_data

_embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

_news_store = Chroma(
    collection_name="news",
    embedding_function=_embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
)
_law_store = Chroma(
    collection_name="law",
    embedding_function=_embeddings,
    persist_directory=CHROMA_PERSIST_DIR,
)

_panel_df = load_panel_data()

# 국가법령정보센터 상세페이지 URL 패턴 (law_api.py가 저장한 id로 조립)
_LAW_DETAIL_URL = {
    "law": "https://www.law.go.kr/lsInfoP.do?lsiSeq={id}",
    "admrul": "https://www.law.go.kr/admRulInfoP.do?admRulSeq={id}",
}


@tool
def search_news(query: str, k: int = 5) -> str:
    """최근 1개월 한국경제/매일경제 뉴스 중 query와 관련된 기사를 검색한다.
    예: '국민카드 연체율', '카드사 대손충당금', '카드론 리스크'
    """
    docs = _news_store.similarity_search(query, k=k)
    if not docs:
        return "관련 뉴스를 찾지 못했습니다."
    lines = []
    for d in docs:
        office = d.metadata.get("office", "")
        published = d.metadata.get("date", "")
        url = d.metadata.get("url", "")
        source = f" (출처: {url})" if url else ""
        lines.append(f"- ({office}, {published}) {d.page_content}{source}")
    return "\n\n".join(lines)


@tool
def search_law(query: str, k: int = 5) -> str:
    """금융위원회 소관 법령/행정규칙 중 query와 관련된 조항을 검색한다.
    예: '카드사 건전성 감독규정', '대손충당금 적립기준', '리볼빙 규제'
    """
    docs = _law_store.similarity_search(query, k=k)
    if not docs:
        return "관련 법령/행정규칙을 찾지 못했습니다."
    lines = []
    for d in docs:
        name = d.metadata.get("name", "")
        doc_type = d.metadata.get("type", "")
        doc_id = d.metadata.get("id", "")
        url_template = _LAW_DETAIL_URL.get(doc_type)
        url = url_template.format(id=doc_id) if (url_template and doc_id) else ""
        source = f" (출처: {url})" if url else ""
        lines.append(f"- [{name}] {d.page_content}{source}")
    return "\n\n".join(lines)


@tool
def get_panel_data(company: str, quarter: str) -> dict:
    """특정 카드사의 특정 분기 재무/거시 패널데이터를 조회한다."""
    row = _panel_df[(_panel_df["회사"] == company) & (_panel_df["분기"] == quarter)]
    return row.iloc[0].to_dict() if len(row) else {}
