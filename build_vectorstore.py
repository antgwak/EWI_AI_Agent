"""
뉴스 기사 & 금융위 법령/행정규칙을 크롤링해 ChromaDB에 임베딩 저장.

[뉴스 처리 방식 - 증분(incremental) 누적 저장]
- 최초 실행: 오늘 기준 최근 NEWS_LOOKBACK_DAYS(기본 30일)치를 전부 수집.
- 이후 실행: crawl_state.json에 저장된 (마지막 수집일 + 1일)부터 오늘까지만 새로 수집하여
    기존 인덱스에 지속 축적 저장 (같은 기사 URL이면 덮어써서 중복 방지).
- 기존 데이터를 삭제하지 않고 계속 누적합니다.

[법령/행정규칙 처리 방식 - 증분(incremental) upsert]
- 법령은 자주 바뀌지 않으므로 더 이상 매번 전체를 재구축하지 않습니다.
- (타입, id, 청크인덱스)로 고유 ID를 만들어 add_documents(upsert)하면,
  이미 존재하는 조문은 덮어쓰고 새로 생긴 조문만 추가됩니다.
- 메타데이터에 id(법령ID/행정규칙일련번호)를 저장해, tools.py에서
  국가법령정보센터 상세페이지 링크(lsInfoP.do / admRulInfoP.do)를 조립할 수 있게 합니다.
  ※ 다만 폐지되어 목록에서 사라진 법령/행정규칙은 자동 삭제되지 않고 인덱스에 남습니다.
    필요하면 _reset_collection("law")으로 수동 재구축하세요.

실행:
    python build_vectorstore.py
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import chromadb
from config import (
    CHROMA_PERSIST_DIR,
    CRAWL_STATE_PATH,
    EMBEDDING_MODEL_NAME,
    LAW_ORG_CODE,
    NEWS_LOOKBACK_DAYS,
    NEWS_MAX_PAGES,
    NEWS_OFFICES,
)
from crawler_news import crawl_news
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from law_api import fetch_admin_rules, fetch_laws

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)


def _load_state() -> dict:
    path = Path(CRAWL_STATE_PATH)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _save_state(state: dict):
    Path(CRAWL_STATE_PATH).write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _reset_collection(name: str):
    """법령/행정규칙을 통째로 다시 구축하고 싶을 때 수동으로 호출하는 용도로 남겨둠."""
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        client.delete_collection(name)
        print(f"기존 '{name}' 컬렉션 삭제")
    except Exception:
        pass  # 컬렉션이 없으면 그냥 진행


def _make_news_id(article: dict, chunk_idx: int) -> str:
    """URL + 청크 인덱스로 고유 ID 생성 → 같은 기사를 다시 수집해도 덮어쓰기(upsert) 되어 중복 방지."""
    return f"{article['url']}::{chunk_idx}"


def build_news_index(embeddings):
    print("\n=== 뉴스 인덱스 업데이트 ===")
    state = _load_state()
    today = date.today()

    last_date_str = state.get("news_last_date")
    if last_date_str:
        # 이미 마지막 수집일까지 저장되어 있으므로 +1일부터 수집 진행
        last_date = datetime.strptime(last_date_str, "%Y-%m-%d").date()
        start_date = last_date + timedelta(days=1)

        if start_date > today:
            print(
                f"최신 데이터가 이미 등록되어 있습니다. (마지막 수집일: {last_date_str})"
            )
            return

        print(
            f"기존 인덱스 축적 유지 → {start_date} ~ {today} 구간 신규 수집(증분)"
        )
    else:
        # 최초 실행 시 지정된 Lookback 기간만큼 전체 수집
        start_date = today - timedelta(days=NEWS_LOOKBACK_DAYS)
        print(
            f"최초 실행 → {start_date} ~ {today} ({NEWS_LOOKBACK_DAYS}일치) 전체 수집 및 인덱싱"
        )

    articles = crawl_news(
        NEWS_OFFICES,
        start_date=start_date,
        end_date=today,
        max_pages=NEWS_MAX_PAGES,
    )

    news_store = Chroma(
        collection_name="news",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    if articles:
        docs, ids = [], []
        total_articles = len(articles)

        print("\n=== 수집된 기사 텍스트 분할(Chunking) 진행 중 ===")
        for idx, a in enumerate(articles, 1):
            if idx % 50 == 0 or idx == total_articles:
                print(
                    f"\r청크 분할 진행률: {idx}/{total_articles}건 ({idx/total_articles*100:.1f}%)",
                    end="",
                    flush=True,
                )

            chunks = splitter.split_text(a["content"]) or [a["content"]]
            for i, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=f"[{a['title']}] {chunk}",
                        metadata={
                            "title": a["title"],
                            "office": a["office"],
                            "date": a["date"],
                            "date_num": int(a["date"].replace("-", "")),
                            "url": a["url"],
                            "chunk": i,
                        },
                    )
                )
                ids.append(_make_news_id(a, i))
        print()

        print(
            f"\n총 {len(docs)}개 뉴스 Chunk를 Vectorstore(ChromaDB)에 임베딩 저장 중..."
        )
        news_store.add_documents(docs, ids=ids)
        print(f"신규 뉴스 chunk {len(docs)}개 저장 완료!")
    else:
        print("신규 뉴스 없음")

    # 수집이 성공적으로 완료되었을 때만 상태 업데이트
    state["news_last_date"] = today.strftime("%Y-%m-%d")
    _save_state(state)


def _make_law_id(item: dict, chunk_idx: int) -> str:
    """[타입]_[ID]_[청크인덱스] 형태의 고유 ID 생성 (예: law_123456_0)
    이 ID를 지정하면 이미 존재하는 조문은 덮어쓰고, 새로 생긴 조문은 추가됩니다.
    """
    return f"{item['type']}_{item['id']}_{chunk_idx}"


def _dedupe_law_items(items: list[dict]) -> list[dict]:
    """law_api.py가 개정 이력 등으로 동일 (type, id)를 중복 반환하는 경우가 있어,
    한 번의 upsert 호출 안에서 동일 ID가 겹치지 않도록 (type, id) 기준으로 중복 제거한다.
    나중에 나온 항목(더 최신 조회 결과)을 우선한다."""
    deduped: dict[tuple, dict] = {}
    for item in items:
        deduped[(item["type"], item["id"])] = item
    return list(deduped.values())


def build_law_index(embeddings):
    print("\n=== 법령/행정규칙 축적 및 업데이트 시작 ===")

    # display=200 으로 금융위 전체 법령/행정규칙 목록 가져오기
    laws = fetch_laws(LAW_ORG_CODE, display=200)
    admrules = fetch_admin_rules(LAW_ORG_CODE, display=200)
    items = _dedupe_law_items(laws + admrules)

    docs, ids = [], []
    for item in items:
        text = item.get("content") or item["name"]
        chunks = splitter.split_text(text) or [text]
        for i, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=f"[{item['name']}] {chunk}",
                    metadata={
                        "name": item["name"],
                        "type": item["type"],
                        "id": item["id"],  # 상세페이지 링크 조립용 (lsiSeq / admRulSeq)
                        "prom_date": item.get("prom_date", ""),
                        "chunk": i,
                    },
                )
            )
            # 고유 ID 생성
            ids.append(_make_law_id(item, i))

    if not docs:
        print("수집된 법령/행정규칙이 없어 인덱스 구축을 건너뜁니다.")
        return

    # 기존 law 컬렉션을 그대로 로드하여 add_documents(upsert) 수행
    law_store = Chroma(
        collection_name="law",
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR,
    )

    law_store.add_documents(docs, ids=ids)
    print(f"법령/행정규칙 인덱스 동기화 완료 (총 {len(docs)}개 chunk 적용/업데이트)")


if __name__ == "__main__":
    print(f"임베딩 모델 로딩 중: {EMBEDDING_MODEL_NAME}")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    build_news_index(embeddings)
    build_law_index(embeddings)

    print("\n인덱스 구축 완료. 이제 main.py를 실행할 수 있습니다.")