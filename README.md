# 카드사 EWI 분석 에이전트 (LangGraph + Groq + ChromaDB)

카드사 패널데이터의 리스크조기경보지표(EWI)를 판정하고, 관련 뉴스·법령/행정규칙을
자율 검색(Tool Calling)해 3단락 분석 리포트를 자동 생성하는 LangGraph 에이전트입니다.

## 구조

```
ewi_agent/
├── data/NEW_카드사_패널데이터_최종.xlsx   # 패널데이터 (엑셀)
├── config.py                          # 환경설정 로더 (.env 사용)
├── ewi.py                             # EWI 판정 로직 (평균 - 1.5*표준편차)
├── crawler_news.py                    # 한경/매경 뉴스 크롤러
├── law_api.py                         # 금융위 법령/행정규칙 API 클라이언트
├── build_vectorstore.py               # 뉴스/법령 → ChromaDB 임베딩 인덱스 구축
├── tools.py                           # LangChain Tool 정의 (search_news, search_law, get_panel_data)
├── agent_graph.py                     # LangGraph 그래프 정의 (핵심 로직)
├── main.py                            # CLI 실행 진입점
├── requirements.txt
└── .env.example
```

## 설치 (VSCode 터미널 기준)

```bash
cd ewi_agent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt

cp .env.example .env            # 이후 .env 파일 열어서 값 채우기
```

`.env`에 채워야 하는 값:
- `GROQ_API_KEY` : Groq API 키
- `LAW_API_OC` : law.go.kr Open API 인증키(이메일 아이디)

## 실행 순서

### 1단계 — 뉴스/법령 인덱스 구축 (최초 1회, 이후 주기적으로 재실행 권장)

```bash
python build_vectorstore.py
```

- 한국경제/매일경제 최근 30일 기사를 크롤링하고,
- 금융위원회 소관 법령/행정규칙을 조회한 뒤,
- `jhgan/ko-sroberta-multitask` 임베딩으로 청크 단위 벡터화해서
- `./chroma_db` 에 `news`, `law` 두 개 컬렉션으로 저장합니다.
- 재실행 시 기존 컬렉션을 삭제 후 재구축하므로 항상 "최근 1개월" 기준을 유지합니다.

### 2단계 — 리포트 생성

```bash
python main.py --company 국민카드 --quarter 2025/Q2
```

- `--quarter` 값은 엑셀의 `분기` 컬럼 형식(`YYYY/Qn`)과 정확히 일치해야 합니다.
- 내부적으로:
  1. 엑셀에서 해당 회사의 전체 분기 평균·표준편차를 계산해 임계값(`평균 - 1.5*표준편차`) 산출
  2. 현재 분기 지표값이 임계값 이하면 `경고`, 아니면 `비경고`로 판정
  3. LLM(Groq)이 `search_news`, `search_law` 도구를 필요하다고 판단할 때만 자율 호출
  4. 판정 결과에 따라 다른 프롬프트 템플릿으로 3단락 리포트 생성

## EWI 임계값 조정

`config.py`의 `EWI_STD_MULTIPLIER = 1.5` 값을 바꾸면 임계값 민감도를 조정할 수 있습니다.

## 참고 / 주의사항

- 네이버 뉴스 크롤링은 사이트 구조 변경에 취약합니다. 셀렉터가 안 맞으면
  `crawler_news.py`의 `_get_article_content` 셀렉터를 갱신하세요.
- Groq는 임베딩 API를 제공하지 않아 임베딩은 로컬 `sentence-transformers` 모델을
  사용합니다(최초 실행 시 모델 다운로드로 시간이 걸릴 수 있음).
- 여러 회사·분기를 일괄 처리하려면 `main.py`를 반복 호출하는 배치 스크립트를
  추가하면 됩니다 (예: 엑셀의 모든 `회사`×최신 `분기` 조합에 대해 for 루프).
