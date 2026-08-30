"""
공통 설정 로더.
Colab의 userdata 대신 .env 파일 + python-dotenv를 사용합니다.
"""
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LAW_API_OC = os.getenv("LAW_API_OC")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY가 설정되어 있지 않습니다. .env 파일을 만들고 값을 채워주세요. "
        "(.env.example 참고)"
    )
if not LAW_API_OC:
    raise ValueError(
        "LAW_API_OC가 설정되어 있지 않습니다. .env 파일을 만들고 값을 채워주세요. "
        "(.env.example 참고, law.go.kr 인증키)"
    )

# 경로 설정
PANEL_DATA_PATH = os.getenv("PANEL_DATA_PATH", "./data/NEW_카드사_패널데이터_최종.xlsx")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CRAWL_STATE_PATH = os.getenv("CRAWL_STATE_PATH", "./crawl_state.json")

# 모델 설정
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "jhgan/ko-sroberta-multitask")
GROQ_MODEL_NAME = os.getenv("GROQ_MODEL_NAME", "llama-3.3-70b-versatile")

# EWI 판정: 회사별 (평균 - EWI_STD_MULTIPLIER * 표준편차) 이하이면 '경고'
EWI_STD_MULTIPLIER = 1.5

# 뉴스 크롤링 설정
NEWS_OFFICES = ["한국경제", "매일경제"]
NEWS_LOOKBACK_DAYS = 30      # 최초 실행 시 수집할 기간(일) — 이후엔 증분 수집으로 전환
NEWS_MAX_PAGES = 100         # 언론사당 검색 결과 페이지 수 (페이지당 100건). 1개월치를 충분히 담기 위해 상향.

# 법령 API 설정
LAW_ORG_CODE = "1160100"  # 금융위원회


