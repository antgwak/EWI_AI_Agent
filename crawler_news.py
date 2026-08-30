"""
네이버 뉴스 검색을 통한 한국경제/매일경제 기사 크롤링.
증분(incremental) 업데이트를 위해 start_date~end_date 범위를 직접 지정할 수 있도록
lookback_days 방식에서 날짜 범위 방식으로 변경했습니다.
"""
import re
import time
import urllib.parse
from datetime import date, datetime, timedelta
from typing import Optional

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    )
}

def _make_search_urls(office_name: str, start_date: date, end_date: date, start_pg: int, end_pg: int) -> list[str]:
    encoded_query = urllib.parse.quote(office_name)
    ds_date = start_date.strftime("%Y.%m.%d")
    de_date = end_date.strftime("%Y.%m.%d")

    urls = []
    for page in range(start_pg, end_pg + 1):
        page_num = 1 + 10 * (page - 1)
        url = (
            "https://search.naver.com/search.naver?ssc=tab.news.all"
            f"&query={encoded_query}"
            "&sm=tab_opt&sort=0&photo=3&field=0&pd=2"
            f"&ds={ds_date}&de={de_date}"
            "&docid=&qdt=0&related=0&mynews=0&office_type=0&office_section_code=0"
            "&news_office_checked=&nso=&is_sug_officeid=1&office_category=0&service_area=0"
            f"&start={page_num}"
        )
        urls.append(url)
    return urls


def _get_article_urls(search_page_url: str) -> list[str]:
    try:
        response = requests.get(search_page_url, headers=HEADERS, timeout=10)
        soup = BeautifulSoup(response.text, "html.parser")
        article_urls = []
        for link in soup.find_all("a"):
            href = link.attrs.get("href", "")
            if "news.naver.com/mnews/article/" in href:
                clean_url = href.split("?")[0]
                if clean_url not in article_urls:
                    article_urls.append(clean_url)
        return article_urls
    except Exception as e:
        print(f"[URL 수집 오류] {e}")
        return []


def _parse_article_date(raw_date: str, crawl_time: datetime) -> str:
    """
    네이버 뉴스 상세페이지의 날짜 문자열을 'YYYY-MM-DD' 형식으로 정규화.
    '3시간 전', '2025.08.29. 14:23' 등 다양한 형식을 처리.
    파싱에 실패하면 크롤링 당일 날짜로 대체(수집 자체는 누락시키지 않기 위함).
    """
    raw = raw_date.strip()
    try:
        if "분 전" in raw:
            minutes = int(re.sub(r"\D", "", raw) or 0)
            return (crawl_time - timedelta(minutes=minutes)).strftime("%Y-%m-%d")
        if "시간 전" in raw:
            hours = int(re.sub(r"\D", "", raw) or 0)
            return (crawl_time - timedelta(hours=hours)).strftime("%Y-%m-%d")
        if "일 전" in raw:
            days = int(re.sub(r"\D", "", raw) or 0)
            return (crawl_time - timedelta(days=days)).strftime("%Y-%m-%d")

        match = re.search(r"(\d{4})\D+(\d{1,2})\D+(\d{1,2})", raw)
        if match:
            y, m, d = match.groups()
            return f"{int(y):04d}-{int(m):02d}-{int(d):02d}"
    except Exception:
        pass
    return crawl_time.strftime("%Y-%m-%d")

def _get_article_content(article_url: str, crawl_time: datetime) -> Optional[dict]:
    # --- [수정] 재시도 로직(3회) 추가 ---
    max_retries = 3
    response = None

    for attempt in range(max_retries):
        try:
            # timeout = (연결 대기 5초, 데이터 응답 대기 15초)
            response = requests.get(article_url, headers=HEADERS, timeout=(5, 15))
            if response.status_code == 200:
                break
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\n[본문 수집 최종 실패 - 스킵] {article_url} ({e})")
                return None
            time.sleep(1)  # 재시도 전 1초 대기

    if not response or response.status_code != 200:
        return None

    try:
        soup = BeautifulSoup(response.text, "html.parser")

        title_el = (
            soup.select_one(
                "#ct > div.media_end_head.go_trans > div.media_end_head_title > h2"
            )
            or soup.select_one("h2#title_area")
            or soup.select_one("#content > div.end_ct > div > h2")
        )
        title = title_el.get_text(strip=True) if title_el else "제목 없음"

        content_el = (
            soup.select_one("article#dic_area")
            or soup.select_one("#articeBody")
            or soup.select_one("#newsct_article")
        )
        content = content_el.get_text(strip=True) if content_el else ""
        content = re.sub(r"\s+", " ", content).strip()

        date_el = soup.select_one("span.media_end_head_info_datestamp_time")
        raw_date = date_el.get_text(strip=True) if date_el else ""
        parsed_date = _parse_article_date(raw_date, crawl_time)

        if not content:
            return None

        return {
            "title": title,
            "content": content,
            "date": parsed_date,
            "raw_date": raw_date,
            "url": article_url,
        }
    except Exception as e:
        print(f"\n[파싱 오류 - 스킵] {article_url} ({e})")
        return None

def crawl_news(
    offices: list[str],
    start_date: date,
    end_date: date,
    max_pages: int = 10,
    delay: float = 0.3,
) -> list[dict]:
    """
    여러 언론사의 start_date~end_date 범위 기사를 크롤링해 리스트[dict]로 반환.
    각 dict: {title, content, date(YYYY-MM-DD), raw_date, url, office}
    """
    crawl_time = datetime.now()
    all_results = []

    for office in offices:
        search_urls = _make_search_urls(office, start_date, end_date, 1, max_pages)

        seen_urls = []
        for page_url in search_urls:
            for u in _get_article_urls(page_url):
                if u not in seen_urls:
                    seen_urls.append(u)
            time.sleep(delay)

        print(f"[{office}] 기사 URL {len(seen_urls)}건 발견 ({start_date} ~ {end_date})")
# --- [수정] 진행상황 실시간 출력 추가 ---
        total_urls = len(seen_urls)
        for idx, url in enumerate(seen_urls, 1):
            percent = (idx / total_urls) * 100
            print(
                f"\r[{office}] 본문 수집 중: {idx}/{total_urls}건 ({percent:.1f}%)",
                end="",
                flush=True,
            )

            data = _get_article_content(url, crawl_time)
            if data:
                data["office"] = office
                all_results.append(data)
            time.sleep(delay)
        print()  # 줄바꿈

    print(f"총 {len(all_results)}건 기사 수집 완료")
    return all_results


if __name__ == "__main__":
    # 단독 실행 테스트: python crawler_news.py
    today = date.today()
    results = crawl_news(["한국경제", "매일경제"], start_date=today - timedelta(days=3), end_date=today, max_pages=1)
    for r in results[:3]:
        print(r["office"], r["date"], "-", r["title"])
