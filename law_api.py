"""
국가법령정보센터 Open API를 통한 금융위원회 소관 법령/행정규칙 조회.
업로드하신 노트북(금융위원회_법령_행정규칙.ipynb) 로직을 재사용 가능한 함수로 정리했습니다.
"""
import xml.etree.ElementTree as ET
from typing import Optional

import requests

from config import LAW_API_OC

BASE_URL = "http://www.law.go.kr/DRF"


def _fetch_xml(url: str, params: dict) -> Optional[ET.Element]:
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return ET.fromstring(response.content)
    except Exception as e:
        print(f"[법령API 오류] {e}")
        return None


def fetch_laws(org_code: str, display: int = 200) -> list[dict]:
    """금융위원회(등) 소관 현행법령 목록 + 조문 텍스트 조회."""
    root = _fetch_xml(
        f"{BASE_URL}/lawSearch.do",
        {
            "OC": LAW_API_OC,
            "target": "eflaw",
            "type": "XML",
            "org": org_code,
            "nw": "2,3",
            "display": display,
            "page": 1,
        },
    )
    if root is None:
        return []

    results = []
    for law in root.findall("law"):
        law_id = law.findtext("법령ID")
        # ⚠️ 상세페이지(lsInfoP.do)의 lsiSeq 파라미터는 '법령ID'가 아니라
        #    '법령일련번호'(MST)를 요구한다. 이 둘을 혼동하면 완전히 다른(엉뚱한) 법령의
        #    상세페이지로 연결되는 버그가 발생한다 (예: 여신전문금융업법 → 석유 관련 법령).
        #    lawService.do 상세조회는 기존처럼 법령ID로도 조회 가능하므로 그대로 두고,
        #    URL 조립용 id만 법령일련번호로 분리해서 저장한다.
        law_mst = law.findtext("법령일련번호") or law_id
        law_name = law.findtext("법령명한글")
        prom_date = law.findtext("공포일자")
        enforce_date = law.findtext("시행일자")

        detail_root = _fetch_xml(
            f"{BASE_URL}/lawService.do",
            {"OC": LAW_API_OC, "target": "eflaw", "type": "XML", "ID": law_id},
        )

        rev_type = ""
        content = ""
        if detail_root is not None:
            info = detail_root.find("기본정보")
            if info is not None:
                rev_type = info.findtext("제개정구분명") or ""
            # 상세 XML 내 모든 텍스트 노드를 이어붙여 본문 텍스트 근사치로 사용
            texts = [el.text.strip() for el in detail_root.iter() if el.text and el.text.strip()]
            content = " ".join(texts)[:3000]

        results.append(
            {
                "type": "law",
                "name": law_name,
                "id": law_mst,  # 상세페이지 URL(lsiSeq) 조립용 — 법령일련번호
                "law_id": law_id,  # 참고용 원본 법령ID (필요시 다른 조회에 사용)
                "prom_date": prom_date,
                "enforce_date": enforce_date,
                "rev_type": rev_type,
                "content": content,
            }
        )

    print(f"법령 {len(results)}건 수집 완료")
    return results


def fetch_admin_rules(org_code: str, display: int = 200) -> list[dict]:
    """금융위원회(등) 소관 행정규칙 목록 + 본문 텍스트 조회."""
    root = _fetch_xml(
        f"{BASE_URL}/lawSearch.do",
        {
            "OC": LAW_API_OC,
            "target": "admrul",
            "type": "XML",
            "org": org_code,
            "nw": 1,
            "display": display,
            "page": 1,
        },
    )
    if root is None:
        return []

    results = []
    for admrul in root.findall("admrul"):
        seq = admrul.findtext("행정규칙일련번호")
        name = admrul.findtext("행정규칙명")
        prml_date = admrul.findtext("발령일자")
        prml_no = admrul.findtext("발령번호")

        detail_root = _fetch_xml(
            f"{BASE_URL}/lawService.do",
            {"OC": LAW_API_OC, "target": "admrul", "type": "XML", "ID": seq},
        )

        kind = ""
        content = ""
        if detail_root is not None:
            info = detail_root.find("기본정보")
            if info is not None:
                kind = info.findtext("행정규칙종류") or ""
            texts = [el.text.strip() for el in detail_root.iter() if el.text and el.text.strip()]
            content = " ".join(texts)[:3000]

        results.append(
            {
                "type": "admrul",
                "name": name,
                "id": seq,
                "prom_date": prml_date,
                "prom_no": prml_no,
                "kind": kind,
                "content": content,
            }
        )

    print(f"행정규칙 {len(results)}건 수집 완료")
    return results


if __name__ == "__main__":
    # 단독 실행 테스트: python law_api.py
    laws = fetch_laws("1160100", display=10)
    for l in laws:
        print(l["name"], "-", l["enforce_date"])     
