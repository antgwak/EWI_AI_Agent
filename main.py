""""
카드사 EWI 리포트 생성 에이전트 실행 진입점.

사용 전:
    1. python build_vectorstore.py   # 뉴스/법령 인덱스 최초(또는 주기적) 구축
    2. python main.py --company 국민카드 --quarter 2025/Q2

주의: 엑셀 파일의 '분기' 값 형식(예: 2025/Q2)과 정확히 일치해야 합니다.
"""
import argparse

from agent_graph import build_graph


def main():
    parser = argparse.ArgumentParser(description="카드사 EWI 리포트 생성 에이전트")
    parser.add_argument("--company", required=True, help="예: 우리카드")
    parser.add_argument("--quarter", required=True, help="예: 2025/Q2")
    args = parser.parse_args()

    app = build_graph()
    result = app.invoke({"company": args.company, "quarter": args.quarter})

    ewi_info = result["ewi_info"]
    print("\n" + "=" * 70)
    print(f"[{args.company} {args.quarter}] EWI 판정: {ewi_info['level']}")
    print(
        f"(지표값 {ewi_info['current_value']} / 임계값 {ewi_info['threshold']} "
        f"= 평균 {ewi_info['mean']} - 1.5*표준편차 {ewi_info['std']})"
    )
    print("=" * 70)
    print(result["report"])
    print("=" * 70)


if __name__ == "__main__":
    main()

# 실행 예시:
#   python main.py --company 우리카드 --quarter 2025/Q3
