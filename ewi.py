"""
EWI(리스크조기경보지표) 판정 로직.
회사별 전체 분기의 (평균 - 1.5*표준편차)를 경고 임계값으로 사용합니다.
지표값이 임계값 이하면 '경고', 초과하면 '비경고'로 판정합니다.

추가로, EWI를 구성하는 4대 핵심지표를 "회사 자기 자신의 시계열" 기준으로
z-score 분해해서 이번 분기 어떤 지표가 판정을 주도했는지 계산합니다.
(LASSO/OLS 풀링회귀는 회사 규모에 따른 스케일 왜곡 위험이 있어 제거했고,
 대신 이 방식을 리포트 근거로 사용합니다 — 항상 같은 변수가 아니라
 회사·분기마다 다른 원인이 나옵니다.)

※ 표준편차 배수(1.5)는 config.py의 EWI_STD_MULTIPLIER에서 조정 가능합니다.
"""
import pandas as pd

from config import PANEL_DATA_PATH, EWI_STD_MULTIPLIER

EWI_COLUMN = "리스크조기경보지표"

# EWI 산출에 쓰인 4대 핵심지표와 위험 방향.
# (카드사_패널데이터_구축.py의 build_target()과 반드시 일치시킬 것)
CORE_INDICATORS = {
    "연체율_1개월이상": "up",       # 높을수록 위험
    "고정이하여신비율": "up",       # 높을수록 위험
    "대손충당금적립비율": "down",   # 낮을수록 위험
    "조정자기자본비율": "down",     # 낮을수록 위험
}


def load_panel_data() -> pd.DataFrame:
    df = pd.read_excel(PANEL_DATA_PATH)
    return df


def get_component_diagnosis(company_df: pd.DataFrame, row: pd.Series) -> list[dict]:
    """
    EWI를 구성하는 4개 핵심지표별로, 해당 회사 자체 히스토리 대비 z-score와
    전분기 대비 변화(QoQ)를 계산한다.

    z-score는 '위험 방향'으로 부호를 통일한다 (+ 값일수록 위험 기여).
    → 값이 큰 순으로 정렬해서 반환하므로, 상위 1~2개가 "이번 판정의 핵심 원인 후보".
    """
    company_df = company_df.sort_values("분기").reset_index(drop=True)
    idx = company_df.index[company_df["분기"] == row["분기"]].tolist()[0]
    prev_row = company_df.loc[idx - 1] if idx > 0 else None

    diagnosis = []
    for col, direction in CORE_INDICATORS.items():
        mean = company_df[col].mean()
        std = company_df[col].std()
        z = (row[col] - mean) / std if std else 0.0
        z_signed = z if direction == "up" else -z
        qoq = (row[col] - prev_row[col]) if prev_row is not None else None
        diagnosis.append({
            "지표": col,
            "값": round(float(row[col]), 3),
            "회사평균": round(float(mean), 3),
            "위험기여_z": round(float(z_signed), 2),
            "전분기대비_변화": round(float(qoq), 3) if qoq is not None else None,
        })

    diagnosis.sort(key=lambda d: d["위험기여_z"], reverse=True)
    return diagnosis


def get_supplementary_diagnosis(company_df: pd.DataFrame, row: pd.Series, supplementary_keys: list[str]) -> list[dict]:
    """
    EWI 4대 핵심지표(CORE_INDICATORS)가 아닌 보조 재무/거시 데이터
    (수익성·외형·리볼빙·대환대출·거시경제 등)에 대해서도 회사 자체 히스토리 대비
    편차(z-score)와 전분기 대비 변화(QoQ)를 계산한다.

    보고서 2단락 "핵심 원인 2가지" 표는 원래 EWI를 구성하는 4대 핵심지표(z-score)로
    만들었으나, 1단락에서 이미 그 지표들로 경고/비경고를 설명하므로 중복을 피하기 위해
    2단락 표는 이 보조 재무데이터 중 회사 자체 평균 대비 변동이 가장 큰 2개를 사용한다.
    (요청사항 반영: 2단락 표에는 z-score 4대 핵심지표를 다시 쓰지 않음)

    방향성(위험/안전)이 정해져 있지 않은 값들이므로 부호는 그대로 두고,
    |z-score|가 큰 순으로 정렬해 "이번 분기 가장 크게 움직인 재무데이터"를 골라낸다.
    """
    company_df = company_df.sort_values("분기").reset_index(drop=True)
    idx = company_df.index[company_df["분기"] == row["분기"]].tolist()[0]
    prev_row = company_df.loc[idx - 1] if idx > 0 else None

    diagnosis = []
    for col in supplementary_keys:
        if not pd.api.types.is_numeric_dtype(company_df[col]):
            continue  # 숫자가 아닌 컬럼(예: 문자열 코드)은 z-score 계산 대상에서 제외

        mean = company_df[col].mean()
        std = company_df[col].std()
        z = (row[col] - mean) / std if std else 0.0
        qoq = (row[col] - prev_row[col]) if prev_row is not None else None
        diagnosis.append({
            "지표": col,
            "값": round(float(row[col]), 3),
            "회사평균": round(float(mean), 3),
            "편차_z": round(float(z), 2),
            "전분기대비_변화": round(float(qoq), 3) if qoq is not None else None,
        })

    diagnosis.sort(key=lambda d: abs(d["편차_z"]), reverse=True)
    return diagnosis


def get_ewi_level(df: pd.DataFrame, company: str, quarter: str) -> dict:
    """
    특정 회사/분기의 EWI 판정 결과와 핵심지표 분해(diagnosis)를 반환.

    Returns:
        dict: company, quarter, current_value, mean, std, threshold, level,
              diagnosis(핵심 4개 지표를 위험기여도 순으로 정렬한 리스트),
              supplementary(그 외 참고용 원자료 — ROA/ROE/총자산/리볼빙 등)
    """
    company_df = df[df["회사"] == company]
    if company_df.empty:
        raise ValueError(f"'{company}' 데이터가 패널에 없습니다. 회사명을 확인하세요.")

    mean = company_df[EWI_COLUMN].mean()
    std = company_df[EWI_COLUMN].std()
    threshold = mean - EWI_STD_MULTIPLIER * std

    row_df = company_df[company_df["분기"] == quarter]
    if row_df.empty:
        available = ", ".join(sorted(company_df["분기"].unique()))
        raise ValueError(
            f"'{company}'의 '{quarter}' 데이터가 없습니다. 사용 가능한 분기: {available}"
        )
    row = row_df.iloc[0]

    current_value = float(row[EWI_COLUMN])
    level = "경고" if current_value <= threshold else "비경고"

    diagnosis = get_component_diagnosis(company_df, row)

    # 4대 핵심지표는 diagnosis에서 이미 다루므로, 나머지(수익성/외형/거시변수)만 참고용으로 분리
    exclude_cols = {"회사", "분기", EWI_COLUMN, *CORE_INDICATORS.keys()}
    supplementary = {k: v for k, v in row.to_dict().items() if k not in exclude_cols}

    # 2단락 "핵심 원인 2가지" 표용 — 보조 재무데이터 기준 변동 진단 (4대 z-score 지표와 분리)
    supplementary_diagnosis = get_supplementary_diagnosis(company_df, row, list(supplementary.keys()))

    return {
        "company": company,
        "quarter": quarter,
        "current_value": round(current_value, 4),
        "mean": round(float(mean), 4),
        "std": round(float(std), 4),
        "threshold": round(float(threshold), 4),
        "level": level,
        "diagnosis": diagnosis,
        "supplementary": supplementary,
        "supplementary_diagnosis": supplementary_diagnosis,
    }


if __name__ == "__main__":
    # 단독 실행 테스트: python ewi.py
    df = load_panel_data()
    company = df["회사"].iloc[0]
    quarter = df[df["회사"] == company]["분기"].iloc[-1]
    result = get_ewi_level(df, company, quarter)
    print(result)