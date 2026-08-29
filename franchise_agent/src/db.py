from functools import lru_cache

import pandas as pd
from sqlalchemy import create_engine, text

from .config import DATABASE_URL

# DB 금액 컬럼은 정보공개서 원문에서 파싱된 단위(THOUSAND_KRW/KRW/MAN_KRW/MILLION_KRW/
# HUNDRED_MILLION_KRW)가 브랜드마다 제각각이고, 설문/프로파일은 전부 '만원' 단위다(질문파일.pdf).
# 단위를 확정할 수 없는 값(NULL, SOURCE_UNIT 등)은 스케일을 알 수 없으므로 변환하지 않고 NaN 처리한다.
MONEY_UNIT_TO_MANWON = {
    "THOUSAND_KRW": 0.1,
    "MAN_KRW": 1.0,
    "KRW": 0.0001,
    "MILLION_KRW": 100.0,
    "HUNDRED_MILLION_KRW": 10000.0,
}


@lru_cache(maxsize=1)
def get_engine():
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def _convert_to_manwon(
    df: pd.DataFrame, amount_cols: list[str], unit_col: str, default_unit: str | None = None
) -> pd.DataFrame:
    """default_unit: 단위가 NULL일 때 쓸 기본 가정. 근처 텍스트에 '천원'이라는 말이 없어서
    뷰가 단위를 못 정한 것뿐, 대부분 실제로는 THOUSAND_KRW인 경우가 많다(전체 DB 조사 결과 약 82%).
    NULL을 전부 버리는 것보다 다수결 단위로 가정하는 편이 실제 값에 더 가깝다."""
    if df.empty or unit_col not in df.columns:
        return df
    df = df.copy()
    unit = df[unit_col].fillna(default_unit) if default_unit else df[unit_col]
    factor = unit.map(MONEY_UNIT_TO_MANWON)
    for col in amount_cols:
        if col in df.columns:
            df[col] = df[col] * factor
    return df


# 원본 정보공개서 표 파싱 과정에서 인접 셀 텍스트가 붙어버리는 경우가 있어
# (예: "71곳 산정" + "5034곳 산정" -> 숫자 결합) 물리적으로 불가능한 값이 섞여 나온다.
# 실제 프랜차이즈가 절대 넘지 않을 여유 있는 상한선으로 이상치만 걸러내고 NaN 처리한다.
SANITY_CAP_MANWON = {
    "average_annual_sales": 500_000,  # 매장당 연매출 50억원
    "maximum_annual_sales": 500_000,
    "minimum_annual_sales": 500_000,
    "average_sales_per_3_3sqm": 50_000,  # 3.3㎡당 연매출 5억원
    "minimum_startup_total": 100_000,  # 창업비용 10억원
    "maximum_startup_total": 100_000,
}
SANITY_CAP_STORE_COUNT = 50_000

# 매출은 상한뿐 아니라 하한도 필요하다 — 연매출 47만원처럼 원본 공시 자체의 오기로 보이는
# 비현실적으로 작은 값도 실제로 나왔다. 개인 소자본 창업도 연 500만원 밑으로는 사실상 없다고 보고
# 아주 여유 있게 잡은 하한선이다(진짜 작은 매장도 안전하게 통과하도록).
SANITY_FLOOR_MANWON = {
    "average_annual_sales": 500,
    "maximum_annual_sales": 500,
    "minimum_annual_sales": 500,
}


def _null_out_implausible(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col, cap in SANITY_CAP_MANWON.items():
        if col in df.columns:
            df.loc[df[col] > cap, col] = None
    for col, floor in SANITY_FLOOR_MANWON.items():
        if col in df.columns:
            df.loc[df[col] < floor, col] = None
    for col in ("store_count", "franchise_count", "direct_count", "total_unit_count"):
        if col in df.columns:
            df.loc[df[col] > SANITY_CAP_STORE_COUNT, col] = None
    return df


def _fetch_by_disclosure(view_name: str, disclosure_id: str) -> pd.DataFrame:
    query = text(f"SELECT * FROM {view_name} WHERE disclosure_id = :disclosure_id")
    with get_engine().connect() as conn:
        return pd.read_sql(query, conn, params={"disclosure_id": disclosure_id})


def get_startup_costs(disclosure_id: str) -> pd.DataFrame:
    df = _fetch_by_disclosure("v_agent_startup_costs", disclosure_id)
    df = _convert_to_manwon(
        df, ["amount_numeric", "amount_per_3_3sqm_numeric"], "amount_unit", default_unit="THOUSAND_KRW"
    )
    if not df.empty:
        df.loc[df["amount_numeric"] > 100_000, "amount_numeric"] = None
    return df


def get_sales(disclosure_id: str) -> pd.DataFrame:
    df = _fetch_by_disclosure("v_agent_sales", disclosure_id)
    money_cols = [
        "average_annual_sales",
        "average_sales_per_3_3sqm",
        "maximum_annual_sales",
        "minimum_annual_sales",
    ]
    df = _convert_to_manwon(df, money_cols, "sales_unit")
    return _null_out_implausible(df)


def get_operating_burdens(disclosure_id: str) -> pd.DataFrame:
    return _fetch_by_disclosure("v_agent_operating_burdens", disclosure_id)


def get_support(disclosure_id: str) -> pd.DataFrame:
    return _fetch_by_disclosure("v_agent_support", disclosure_id)


def get_contract_exit(disclosure_id: str) -> pd.DataFrame:
    return _fetch_by_disclosure("v_agent_contract_exit", disclosure_id)
