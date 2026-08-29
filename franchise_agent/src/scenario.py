import re
from typing import Optional

from .db import MONEY_UNIT_TO_MANWON, get_sales, get_startup_costs
from .finance import generate_amortization_schedule
from .profile import CustomerProfile, DerivedMetrics

DEFAULT_OPERATING_MARGIN = 0.15  # 매출 대비 영업이익률 가정치 (DB에 원가정보가 없어 사용하는 근사값)
RAMP_UP_MONTHS = 6
SIMULATION_MONTHS = 36
DEFAULT_GRACE_MONTHS = 6  # 거치후분할상환 선택 시, 설문에 거치기간이 없어 기본값으로 사용


def funding_structure(profile: CustomerProfile) -> dict:
    """설문에서 자금 출처를 항목별로 안 받기로 했으므로(단순화), 즉시 가용자금은 전부 자기자본,
    희망 대출액은 전부 부채로 취급한다 — "본인 자금 중 투자 최대금액" 같은 세부 항목은 더 이상
    없다."""
    equity = profile.liquid_capital
    debt = profile.new_loan.desired_amount or 0.0
    total = equity + debt

    return {
        "breakdown": {"본인 자금": equity, "대출": debt} if debt else {"본인 자금": equity},
        "equity": equity,
        "debt": debt,
        "total": total,
        "debt_ratio": (debt / total) if total else None,
    }


# amount_numeric이 NULL인 원문(amount_raw)은 대부분 세 가지 패턴 중 하나다:
# (1) '[ 2,000 ]'/'[171,100]'처럼 계산값 표시로 대괄호를 쳐서 숫자 파서가 못 읽은 경우,
# (2) '33,250~38,000', '[96,250 – 98,450]'처럼 구간값이거나 '36,300(가스용) 38,500(전기용)'처럼
#     조건별 복수값인 경우 — 대표값으로 더 큰 쪽을 쓴다(고객에게 안전한 쪽으로 보수적으로 추정),
# (3) '\ 38 , 5 00,000원'처럼 PDF 추출 과정에서 숫자 중간에 공백이 끼어든 경우,
# (4) '200만원'처럼 단위가 컬럼이 아니라 원문 텍스트 안에 직접 적혀 있는 경우.
# 이 네 가지는 전부 규칙만으로 복원 가능해서 LLM 없이 정규식으로 처리한다.
_WS_BETWEEN_DIGITS_RE = re.compile(r"(?<=[\d,])\s+(?=[\d,])")
# 단위 글자 사이에도 OCR 공백이 낄 수 있다('만 원' 처럼) — 각 글자 사이에 \s*를 허용한다.
# 그렇게 안 하면 '만원'을 못 찾고 컬럼의 amount_unit(예: 억원)을 엉뚱하게 적용해
# 2,000(만원)을 2,000억원으로 읽는 등 자릿수가 완전히 틀어질 수 있다.
_EMBEDDED_UNIT_RE = re.compile(r"([\d,]{2,})\s*(억\s*원|백\s*만\s*원|천\s*원|만\s*원|원)")
_PLAIN_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d{4,}")

_EMBEDDED_UNIT_TO_MANWON = {"억원": 10000.0, "백만원": 100.0, "천원": 0.1, "만원": 1.0, "원": 0.0001}
# 이 함수가 만들어낼 수 있는 최댓값의 안전장치. db.py의 SANITY_CAP_MANWON(창업비용 10억원)과
# 동일한 기준 — 파싱 로직에 미처 못 잡은 예외가 있어도 터무니없는 값이 화면에 뜨는 것을 막는다.
_PARSE_SANITY_CAP_MANWON = 100_000


def _parse_amount_raw(raw: Optional[str], unit: Optional[str]) -> Optional[float]:
    """amount_raw 원문에서 대표 금액을 만원 단위로 복원한다. 여러 값(구간·조건별 병기)이
    있으면 더 큰 값을 대표값으로 쓴다. 짧은 숫자(예: '9 대 이상'의 '9')는 콤마 구분자나 단위
    표기가 없으면 금액으로 보지 않고 버린다 — 오탐(false positive)을 피하기 위함이다.
    단위가 NULL이면 db.py의 다른 곳과 동일하게 THOUSAND_KRW로 가정한다(전체 DB 조사 결과
    NULL 단위의 약 82%가 실제로는 천원 단위였음)."""
    if not raw:
        return None
    cleaned = _WS_BETWEEN_DIGITS_RE.sub("", raw)

    embedded = [
        float(m.group(1).replace(",", "")) * _EMBEDDED_UNIT_TO_MANWON[re.sub(r"\s+", "", m.group(2))]
        for m in _EMBEDDED_UNIT_RE.finditer(cleaned)
        if m.group(1).replace(",", "")
    ]
    if embedded:
        value = max(embedded)
        return value if value <= _PARSE_SANITY_CAP_MANWON else None

    factor = MONEY_UNIT_TO_MANWON.get(unit or "THOUSAND_KRW")
    if factor is None:
        return None
    values = [float(m.group(0).replace(",", "")) for m in _PLAIN_NUMBER_RE.finditer(cleaned)]
    if not values:
        return None
    value = max(values) * factor
    return value if value <= _PARSE_SANITY_CAP_MANWON else None


def _group_total(df_group) -> float:
    """정보공개서 표는 보통 항목별 행 + '합계/총계' 행을 함께 담고 있어
    전부 더하면 이중 집계된다. 합계 행이 있으면 그 값을 쓰고(숫자로 못 읽었으면 원문에서 복원),
    없으면 항목별 합으로 대체한다. 항목별 행도 amount_numeric이 비어 있으면 원문에서 복원해서
    합산한다 — 안 그러면 파싱 실패한 항목이 조용히 누락돼 총액이 실제보다 작게 잡힌다."""
    if df_group.empty:
        return 0.0
    normalized = df_group["cost_item"].fillna("").str.replace(" ", "", regex=False)
    is_total_row = normalized.str.contains("합계") | normalized.str.contains("총계")
    total_rows = df_group[is_total_row]
    if not total_rows.empty:
        numeric_totals = total_rows["amount_numeric"].dropna()
        if not numeric_totals.empty:
            return float(numeric_totals.max())
        derived_values = [
            v
            for v in (
                _parse_amount_raw(row.get("amount_raw"), row.get("amount_unit")) for _, row in total_rows.iterrows()
            )
            if v is not None
        ]
        if derived_values:
            return max(derived_values)

    item_rows = df_group[~is_total_row]
    numeric_sum = item_rows["amount_numeric"].dropna().sum()
    missing = item_rows[item_rows["amount_numeric"].isna()]
    recovered_sum = sum(
        v
        for v in (
            _parse_amount_raw(row.get("amount_raw"), row.get("amount_unit")) for _, row in missing.iterrows()
        )
        if v is not None
    )
    return float(numeric_sum + recovered_sum)


def total_startup_cost(disclosure_id: str, other_cost_estimate: Optional[float]) -> dict:
    df = get_startup_costs(disclosure_id)
    fee_groups = ["INITIAL_FRANCHISE_FEE", "DEPOSIT", "ESCROW_FRANCHISE_FEE"]
    fee_detail: dict[str, float] = {}
    for group in fee_groups:
        fee_detail[group] = _group_total(df[df["cost_group"] == group]) if not df.empty else 0.0

    # ESCROW_FRANCHISE_FEE(예치가맹금)는 INITIAL_FRANCHISE_FEE(최초가맹금) 중 예치기관에
    # 맡겨야 하는 금액을 별도 목차로 공시한 것뿐, 추가로 더 내는 돈이 아니다.
    # INITIAL_FRANCHISE_FEE가 비어 있을 때만 대체값으로 사용한다.
    initial_fee = fee_detail["INITIAL_FRANCHISE_FEE"] or fee_detail["ESCROW_FRANCHISE_FEE"]
    chargeable = initial_fee + fee_detail["DEPOSIT"]

    other_cost = _group_total(df[df["cost_group"] == "OTHER_STARTUP_COST"]) if not df.empty else 0.0
    if not other_cost and other_cost_estimate:
        other_cost = float(other_cost_estimate)
    total = chargeable + other_cost
    # 실제로 비용이 0원인 프랜차이즈는 없다 — 항목이 하나도 안 잡히면 "0원"이 아니라
    # "정보 없음"이다. 원본 표 파싱이 안 됐거나(구간값 등) 정보공개서에 해당 항목이 없는 경우.
    # 반대로 지역별 대안 옵션(활성지역/안착지역 등)을 하나로 합산하는 과정에서처럼 항목별
    # 합산이 비정상적으로 커질 수도 있어 상한선(10억원)을 넘으면 신뢰할 수 없는 값으로 본다.
    has_data = not df.empty and 0 < total <= _PARSE_SANITY_CAP_MANWON

    return {"fees": fee_detail, "other_cost": other_cost, "total": total, "has_data": has_data}


def get_sales_scenarios(disclosure_id: str) -> dict:
    df = get_sales(disclosure_id)
    if df.empty:
        return {"optimistic": None, "base": None, "pessimistic": None}

    national = df[df["region_name"].str.replace(" ", "", regex=False).isin(["전국", "전체"])]
    row = national.iloc[0] if not national.empty else df.iloc[0]

    return {
        "optimistic": row.get("maximum_annual_sales"),
        "base": row.get("average_annual_sales"),
        "pessimistic": row.get("minimum_annual_sales"),
    }


def simulate_scenario(
    annual_sales: Optional[float],
    loan_schedule: list[dict],
    monthly_repayment_capacity: float,
    initial_reserve: float,
    total_investment: Optional[float],
    operating_margin: float = DEFAULT_OPERATING_MARGIN,
    months: int = SIMULATION_MONTHS,
    ramp_up_months: int = RAMP_UP_MONTHS,
) -> dict:
    if not annual_sales:
        return {"rows": [], "breakeven_month": None, "payback_month": None, "runway_month": None}

    target_monthly_sales = annual_sales / 12
    payments = [row["payment"] for row in loan_schedule]

    cumulative_profit = 0.0
    reserve_balance = initial_reserve
    breakeven_month = None
    payback_month = None
    runway_month = None
    rows = []

    for m in range(1, months + 1):
        sales = target_monthly_sales * min(1.0, m / ramp_up_months)
        operating_profit = sales * operating_margin
        loan_payment = payments[m - 1] if m - 1 < len(payments) else 0.0
        store_net = operating_profit - loan_payment
        household_net = store_net + monthly_repayment_capacity

        cumulative_profit += operating_profit
        reserve_balance += household_net

        if breakeven_month is None and store_net >= 0:
            breakeven_month = m
        if payback_month is None and total_investment is not None and cumulative_profit >= total_investment:
            payback_month = m
        if runway_month is None and reserve_balance < 0:
            runway_month = m

        rows.append(
            {
                "month": m,
                "sales": sales,
                "operating_profit": operating_profit,
                "loan_payment": loan_payment,
                "store_net": store_net,
                "household_net": household_net,
                "reserve_balance": reserve_balance,
            }
        )

    return {
        "rows": rows,
        "breakeven_month": breakeven_month,
        "payback_month": payback_month,
        "runway_month": runway_month,
    }


def build_financial_report(
    disclosure_id: str,
    candidate: dict,
    profile: CustomerProfile,
    derived: DerivedMetrics,
) -> dict:
    funding = funding_structure(profile)
    costs = total_startup_cost(disclosure_id, candidate.get("maximum_startup_total"))
    shortfall = max(0.0, costs["total"] - derived.available_startup_budget) if costs["has_data"] else None

    loan = profile.new_loan
    grace = DEFAULT_GRACE_MONTHS if loan.repayment_method == "거치 후 분할상환" else 0
    loan_schedule = generate_amortization_schedule(
        loan.desired_amount, loan.expected_interest_rate, loan.repayment_period_months, loan.repayment_method, grace
    )
    loan_summary = {
        "monthly_payment_first": loan_schedule[0]["payment"] if loan_schedule else None,
        "total_payment": sum(r["payment"] for r in loan_schedule) if loan_schedule else None,
        "total_interest": sum(r["interest"] for r in loan_schedule) if loan_schedule else None,
    }

    sales_scenarios = get_sales_scenarios(disclosure_id)
    scenarios = {}
    for label, annual_sales in [
        ("optimistic", sales_scenarios["optimistic"]),
        ("base", sales_scenarios["base"]),
        ("pessimistic", sales_scenarios["pessimistic"]),
    ]:
        scenarios[label] = simulate_scenario(
            annual_sales=annual_sales,
            loan_schedule=loan_schedule,
            monthly_repayment_capacity=derived.monthly_repayment_capacity,
            initial_reserve=profile.operating_reserve,
            total_investment=costs["total"] if costs["has_data"] else None,
        )

    all_unsuitable = all(s["breakeven_month"] is None or s["runway_month"] is not None for s in scenarios.values())
    # 창업비용의 10%를 넘는 부족분은 시나리오 결과와 무관하게 그 자체로 감당 불가능한 조건이다
    # (비용 정보 자체가 없으면 부족분을 판단할 수 없으니 판단을 보류한다)
    meaningful_shortfall = shortfall is not None and shortfall / costs["total"] > 0.10
    needs_recommendation_loop = all_unsuitable or meaningful_shortfall

    return {
        "disclosure_id": disclosure_id,
        "funding": funding,
        "costs": costs,
        "shortfall": shortfall,
        "loan_schedule": loan_schedule,
        "loan_summary": loan_summary,
        "scenarios": scenarios,
        "needs_recommendation_loop": needs_recommendation_loop,
    }
