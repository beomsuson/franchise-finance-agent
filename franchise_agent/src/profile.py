from dataclasses import dataclass, field
from typing import Optional

from .finance import estimate_monthly_payment

# ---- 질문파일.pdf 옵션 목록 ----

# 원래 7단계였는데("아직 확인 안함"~"승인 완료" 등), 모델(brand_recommender.py)이 실제로
# 구분하는 건 "승인 완료냐 아니냐" 이 두 갈래뿐이라 설문을 이분법으로 단순화했다.
LOAN_STATUS_OPTIONS = ["대출 가능", "대출 불가능/미정"]

OPERATION_PERIOD_OPTIONS = ["1년 미만", "1~3년", "3~5년", "5~10년", "10년 이상", "아직 정하지 않음"]

STARTUP_TIMING_OPTIONS = ["1년 이내", "아직 미정"]


@dataclass
class PlannedExpense:
    purpose: str
    amount: float  # 만원
    expected_timing: str


@dataclass
class NewLoanPlan:
    desired_amount: Optional[float] = None  # 만원
    status: str = "대출 불가능/미정"
    expected_interest_rate: Optional[float] = None  # 연 %
    repayment_period_months: Optional[int] = None
    repayment_method: Optional[str] = None


@dataclass
class ExistingDebt:
    total_amount: float = 0.0  # 만원
    monthly_payment: float = 0.0  # 만원


@dataclass
class CustomerProfile:
    # 1. 창업 자금 — "본인 투자 최대금액"·"자금 출처별 금액"은 설문에서 뺐다(단순화). 즉시
    # 가용자금 전액을 자기자본으로, 대출 희망액을 부채로 본다(scenario.funding_structure 참고).
    liquid_capital: float  # 즉시 가용 총자금 (만원)
    operating_reserve: float  # 운영 예비자금 (만원)

    # 2. 대출
    new_loan: NewLoanPlan = field(default_factory=NewLoanPlan)
    existing_debt: ExistingDebt = field(default_factory=ExistingDebt)

    # 3. 생계비
    min_monthly_living_cost: float = 0.0  # 만원
    maintained_monthly_income: float = 0.0  # 만원
    planned_major_expenses: list[PlannedExpense] = field(default_factory=list)

    # 4. 기타 사항
    existing_store_count: int = 0
    existing_store_monthly_cashflow: Optional[float] = None  # 만원, 적자면 음수
    planned_operation_period: str = "아직 정하지 않음"
    target_monthly_income: Optional[float] = None  # 만원
    target_payback_period_years: Optional[float] = None
    startup_timing: str = "아직 미정"

    # 업종/희망 조건 (필터링용, 설문에 없지만 UI에서 별도로 받음). 복수 선택 가능 —
    # 예: "치킨"과 "카페"를 동시에 고려 중인 고객도 있어서, 에이전트가 업종별로 각각
    # search_candidates를 호출해 후보를 모은 뒤 통합 비교하도록 한다.
    desired_industries: list[str] = field(default_factory=list)


@dataclass
class DerivedMetrics:
    available_startup_budget: float  # 가용 창업예산
    monthly_repayment_capacity: float  # 월 상환여력
    estimated_new_loan_monthly_payment: Optional[float]  # 신규대출 예상 월상환액
    required_monthly_net_profit: Optional[float]  # 필요 월 순이익
    target_payback_period_years: Optional[float]  # 목표 회수기간


def compute_derived_metrics(profile: CustomerProfile) -> DerivedMetrics:
    available_startup_budget = profile.liquid_capital

    monthly_repayment_capacity = (
        profile.maintained_monthly_income
        - profile.min_monthly_living_cost
        - profile.existing_debt.monthly_payment
    )

    estimated_new_loan_monthly_payment = estimate_monthly_payment(
        profile.new_loan.desired_amount,
        profile.new_loan.expected_interest_rate,
        profile.new_loan.repayment_period_months,
        profile.new_loan.repayment_method,
    )

    # 필요 월 순이익 = (대출 상환액이 가구 여유자금을 초과하는 만큼) + (목표 세후 월소득).
    # 예전엔 목표 세후 월소득(target_monthly_income)이 계산에서 아예 빠져 있었다 — 설문에서
    # 받아놓고 안 쓰던 버그. monthly_repayment_capacity(생계비·기존부채 제하고 남는 돈)로 대출을
    # 감당하고 남거나 모자라는 만큼을 먼저 따지고, 거기에 고객이 이 사업에서 추가로 벌고 싶어하는
    # 목표소득을 더해야 "이 브랜드가 실제로 얼마를 벌어줘야 하는지"가 정확해진다.
    target_income = profile.target_monthly_income or 0.0
    if estimated_new_loan_monthly_payment is None and target_income == 0.0:
        required_monthly_net_profit = None
    else:
        loan_shortfall = max(0.0, (estimated_new_loan_monthly_payment or 0.0) - monthly_repayment_capacity)
        required_monthly_net_profit = loan_shortfall + target_income

    return DerivedMetrics(
        available_startup_budget=available_startup_budget,
        monthly_repayment_capacity=monthly_repayment_capacity,
        estimated_new_loan_monthly_payment=estimated_new_loan_monthly_payment,
        required_monthly_net_profit=required_monthly_net_profit,
        target_payback_period_years=profile.target_payback_period_years,
    )


def profile_from_dict(d: dict) -> CustomerProfile:
    """LangGraph 상태(state)는 dataclass가 아니라 plain dict로 오가므로,
    dataclasses.asdict()로 평탄화된 dict를 다시 CustomerProfile로 복원한다."""
    return CustomerProfile(
        liquid_capital=d["liquid_capital"],
        operating_reserve=d["operating_reserve"],
        new_loan=NewLoanPlan(**d["new_loan"]) if d.get("new_loan") else NewLoanPlan(),
        existing_debt=ExistingDebt(**d["existing_debt"]) if d.get("existing_debt") else ExistingDebt(),
        min_monthly_living_cost=d.get("min_monthly_living_cost", 0.0),
        maintained_monthly_income=d.get("maintained_monthly_income", 0.0),
        planned_major_expenses=[PlannedExpense(**pe) for pe in d.get("planned_major_expenses", [])],
        existing_store_count=d.get("existing_store_count", 0),
        existing_store_monthly_cashflow=d.get("existing_store_monthly_cashflow"),
        planned_operation_period=d.get("planned_operation_period", "아직 정하지 않음"),
        target_monthly_income=d.get("target_monthly_income"),
        target_payback_period_years=d.get("target_payback_period_years"),
        startup_timing=d.get("startup_timing", "아직 미정"),
        desired_industries=d.get("desired_industries", []),
    )


def derived_from_dict(d: dict) -> DerivedMetrics:
    return DerivedMetrics(**d)
