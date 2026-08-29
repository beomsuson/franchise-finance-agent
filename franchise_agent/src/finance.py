from typing import Optional

# repayment_method values, matching 질문파일.pdf section 2-1-5
EQUAL_PRINCIPAL_INTEREST = "원리금균등상환"
EQUAL_PRINCIPAL = "원금균등상환"
BULLET = "만기일시상환"
GRACE_THEN_AMORTIZE = "거치 후 분할상환"
OTHER = "기타"
UNKNOWN = "아직 모름"

REPAYMENT_METHOD_OPTIONS = [
    EQUAL_PRINCIPAL_INTEREST,
    EQUAL_PRINCIPAL,
    BULLET,
    GRACE_THEN_AMORTIZE,
    OTHER,
    UNKNOWN,
]


def estimate_monthly_payment(
    principal: Optional[float],
    annual_rate_pct: Optional[float],
    repayment_period_months: Optional[int],
    repayment_method: Optional[str],
) -> Optional[float]:
    """Rough average monthly payment (만원) for early-stage budgeting.
    Not an exact amortization schedule — that belongs to the stage-4 scenario simulator.
    """
    if not principal or principal <= 0:
        return 0.0
    if not repayment_period_months or repayment_period_months <= 0:
        return None
    if annual_rate_pct is None or repayment_method in (None, OTHER, UNKNOWN):
        return None

    r = (annual_rate_pct / 100) / 12
    n = repayment_period_months

    if repayment_method == EQUAL_PRINCIPAL_INTEREST:
        if r == 0:
            return principal / n
        return principal * r / (1 - (1 + r) ** -n)

    if repayment_method in (EQUAL_PRINCIPAL, GRACE_THEN_AMORTIZE):
        # average payment across the schedule: principal repaid evenly,
        # interest charged on the (linearly declining) average balance
        return principal / n + (principal * r) / 2

    if repayment_method == BULLET:
        return principal * r

    return None


def generate_amortization_schedule(
    principal: Optional[float],
    annual_rate_pct: Optional[float],
    months: Optional[int],
    repayment_method: Optional[str],
    grace_months: int = 0,
) -> list[dict]:
    """Month-by-month payment/interest/principal/balance. 만원 단위.
    거치후분할상환의 거치기간은 설문에 없는 값이라 grace_months로 별도 지정(기본 0=거치없음 취급).
    """
    if not principal or principal <= 0 or not months or months <= 0:
        return []
    if annual_rate_pct is None or repayment_method in (None, OTHER, UNKNOWN):
        return []

    r = (annual_rate_pct / 100) / 12
    balance = principal
    schedule: list[dict] = []

    if repayment_method == EQUAL_PRINCIPAL_INTEREST:
        payment = estimate_monthly_payment(principal, annual_rate_pct, months, repayment_method)
        for m in range(1, months + 1):
            interest = balance * r
            principal_paid = min(payment - interest, balance)
            balance = max(balance - principal_paid, 0.0)
            schedule.append(
                {"month": m, "payment": payment, "interest": interest, "principal_paid": principal_paid, "balance": balance}
            )

    elif repayment_method == BULLET:
        for m in range(1, months + 1):
            interest = balance * r
            principal_paid = principal if m == months else 0.0
            balance = balance - principal_paid
            schedule.append(
                {
                    "month": m,
                    "payment": interest + principal_paid,
                    "interest": interest,
                    "principal_paid": principal_paid,
                    "balance": balance,
                }
            )

    elif repayment_method in (EQUAL_PRINCIPAL, GRACE_THEN_AMORTIZE):
        g = grace_months if repayment_method == GRACE_THEN_AMORTIZE else 0
        amortize_months = max(months - g, 1)
        principal_per_month = principal / amortize_months
        for m in range(1, months + 1):
            interest = balance * r
            principal_paid = 0.0 if m <= g else min(principal_per_month, balance)
            balance = max(balance - principal_paid, 0.0)
            schedule.append(
                {
                    "month": m,
                    "payment": interest + principal_paid,
                    "interest": interest,
                    "principal_paid": principal_paid,
                    "balance": balance,
                }
            )

    return schedule
