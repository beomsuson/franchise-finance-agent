import pandas as pd

from .scenario import DEFAULT_OPERATING_MARGIN


def _safe(value, default=None):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    return value


def compute_brand_risk_breakdown(candidate: dict) -> dict:
    """0(안전) ~ 100(위험) 브랜드 리스크 점수를 항목별로 분해해서 반환.
    v_agent_candidate_features 행 기준. 고객에게 '왜 이 점수인지' 보여줄 때 쓴다."""
    closure_rate = _safe(candidate.get("closure_termination_rate"), 0.15)
    closure_component = round(min(closure_rate * 200, 60), 1)

    no_credit_support = _safe(candidate.get("has_credit_support")) != 1
    no_operation_support = _safe(candidate.get("has_operation_support")) != 1
    no_contract_info = _safe(candidate.get("has_contract_exit_information")) != 1
    completeness = _safe(candidate.get("feature_completeness_score"), 100)
    low_completeness = completeness < 60

    total = closure_component
    total += 15 if no_credit_support else 0
    total += 10 if no_operation_support else 0
    total += 5 if no_contract_info else 0
    total += 10 if low_completeness else 0

    return {
        "total": round(min(total, 100), 1),
        "closure_termination_rate": closure_rate,
        "closure_component": closure_component,
        "no_credit_support": no_credit_support,
        "no_operation_support": no_operation_support,
        "no_contract_info": no_contract_info,
        "low_completeness": low_completeness,
    }


def compute_brand_risk_score(candidate: dict) -> float:
    """0(안전) ~ 100(위험) 브랜드 리스크 점수 (단일 값만 필요할 때)."""
    return compute_brand_risk_breakdown(candidate)["total"]


def brand_fit_reasons(candidate: dict, derived: dict) -> list[str]:
    """폐업률뿐 아니라 예산 여유·수익 커버리지·지원제도까지 종합해서
    '왜 이 브랜드가 이 고객에게 맞는지'를 사람이 읽을 수 있는 근거로 만든다.
    전부 결정론적 계산이라 고객에게 그대로 보여줘도 근거가 명확하다."""
    breakdown = compute_brand_risk_breakdown(candidate)
    reasons = []

    rate_pct = breakdown["closure_termination_rate"] * 100
    if candidate.get("closure_termination_rate") is not None:
        level = "낮은 편" if rate_pct < 10 else ("평균적" if rate_pct < 20 else "다소 높은 편")
        reasons.append(f"최근 3년 폐업·해지율 {rate_pct:.1f}% ({level})")
    else:
        reasons.append("폐업·해지율 데이터가 없어 평균치(15%)로 보수적으로 계산함")

    reasons.append("가맹본부 금융지원 제도 " + ("있음" if not breakdown["no_credit_support"] else "없음"))
    reasons.append("가맹본부 운영지원 제도 " + ("있음" if not breakdown["no_operation_support"] else "없음"))

    budget = derived.get("available_startup_budget")
    cost = candidate.get("maximum_startup_total")
    if budget and cost:
        margin_pct = (budget - cost) / budget * 100
        if margin_pct >= 0:
            reasons.append(f"조달 가능액 대비 창업비용 여유 {margin_pct:.0f}%")
        else:
            reasons.append(f"조달 가능액보다 창업비용이 {-margin_pct:.0f}% 더 큼 (예산 빠듯함)")

    required = derived.get("required_monthly_net_profit")
    sales = candidate.get("average_annual_sales")
    if required and required > 0 and sales:
        # average_annual_sales는 매출(revenue)이지 순이익이 아니다 — 매출을 그대로 필요 순이익과
        # 비교하면 항상 몇 배씩 부풀려진 숫자가 나온다(예: 1000%대). scenario.py의 시나리오
        # 시뮬레이션과 동일한 영업이익률 가정(15%)을 적용해 추정 순이익으로 환산한 뒤 비교한다.
        estimated_monthly_profit = (sales / 12) * DEFAULT_OPERATING_MARGIN
        coverage_pct = estimated_monthly_profit / required * 100
        reasons.append(
            f"브랜드 평균 매출 기준 추정 월순이익(영업이익률 {DEFAULT_OPERATING_MARGIN:.0%} 가정)이 "
            f"고객이 필요로 하는 월순이익의 약 {coverage_pct:.0f}% 수준"
        )

    return reasons
