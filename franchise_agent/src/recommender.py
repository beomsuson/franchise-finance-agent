"""① 설문 → ② 1차 필터링을 대체하는, 미리 학습된 브랜드 추천 모델(brand_recommender_runtime)
과의 연결부. 예전엔 SQL 조건(예산·업종)으로 후보를 걸러냈지만, 이제는 439개 브랜드로 학습된
CatBoost 랭킹 모델이 고객 프로필을 보고 1~4개 브랜드를 직접 추천한다.

모델은 disclosure_id를 모르고 brand_name만 안다 — 733개 DB 브랜드 중 439개와 완전히
동일한 표기로 겹친다는 걸 미리 확인했다(check_model_brands.py, 100% exact match).
그래서 recommend_survey_candidates()가 brand_name으로 v_agent_brand_overview에서
disclosure_id를 찾아 이어붙인다."""

import sys
from functools import lru_cache
from pathlib import Path

# brand_recommender_runtime은 franchise_agent와 같은 프로젝트 루트 아래 있는 별도 패키지라
# 여기서만 sys.path에 추가한다 — franchise_agent 패키지 안으로 옮기지 않은 이유는, 팀원이
# 준 원본 폴더 구조(README에 "파일 이름과 폴더 구조를 변경하지 마세요")를 그대로 유지하기 위함.
_RECOMMENDER_DIR = Path(__file__).resolve().parent.parent.parent / "brand_recommender_runtime"
if str(_RECOMMENDER_DIR) not in sys.path:
    sys.path.insert(0, str(_RECOMMENDER_DIR))

from brand_recommender import BrandRecommender  # noqa: E402

from .db import get_engine  # noqa: E402
from .scenario import total_startup_cost  # noqa: E402

# 모델의 loan_status/actual_startup_timing_months는 설문 옵션과 이름이 달라서 매핑이 필요하다.
# 설문은 "대출 가능/불가능·미정" 이분법으로 단순화했는데, 모델이 실제로 구분하는 것도
# "승인 완료냐 아니냐" 이 두 갈래뿐이라(budget()의 loan_weight 계산 참고) 정보 손실이 없다.
_LOAN_STATUS_MAP = {"대출 가능": "승인 완료", "대출 불가능/미정": "검토 중"}
# "1년 이내"는 중간값인 6개월로, "아직 미정"은 넉넉히 24개월로 어림한다(모델이 숫자를 요구하는데
# 설문은 두 구간짜리 선택지라 정확한 개월 수를 알 수 없음 — 근사치임을 명확히 문서화해둔다).
_STARTUP_TIMING_MONTHS = {"1년 이내": 6, "아직 미정": 24}


@lru_cache(maxsize=1)
def get_recommender() -> BrandRecommender:
    return BrandRecommender(artifacts_dir=_RECOMMENDER_DIR / "artifacts")


def model_industries() -> list[str]:
    """설문의 업종 선택지는 이 목록에서만 골라야 한다 — DB 전체 업종 목록과 살짝 다르고
    (모델은 439개 브랜드 기준 34개 업종만 앎), 여기 없는 값을 넣으면 선호도 매칭이 안 된다."""
    return get_recommender().industries()


def profile_to_model_input(profile) -> dict:
    """CustomerProfile(dataclass) -> brand_recommender가 기대하는 평평한 dict.
    금액 단위는 이미 둘 다 만원이라 그대로 옮기면 되고, 대출상태·창업시기처럼 옵션 문구가
    다른 항목만 위 매핑 테이블을 거친다. 설문에서 "본인 투자 최대금액"·자금 출처별 세부금액을
    더 이상 안 받으므로(단순화), 자기자본 관련 입력은 전부 즉시 가용자금(liquid_capital) 하나로
    본다 — max_self_investment_10k_krw도 같은 값을 넣는다(모델이 두 필드를 각각 요구하지만
    실질적으로 같은 자금을 가리킴)."""
    industries = (profile.desired_industries or [])[:3]
    industries += [""] * (3 - len(industries))

    return {
        "immediate_available_capital_10k_krw": profile.liquid_capital,
        "max_self_investment_10k_krw": profile.liquid_capital,
        "operating_reserve_10k_krw": profile.operating_reserve,
        "self_savings_amount_10k_krw": profile.liquid_capital,
        "funding_family_flag": 0,
        "family_funding_amount_10k_krw": 0,
        "funding_investor_flag": 0,
        "investor_funding_amount_10k_krw": 0,
        "funding_loan_flag": 1 if profile.new_loan.desired_amount else 0,
        "loan_status": _LOAN_STATUS_MAP.get(profile.new_loan.status, "검토 중"),
        "desired_loan_amount_10k_krw": profile.new_loan.desired_amount or 0,
        "expected_interest_rate_pct": profile.new_loan.expected_interest_rate or 0,
        "loan_term_months": profile.new_loan.repayment_period_months or 0,
        "current_debt_total_10k_krw": profile.existing_debt.total_amount,
        "current_monthly_debt_payment_10k_krw": profile.existing_debt.monthly_payment,
        "minimum_monthly_living_cost_10k_krw": profile.min_monthly_living_cost,
        "post_startup_household_monthly_after_tax_income_10k_krw": profile.maintained_monthly_income,
        "current_franchise_store_count": profile.existing_store_count,
        "existing_store_monthly_net_profit_10k_krw": profile.existing_store_monthly_cashflow or 0,
        "target_monthly_after_tax_income_10k_krw": profile.target_monthly_income or 0,
        "target_payback_years": profile.target_payback_period_years or 0,
        "actual_startup_timing_months": _STARTUP_TIMING_MONTHS.get(profile.startup_timing, 24),
        "preferred_industry_1": industries[0],
        "preferred_industry_2": industries[1],
        "preferred_industry_3": industries[2],
    }


def recommend_survey_candidates(profile) -> list[dict]:
    """모델이 뽑은 1~4개 브랜드를 돌려준다. ② 화면에 보여주는 근거 숫자(창업비용/평당매출/가맹점수)는
    전부 브랜드데이터.csv(=metadata.joblib) 값 그대로다 — SQL은 이 단계에서 전혀 참고하지 않는다.
    DB는 disclosure_id를 찾는 용도로만 딱 한 번 쓰는데, 이건 ③단계에서 LLM 에이전트가 정보공개서
    원문(get_contract_risks 등)을 조회할 때 필요한 키일 뿐, 화면 숫자와는 무관하다. closure_
    termination_rate처럼 CSV에 아예 없는 값은 이 단계에서 채우지 않고 None으로 둔다 — ③단계 이후
    finalize 시점에 build_brand_detail()이 DB에서 따로 채워준다."""
    recommender = get_recommender()
    rows = recommender.recommend_one(profile_to_model_input(profile))
    brand_names = [r["brand_name"] for r in rows]
    if not brand_names:
        return []

    from sqlalchemy import bindparam, text

    query = text(
        "SELECT disclosure_id, brand_name FROM v_agent_brand_overview WHERE brand_name IN :names"
    ).bindparams(bindparam("names", expanding=True))
    with get_engine().connect() as conn:
        import pandas as pd

        db_df = pd.read_sql(query, conn, params={"names": brand_names})
    disclosure_id_by_name = dict(zip(db_df["brand_name"], db_df["disclosure_id"]))

    candidates = []
    for r in rows:
        did = disclosure_id_by_name.get(r["brand_name"])
        if did is None:
            continue
        csv_brand = recommender.brands.get(r["brand_name"], {})
        candidates.append(
            {
                "disclosure_id": did,
                "brand_name": r["brand_name"],
                "industry_middle": r.get("industry_middle"),
                # 아래 두 값은 SQL 재계산 없이 브랜드데이터.csv 값을 그대로 쓴다(사용자 요청).
                "maximum_startup_total": r.get("startup_cost_10k_krw"),  # = CSV '총계' / 10
                # CSV의 average_sales_per_3_3sqm은 천원 단위라 ×0.1로 만원 환산(다른 금액
                # 필드와 단위를 통일 — 만원으로 안 바꾸면 실제보다 10배 부풀려 보임).
                "average_sales_per_3_3sqm": _thousand_krw_to_manwon(csv_brand.get("average_sales_per_3_3sqm")),
                "store_count": _number_or_none(csv_brand.get("store_count")),
                # source_url이 "LLM 추정값"이면 창업비용(총계)이 실제 출처 없이 LLM이 추측한
                # 값이라는 뜻 — 442개 브랜드 중 99개(약 22%)가 여기 해당한다. 화면에 신뢰도
                # 경고를 보여주기 위해 그대로 넘긴다.
                "cost_is_estimated": csv_brand.get("source_url") == "LLM 추정값",
                # 아래는 CSV에 없는 값이라 여기서는 채우지 않는다 — finalize 단계에서 DB로 보강됨.
                "closure_termination_rate": None,
                "has_credit_support": None,
                "has_operation_support": None,
                "has_contract_exit_information": None,
                "feature_completeness_score": None,
                "average_annual_sales": None,
                "model_recommendation_rank": r["recommendation_rank"],
                "model_reason": r["reason"],
                "model_affordable": bool(r["affordable"]),
            }
        )
    return candidates


def _number_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _thousand_krw_to_manwon(value):
    n = _number_or_none(value)
    return n * 0.1 if n is not None else None


def enrich_with_db_metrics(disclosure_id: str) -> dict:
    """②미리보기는 CSV만 쓰지만, 최종 1~2개로 확정되는 순간에는 리스크 점수·추천사유 계산에
    필요한 실제 DB 지표(폐업률, 지원제도 여부, 정확한 창업비용 등)로 보강해야 품질이 안 떨어진다.
    최종 후보(많아야 2개)에만 호출하므로 SQL 비용은 미미하다."""
    from sqlalchemy import text

    query = text(
        "SELECT f.closure_termination_rate, f.feature_completeness_score, f.has_credit_support, "
        "f.has_operation_support, f.has_contract_exit_information, f.average_annual_sales "
        "FROM v_agent_candidate_features f WHERE f.disclosure_id = :d"
    )
    with get_engine().connect() as conn:
        import pandas as pd

        df = pd.read_sql(query, conn, params={"d": disclosure_id})
    metrics = df.iloc[0].to_dict() if not df.empty else {}

    cost = total_startup_cost(disclosure_id, None)
    if cost["has_data"]:
        metrics["maximum_startup_total"] = cost["total"]
    return metrics
