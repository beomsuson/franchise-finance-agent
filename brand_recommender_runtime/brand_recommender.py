import argparse
import json
import math
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRanker


BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS = BASE_DIR / "artifacts"
COST_COLUMN = "총계"
NUMERIC_USER_COLUMNS = [
    "immediate_available_capital_10k_krw", "max_self_investment_10k_krw",
    "operating_reserve_10k_krw", "self_savings_amount_10k_krw",
    "family_funding_amount_10k_krw", "investor_funding_amount_10k_krw",
    "desired_loan_amount_10k_krw", "expected_interest_rate_pct", "loan_term_months",
    "current_debt_total_10k_krw", "current_monthly_debt_payment_10k_krw",
    "minimum_monthly_living_cost_10k_krw",
    "post_startup_household_monthly_after_tax_income_10k_krw",
    "current_franchise_store_count", "existing_store_monthly_net_profit_10k_krw",
    "target_monthly_after_tax_income_10k_krw", "target_payback_years",
    "actual_startup_timing_months",
]


def number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def normalize(value):
    return " ".join(str(value).split())


def budget(profile):
    total = max(
        number(profile.get("immediate_available_capital_10k_krw")),
        number(profile.get("max_self_investment_10k_krw")),
    )
    if str(profile.get("funding_family_flag", "0")) == "1":
        total += number(profile.get("family_funding_amount_10k_krw"))
    if str(profile.get("funding_investor_flag", "0")) == "1":
        total += number(profile.get("investor_funding_amount_10k_krw"))
    if str(profile.get("funding_loan_flag", "0")) == "1":
        loan_weight = 1 if profile.get("loan_status") == "승인 완료" else 0.5
        total += number(profile.get("desired_loan_amount_10k_krw")) * loan_weight
    return max(total - number(profile.get("operating_reserve_10k_krw")), 0)


def cat_distance(a, b):
    x, y = str(a).zfill(3)[-3:], str(b).zfill(3)[-3:]
    return sum(abs(int(i) - int(j)) for i, j in zip(x, y))


def matched_preference(profile, brand):
    industry = normalize(brand["industry_middle"])
    for rank in range(1, 4):
        raw = profile.get(f"preferred_industry_{rank}", "")
        if normalize(raw) == industry:
            return str(raw)
    return ""


def affordable(profile, brand):
    cost = number(brand[COST_COLUMN]) / 10
    return bool(cost) and cost <= budget(profile)


def pair_features(profile, brand):
    row = {column: number(profile.get(column)) for column in NUMERIC_USER_COLUMNS}
    for column in ["preferred_industry_1", "preferred_industry_2", "preferred_industry_3", "loan_status"]:
        row[column] = str(profile.get(column, ""))
    row.update({
        "brand_name": brand["brand_name"],
        "brand_industry": brand["industry_middle"],
        "brand_cat": str(brand["cat"]).zfill(3),
        "brand_sales": number(brand["average_sales_per_3_3sqm"]),
        "brand_store_count": number(brand["store_count"]),
        "brand_cost_10k": number(brand[COST_COLUMN]) / 10,
        "brand_sales_class": number(brand["average_sales_per_3_3sqm_s"]),
        "brand_store_class": number(brand["store_count_s"]),
        "brand_cost_class": number(brand["총계_s"]),
    })
    preferences = [normalize(profile.get(f"preferred_industry_{rank}", "")) for rank in range(1, 4)]
    brand_industry = normalize(brand["industry_middle"])
    row["preferred_rank"] = 3 - preferences.index(brand_industry) if brand_industry in preferences else 0
    row["available_budget_10k"] = budget(profile)
    row["affordable"] = int(affordable(profile, brand))
    row["budget_cost_ratio"] = row["available_budget_10k"] / max(row["brand_cost_10k"], 1)
    row["debt_to_budget"] = row["current_debt_total_10k_krw"] / max(row["available_budget_10k"], 1)
    return row


def similarity_score(profile, first, candidate):
    preferences = {normalize(profile.get(f"preferred_industry_{rank}", "")) for rank in range(1, 4)}
    score = 55 * (normalize(candidate["industry_middle"]) == normalize(first["industry_middle"]))
    score += 35 * (normalize(candidate["industry_middle"]) in preferences)
    score += 30 - 10 * cat_distance(candidate["cat"], first["cat"])
    score += 20 if affordable(profile, candidate) else -20
    for weight, column in [(8, "average_sales_per_3_3sqm"), (4, "store_count"), (5, COST_COLUMN)]:
        score -= weight * abs(math.log1p(number(candidate[column])) - math.log1p(number(first[column])))
    return score


def ranked_additional(profile, first_name, brands, names, count):
    first = brands[first_name]
    pool = [brands[name] for name in names if name != first_name]
    return sorted(pool, key=lambda brand: (-similarity_score(profile, first, brand), brand["brand_name"]))[:count]


def select_brands(model, feature_columns, profile, brands):
    names = list(brands)
    features = pd.DataFrame([pair_features(profile, brands[name]) for name in names])
    scores = model.predict(features.reindex(columns=feature_columns))
    score_map = dict(zip(names, scores))
    preferred = [name for name in names if matched_preference(profile, brands[name])]
    if preferred:
        affordable_names = [name for name in preferred if affordable(profile, brands[name])]
        first_name = max(affordable_names or preferred, key=score_map.get)
        candidates = preferred
    else:
        first_name = max(names, key=score_map.get)
        industry = normalize(brands[first_name]["industry_middle"])
        candidates = [name for name in names if normalize(brands[name]["industry_middle"]) == industry]

    used = {first_name}
    additional = []

    # 고객이 업종을 여러 개 골랐으면, similarity_score가 "같은 업종"에 큰 가중치를 줘서 1등과
    # 같은 업종만 줄줄이 뽑히는 문제가 있었다. 그래서 1등 업종이 아닌 다른 선호 업종에서도
    # 먼저 한 자리씩 확보해 골고루 섞이게 한다(모델 점수 자체는 그대로 쓰고, 순서만 조정).
    other_preferred_industries = []
    for rank in range(1, 4):
        raw = normalize(profile.get(f"preferred_industry_{rank}", ""))
        if raw and raw != normalize(brands[first_name]["industry_middle"]) and raw not in other_preferred_industries:
            other_preferred_industries.append(raw)

    for industry in other_preferred_industries:
        if len(additional) >= 3:
            break
        pool = [
            name
            for name in candidates
            if name not in used and normalize(brands[name]["industry_middle"]) == industry
        ]
        affordable_pool = [name for name in pool if affordable(profile, brands[name])]
        pick_pool = affordable_pool or pool
        if not pick_pool:
            continue
        best = max(pick_pool, key=score_map.get)
        additional.append(brands[best])
        used.add(best)

    if len(additional) < 3:
        affordable_names = [name for name in candidates if name not in used and affordable(profile, brands[name])]
        more = ranked_additional(profile, first_name, brands, affordable_names, 3 - len(additional))
        additional += more
        used.update(brand["brand_name"] for brand in more)

    if len(additional) < 3:
        remaining = [name for name in candidates if name not in used]
        additional += ranked_additional(profile, first_name, brands, remaining, 3 - len(additional))

    return [brands[first_name], *additional]


def result_rows(profile, selected):
    rows = []
    for rank, brand in enumerate(selected, 1):
        preference = matched_preference(profile, brand)
        is_affordable = affordable(profile, brand)
        cost_note = "가용 예산 범위" if is_affordable else "예산 내 대안 부족으로 예산 초과"
        rows.append({
            "profile_id": profile.get("profile_id", ""),
            "recommendation_rank": rank,
            "brand_name": brand["brand_name"],
            "industry_middle": normalize(brand["industry_middle"]),
            "cat": str(brand["cat"]).zfill(3),
            "startup_cost_10k_krw": number(brand[COST_COLUMN]) / 10,
            "available_budget_10k_krw": budget(profile),
            "affordable": int(is_affordable),
            "matched_preference": normalize(preference),
            "reason": f"선호 업종 {normalize(preference) or normalize(brand['industry_middle'])}, {cost_note}, 모델 적합도와 유사 특성 반영",
        })
    return rows


class BrandRecommender:
    def __init__(self, artifacts_dir=ARTIFACTS):
        artifacts_dir = Path(artifacts_dir)
        self.model = CatBoostRanker()
        self.model.load_model(str(artifacts_dir / "brand_ranker.cbm"))
        metadata = joblib.load(artifacts_dir / "metadata.joblib")
        self.brands = metadata["brands"]
        self.feature_columns = metadata["feature_columns"]

    def industries(self):
        return sorted({normalize(brand["industry_middle"]) for brand in self.brands.values()})

    def recommend_one(self, profile):
        selected = select_brands(self.model, self.feature_columns, profile, self.brands)
        return result_rows(profile, selected)

    def recommend_many(self, profiles):
        return [row for profile in profiles for row in self.recommend_one(profile)]


def main():
    parser = argparse.ArgumentParser(description="브랜드 추천 모델 실행")
    commands = parser.add_subparsers(dest="command", required=True)
    recommend = commands.add_parser("recommend", help="CSV 사용자 입력으로 브랜드 추천")
    recommend.add_argument("--input", required=True)
    recommend.add_argument("--output", default="recommendations.csv")
    commands.add_parser("industries", help="선택 가능한 업종 목록 출력")
    args = parser.parse_args()

    recommender = BrandRecommender()
    if args.command == "industries":
        print(json.dumps(recommender.industries(), ensure_ascii=False, indent=2))
        return

    profiles = pd.read_csv(args.input, encoding="utf-8-sig", dtype=str).fillna("").to_dict("records")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recommender.recommend_many(profiles)).to_csv(output, index=False, encoding="utf-8-sig")
    print(f"saved={output}")


if __name__ == "__main__":
    main()
