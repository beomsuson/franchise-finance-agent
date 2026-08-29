from functools import lru_cache

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_report_narrative(candidate: dict, report: dict) -> str:
    """④단계 리포트 숫자를 사람이 읽는 한국어 해설로 요약한다. 숫자 자체는 파이썬이 이미 계산했고,
    LLM은 그 결과를 해석·설명하는 역할만 한다."""
    scenarios = report["scenarios"]

    def fmt_scenario(label: str, key: str) -> str:
        s = scenarios[key]
        if not s["rows"]:
            return f"{label}: 매출 데이터 없음"
        be = f"{s['breakeven_month']}개월" if s["breakeven_month"] else "36개월 내 미도달"
        pb = f"{s['payback_month']}개월" if s["payback_month"] else "36개월 내 미회수"
        runway = f", {s['runway_month']}개월차 예비자금 소진 경고" if s["runway_month"] else ""
        return f"{label}: 손익분기 {be}, 투자금회수 {pb}{runway}"

    shortfall = report.get("shortfall")
    shortfall_text = f"(부족분 {shortfall:,.0f}만원)" if shortfall is not None else "(부족분 판단 불가)"

    prompt = (
        "당신은 프랜차이즈 창업 재무 상담 에이전트입니다. 아래 계산된 숫자를 바탕으로 "
        f"'{candidate.get('brand_name')}' 브랜드에 대한 금융 시나리오 해설을 한국어로 3~4문장 작성하세요. "
        "숫자를 새로 만들지 말고 주어진 값만 해석하세요. 과장하지 말고 담백하게, 위험 요소가 있으면 솔직히 짚어주세요.\n\n"
        f"총 창업비용: {report['costs']['total']:,.0f}만원\n"
        f"조달 가능액: {report['funding']['total']:,.0f}만원 {shortfall_text}\n"
        f"{fmt_scenario('낙관', 'optimistic')}\n"
        f"{fmt_scenario('기본', 'base')}\n"
        f"{fmt_scenario('비관', 'pessimistic')}\n"
    )

    response = get_client().chat.completions.create(
        model=OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content.strip()
