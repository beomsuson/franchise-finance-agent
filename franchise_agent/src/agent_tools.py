"""설문 이후 전 과정(후보 검색·조사·질문·최종 확정)을 이끄는 tool-calling 에이전트가 쓰는
도구 모음. LangGraph 생태계 관례대로 langchain_core의 @tool 데코레이터로 도구를 정의하고,
langchain_openai.ChatOpenAI.bind_tools()로 모델에 연결한다(raw OpenAI SDK 대신).
실제 판단(어떤 도구를 언제 부를지)은 graph.py의 agent_driven_node가 담당하고, 여기 있는
함수들은 전부 순수 조회 함수라 몇 번을 다시 불러도 항상 같은 결과를 준다 — 에이전트 루프가
LangGraph interrupt로 인해 재실행되더라도 도구 자체의 결과는 안전하다.

화면에 최종 표시하는 리스크 점수·추천사유는 에이전트의 자연어가 아니라 항상 이 프로젝트의
결정론적 함수(risk.py의 compute_brand_risk_score/brand_fit_reasons)에서 다시 계산한다 —
에이전트의 tool 호출 결과는 판단 재료로만 쓰고, 화면에 보여줄 숫자 자체의 환각(hallucination)
위험은 차단한다."""

from functools import lru_cache

from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from .config import OPENAI_API_KEY, OPENAI_MODEL
from .db import (
    get_contract_exit as db_get_contract_exit,
    get_operating_burdens as db_get_operating_burdens,
    get_sales as db_get_sales,
    get_support as db_get_support,
)
from .scenario import total_startup_cost


@lru_cache(maxsize=1)
def get_chat_model() -> ChatOpenAI:
    # gpt-5.6-luna 같은 추론(reasoning) 모델은 /v1/chat/completions에서는 reasoning_effort가
    # 켜진 채로 tool-calling을 거부한다("Function tools with reasoning_effort are not
    # supported ... use /v1/responses"). 원인은 추론 모델이 턴 사이에 reasoning item(내부
    # 사고 결과)을 유지해야 하는데 Chat Completions의 messages 배열에는 그걸 담을 자리가
    # 없기 때문 — Chat Completions가 아예 그 조합을 지원 안 하는 것뿐, "추론하면서 도구
    # 호출"은 Responses API에서는 정상 지원된다. use_responses_api=True로 Responses API를
    # 쓰면 reasoning_effort를 낮추지 않고도(예: high) tool-calling이 그대로 된다.
    # temperature는 추론 모델이 지원하지 않아 뺐다.
    return ChatOpenAI(
        model=OPENAI_MODEL,
        api_key=OPENAI_API_KEY,
        use_responses_api=True,
        reasoning={"effort": "low"},
        # 매 질문 응답마다 전체 대화가 재전송되는 구조라(graph.py 주석 참고) 짧은 시간에
        # 여러 번 답하면 분당 토큰 한도(rate limit)에 걸리기 쉽다. 자동 재시도로 순간적인
        # 429를 흡수한다.
        max_retries=3,
    )


# 화면에 브랜드 상세를 "듬뿍" 보여줄 때 쓰는 사람이 읽을 수 있는 라벨. SQL뷰_컬럼정보.pdf에 나열된
# 전체 유형을 담는다 — dynamic_questions.py의 라벨은 "감수질문을 만들 만큼 부담되는 항목"만 추려서
# 더 적지만, 여기서는 질문 생성이 아니라 정보 열람이 목적이라 중립적인 항목(계약기간 등)도 포함한다.
CONTRACT_RISK_LABELS_FULL = {
    "CONTRACT_PERIOD": "계약기간",
    "RENEWAL_ADDITIONAL_COST": "계약 갱신 시 추가비용",
    "RENEWAL_TERMINATION_PROCESS": "갱신·종료 절차",
    "RENEWAL_REFUSAL_REASON": "본사가 갱신을 거절할 수 있는 사유",
    "TERMINATION_REASON_PROCESS": "계약 해지 사유와 절차",
    "POST_TERMINATION_OBLIGATION": "계약 종료 후 의무",
    "TRANSFER_COST": "점포 양도 비용",
    "TRANSFER": "가맹점 운영권 양도 조건",
    "REPURCHASE": "물품 재매입 조건",
    "DAMAGES": "손해배상·위약금",
    "FRANCHISOR_TERMINATION_MEASURES": "본사 사정에 의한 계약종료 시 조치사항",
    "CONTRACT_MODIFICATION": "계약 내용 수정 절차",
    "SUCCESSION_DELEGATION": "운영권 상속·대리행사·위탁",
}

BURDEN_LABELS_FULL = {
    "ONGOING_FEES": "지속적으로 지급하는 로열티 등 비용",
    "REQUIRED_PURCHASE_FEES": "필수 구매 관련 비용",
    "REQUIRED_PURCHASE_ITEMS": "본사 지정 필수 구매품목",
    "PURCHASE_PRICE_RANGE": "필수품목 공급가격",
    "RELATED_PARTY_BENEFIT": "특수관계인 관련 내용",
    "ADVERTISING_PROMOTION_BURDEN": "광고·판촉비 분담",
    "SUPERVISION": "본사의 관리·감독",
    "FORCED_TRADE_COMPENSATION": "특정 거래 강제·대가",
    "OTHER_OPERATING_BURDEN": "기타 운영상 부담",
}

SUPPORT_TYPE_LABELS = {
    "STORE_RENOVATION": "점포 환경개선 지원",
    "SALES_PROMOTION": "판매촉진 지원",
    "MANAGEMENT_CONSULTING": "경영지도·컨설팅",
    "CREDIT_FINANCE": "신용·금융 지원",
    "STABLE_OPERATION": "안정적인 점포운영 지원",
    "SUPPORT_OVERVIEW": "기타 지원정보",
}


def build_brand_detail(disclosure_id: str) -> dict:
    """③ 최종후보 화면에서 '더 보기'로 펼쳐볼 정보공개서 상세. 고객이 직접 판단할 수 있게
    비용·매출·지원제도·계약위험·운영부담을 사람이 읽기 쉬운 라벨로 정리해서 준다."""
    cost = total_startup_cost(disclosure_id, None)

    sales_df = db_get_sales(disclosure_id)
    sales_rows = []
    if not sales_df.empty:
        cols = [c for c in ["region_name", "average_annual_sales", "store_count"] if c in sales_df.columns]
        sales_rows = sales_df[cols].where(sales_df[cols].notna(), None).to_dict(orient="records")

    support_df = db_get_support(disclosure_id)
    support_list = []
    if not support_df.empty and "support_type" in support_df.columns:
        available = support_df[support_df.get("support_available") == 1]
        support_list = [
            SUPPORT_TYPE_LABELS.get(t, t) for t in available["support_type"].dropna().unique().tolist()
        ]

    contract_df = db_get_contract_exit(disclosure_id)
    contract_list = []
    if not contract_df.empty and "contract_risk_type" in contract_df.columns:
        for t in contract_df["contract_risk_type"].dropna().unique().tolist():
            row = contract_df[contract_df["contract_risk_type"] == t].iloc[0]
            text = (row.get("body_text") or "")[:200]
            contract_list.append({"label": CONTRACT_RISK_LABELS_FULL.get(t, t), "text": text})

    burden_df = db_get_operating_burdens(disclosure_id)
    burden_list = []
    if not burden_df.empty and "burden_type" in burden_df.columns:
        for t in burden_df["burden_type"].dropna().unique().tolist():
            row = burden_df[burden_df["burden_type"] == t].iloc[0]
            text = (row.get("body_text") or "")[:200]
            burden_list.append({"label": BURDEN_LABELS_FULL.get(t, t), "text": text})

    return {
        "cost": cost,
        "sales_by_region": sales_rows,
        "support": support_list,
        "contract_risks": contract_list,
        "operating_burdens": burden_list,
    }


# ---------- 에이전트가 부를 수 있는 도구 (langchain_core.tools.tool) ----------
# 함수의 docstring이 곧 LLM에게 보여줄 도구 설명이 되고, 타입힌트로 파라미터 스키마가
# 자동으로 만들어진다 — 예전처럼 JSON schema를 손으로 따로 안 적어도 된다.


@tool
def get_startup_cost_detail(disclosure_id: str) -> dict:
    """브랜드의 정확한 창업비용(가맹비+보증금+기타비용) 상세를 조회한다."""
    return total_startup_cost(disclosure_id, None)


@tool
def get_sales_detail(disclosure_id: str) -> dict:
    """브랜드의 지역별 평균/최대/최소 연매출 정보를 조회한다."""
    df = db_get_sales(disclosure_id)
    if df.empty:
        return {"rows": []}
    cols = ["region_name", "average_annual_sales", "maximum_annual_sales", "minimum_annual_sales", "store_count"]
    cols = [c for c in cols if c in df.columns]
    return {"rows": df[cols].where(df[cols].notna(), None).to_dict(orient="records")}


@tool
def get_contract_risks(disclosure_id: str) -> dict:
    """브랜드의 계약·해지·갱신 관련 위험 조항 원문을 조회한다."""
    df = db_get_contract_exit(disclosure_id)
    if df.empty:
        return {"rows": []}
    cols = [c for c in ["contract_risk_type", "heading", "body_text"] if c in df.columns]
    rows = df[cols].head(8).to_dict(orient="records")
    for r in rows:
        if r.get("body_text"):
            r["body_text"] = r["body_text"][:300]
    return {"rows": rows}


@tool
def get_support_info(disclosure_id: str) -> dict:
    """브랜드가 가맹점주에게 제공하는 지원제도(금융·운영·판촉 등)를 조회한다."""
    df = db_get_support(disclosure_id)
    if df.empty:
        return {"rows": []}
    available = df[df.get("support_available") == 1] if "support_available" in df.columns else df
    cols = [c for c in ["support_type", "heading"] if c in available.columns]
    return {"rows": available[cols].to_dict(orient="records")}


@tool
def get_operating_burdens(disclosure_id: str) -> dict:
    """브랜드의 지속적 운영 부담(로열티, 필수구매 등) 원문을 조회한다."""
    df = db_get_operating_burdens(disclosure_id)
    if df.empty:
        return {"rows": []}
    cols = [c for c in ["burden_type", "heading", "body_text"] if c in df.columns]
    rows = df[cols].head(8).to_dict(orient="records")
    for r in rows:
        if r.get("body_text"):
            r["body_text"] = r["body_text"][:300]
    return {"rows": rows}


class RankingItem(BaseModel):
    disclosure_id: str
    reasoning: str = Field(
        description="이 브랜드를 이 순서로 추천하는 이유. 조회한 실제 데이터(금액·매출·조건)를 근거로 구체적으로."
    )


@tool
def finalize_recommendation(ranking: list[RankingItem]) -> str:
    """조사가 충분히 끝나서 주어진 후보 중 최종 1~2개를 확정할 때 호출한다. ranking에는 1개나
    2개만 담아라(그 이상은 무시된다) — 이 세션의 목표는 후보군을 최종 1~2개로 좁히는 것이다."""
    return "ok"  # 실제로는 graph.py가 이 호출을 가로채서 args만 읽고 그래프 상태로 넣는다


BASE_TOOLS = [
    get_startup_cost_detail,
    get_sales_detail,
    get_contract_risks,
    get_support_info,
    get_operating_burdens,
    finalize_recommendation,
]
