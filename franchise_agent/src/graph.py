import json
from functools import lru_cache
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from .agent_tools import BASE_TOOLS, build_brand_detail, get_chat_model
from .llm import generate_report_narrative
from .profile import derived_from_dict, profile_from_dict
from .recommender import enrich_with_db_metrics, recommend_survey_candidates
from .risk import brand_fit_reasons, compute_brand_risk_score
from .scenario import build_financial_report

FINAL_MAX_CANDIDATES = 2  # ③단계 끝에서 압축할 최종 후보 수(1~2개)
MAX_AGENT_TURNS = 24  # 에이전트 도구 호출 루프 최대 반복 횟수(무한루프 방지). 후보마다 5개 조사
                      # 도구 + 3개 질문을 다 돌아야 해서 이전(12)보다 넉넉히 잡음
MAX_AGENT_QUESTIONS = 5  # 후보가 1개뿐일 때 적용되는 최소 질문 한도(하한선)


def _to_native(value):
    """LangGraph 체크포인터는 msgpack으로 상태를 직렬화하는데, pandas에서 나온
    numpy.float64/int64는 msgpack이 못 읽는다. 그래프 상태에 넣기 전에 순수 파이썬 타입으로 바꾼다."""
    if isinstance(value, dict):
        return {k: _to_native(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_native(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        v = float(value)
        return None if np.isnan(v) else v
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and value != value:  # NaN
        return None
    return value


class AgentState(TypedDict, total=False):
    profile: dict
    derived: dict
    relax_level: int
    model_candidates: list[dict]
    final_candidates: list[dict]
    comparison_pdf_path: str
    selected_disclosure_id: str
    selected_candidate: dict
    report: dict
    report_summary: str
    pdf_path: str
    loop_choice: str
    status: str


def recommend_node(state: AgentState) -> dict:
    """① 설문 → ② 1차 필터링. 예전엔 SQL 예산·업종 조건으로 걸렀지만, 지금은 439개 브랜드로
    학습된 CatBoost 랭킹 모델(brand_recommender_runtime)이 고객 프로필을 보고 1~4개 브랜드를
    직접 추천한다. 결정론적 계산(모델 추론 + DB 조회)이라 LLM을 안 거친다 — 재현성이 중요한
    1차 후보 선정은 여기서 확정하고, 그 다음 단계(agent_driven_node)에서만 LLM이 이 1~4개를
    조사하고 질문하는 역할을 한다."""
    profile_obj = profile_from_dict(state["profile"])
    candidates = _to_native(recommend_survey_candidates(profile_obj))
    status = "candidates_ready" if candidates else "no_candidates"
    return {"model_candidates": candidates, "status": status}


def present_candidates_node(state: AgentState) -> dict:
    """② 1차 필터링 결과를 감수질문 들어가기 전에 그대로 보여준다 — 모델이 어떤 브랜드를
    왜 골랐는지(model_reason)까지 포함해서, 사용자가 확인하고 넘어가게 한다."""
    interrupt({"type": "candidates_preview", "candidates": state["model_candidates"]})
    return {}


# ask_customer는 LangGraph의 interrupt()를 직접 써야 해서 agent_tools.py가 아니라
# 여기서 정의한다(그래프 컨텍스트에 접근해야 하는 도구).


@tool
def ask_customer(question: str) -> str:
    """고객이 아니면 판단할 수 없는 질문(특정 후보의 위약금·필수구매·매출변동성 등을 감수할 수
    있는지, 후보 간 어느 쪽을 더 선호하는지 등)을 직접 묻고 답을 받는다. 0(전혀 감수 불가)~10(충분히
    감수 가능) 척도로 답하는 질문이 기본이지만, 선호도 비교처럼 척도로 안 맞는 질문은 자연어로
    답하게 해도 된다. 질문 문장에 실제 브랜드 상호명을 절대 넣지 마라 — 고객이 브랜드명만 보고
    선입견으로 답하지 않도록 "후보 1은 ~한 조건인데" 식으로 번호로만 지칭해서 조건을 설명해야 한다."""
    return interrupt({"type": "agent_question", "question_id": "pending", "text": question})


AGENT_TOOLS = BASE_TOOLS + [ask_customer]
TOOLS_BY_NAME = {t.name: t for t in AGENT_TOOLS}

_SYSTEM_PROMPT_TEMPLATE = """당신은 프랜차이즈 창업 재무 상담 에이전트입니다. 1차 추천 모델이 고객 프로필을 보고 이미
브랜드 {candidate_count}개를 골라놨습니다(아래 목록, 후보 1~{candidate_count}로 번호가 매겨져
있음). 당신의 역할은 이 후보들을 실제 정보공개서 데이터로 깊이 조사하고, 고객이 아니면 판단 못
하는 부분을 직접 물어본 뒤, 최종 1~2개로 좁히는 것입니다. 새로운 브랜드를 찾을 필요는 없습니다 —
주어진 후보 안에서만 판단하세요.

절차:
1) 후보마다 get_startup_cost_detail / get_sales_detail / get_contract_risks / get_support_info /
   get_operating_burdens를 전부 호출해서 실제 데이터를 확인하세요. 이 다섯 도구는 SQL뷰_컬럼정보
   에 정의된 항목(가맹비/보증금/기타비용, 지역별 매출, 계약위험 조항, 지원제도, 운영부담)을 각각
   담당하므로 후보마다 최소 이 다섯 도구를 다 확인해야 서로 다른 각도의 질문이 나옵니다.
2) 후보 1개당 정확히 2개의 질문을 만드세요 — 같은 항목(예: 위약금)만 반복하지 말고, 계약위험
   /운영부담/지원제도/매출 중 서로 다른 두 항목에서 뽑아 물으세요. 막연히 "감수 가능한가요"가
   아니라, 방금 조회한 원문의 구체적인 금액·조항·비율을 인용해서 물어야 합니다. 질문에는 반드시
   "후보 1은 ~", "후보 2는 ~"처럼 번호로만 지칭하고 실제 상호명은 절대 쓰지 마세요.
   수익성 관련 질문을 할 때는 "필요 월 순이익" 같은 계산값이 아니라 고객이 직접 입력한
   "목표 세후 월소득({target_income}만원)"을 기준으로 물으세요 — 계산값은 고객이 그 의미를
   바로 이해하기 어렵습니다. 조회한 매출 데이터가 자릿수가 이상하거나 극단적으로 크고 작은 값이
   섞여 있는 등 신뢰하기 어려워 보이면 그 수치로 질문을 만들지 말고, 합리적으로 보이는 값만
   근거로 쓰세요.
3) 후보가 2개 이상이면, 각 후보의 2개 질문이 끝난 뒤 "후보들 중 어느 쪽을 더 선호하시나요, 그
   이유는?" 같은 선호도 비교 질문도 최소 1번 하세요.
4) ask_customer 호출 횟수는 후보 수 × 2 + 여유분 정도로 제한되어 있으니(자세한 한도는 도구 결과의
   에러 메시지로 알 수 있음), 한도 안에서 위 절차를 지키세요.
5) 충분히 조사했으면 최종 후보를 1~2개로 좁혀 finalize_recommendation을 호출해 마무리하세요.

반드시 지킬 것: 숫자를 지어내지 말고 도구로 조회한 값만 근거로 쓰세요. finalize_recommendation의
disclosure_id는 반드시 아래 후보 목록에 있던 값이어야 합니다.

고객 정보:
- 가용 창업예산: {budget}만원
- 월 상환여력: {repayment}만원
- 목표 세후 월소득: {target_income}만원
- 참고용 필요 월 순이익(대출 상환 부담 포함): {required_profit}
- 희망 업종: {industry}

1차 추천 후보 (후보 번호 = 목록 순서):
{candidates_json}
"""


def agent_driven_node(state: AgentState) -> dict:
    """설문 이후 전 과정(후보 검색 → 조사 → 필요시 질문 → 최종 확정)을 tool-calling 에이전트가
    직접 판단하며 처리한다. ask_customer 호출 시 interrupt()로 실제 사용자 입력을 기다린다.

    주의(설계상 트레이드오프): LangGraph는 재개(resume) 시 이 노드 함수를 처음부터 다시 실행한다.
    그래서 사용자가 질문에 답할 때마다 지금까지의 OpenAI 대화가 전부 재호출된다 — 비용은
    gpt-4o-mini 기준 미미하지만(질문 5개 기준 최대 15회 호출 수준), 매 재실행마다 LLM이 완전히
    같은 순서로 도구를 부른다는 보장은 없다(temperature=0으로 최대한 결정론적으로 맞췄지만 100%는
    아니다). 재현성이 생명인 예산 필터링/리스크 채점 같은 계산은 이 노드가 아니라 도구(순수 함수)
    안에서 결정론적으로 처리하고, 에이전트는 "어떤 도구를 언제 부를지"만 판단하도록 역할을 제한했다."""
    profile = state["profile"]
    derived = state["derived"]
    model_candidates = state.get("model_candidates", [])

    industries = profile.get("desired_industries") or []
    candidate_summary = [
        {
            "candidate_number": i + 1,  # 시스템 프롬프트/질문에서 "후보 N"으로 지칭할 때 쓰는 번호
            "disclosure_id": c["disclosure_id"],
            "brand_name": c.get("brand_name"),
            "industry_middle": c.get("industry_middle"),
            "maximum_startup_total_manwon": c.get("maximum_startup_total"),
            "average_annual_sales_manwon": c.get("average_annual_sales"),
            "closure_termination_rate": c.get("closure_termination_rate"),
        }
        for i, c in enumerate(model_candidates)
    ]
    required_profit = derived.get("required_monthly_net_profit")
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        candidate_count=len(model_candidates),
        budget=round(derived.get("available_startup_budget") or 0),
        repayment=round(derived.get("monthly_repayment_capacity") or 0),
        target_income=round(profile.get("target_monthly_income") or 0),
        required_profit=f"{round(required_profit)}만원" if required_profit is not None else "판단 불가",
        industry=", ".join(industries) if industries else "특별히 없음",
        candidates_json=json.dumps(candidate_summary, ensure_ascii=False),
    )
    messages = [SystemMessage(content=system_prompt)]
    model = get_chat_model().bind_tools(AGENT_TOOLS)
    seen_candidates: dict[str, dict] = {c["disclosure_id"]: c for c in model_candidates}
    asked = 0
    # 후보 1개당 2개 질문 + 선호도 비교 질문 여유분. MAX_AGENT_QUESTIONS는 후보가 하나뿐일 때의
    # 절대 최소 하한으로만 쓴다. (질문마다 노드가 처음부터 재실행되는 구조라 개수를 늘릴수록
    # 체감 속도가 급격히 느려짐 — 3개/후보에서 2개/후보로 줄인 이유)
    max_questions = max(MAX_AGENT_QUESTIONS, len(model_candidates) * 2 + 2)

    for _ in range(MAX_AGENT_TURNS):
        response = model.invoke(messages)
        if not response.tool_calls:
            messages.append(response)
            messages.append(HumanMessage(content="도구를 호출하거나, 준비됐으면 finalize_recommendation을 호출하세요."))
            continue

        messages.append(response)
        for call in response.tool_calls:
            name = call["name"]
            args = call["args"]

            if name == "finalize_recommendation":
                final = []
                for r in args.get("ranking", []):
                    cand = seen_candidates.get(r.get("disclosure_id"))
                    if cand is None:
                        continue
                    # ②미리보기는 CSV만 썼으니, 최종 확정되는 이 소수(최대 2개) 후보만 지금
                    # DB 실측치로 보강한다 — 리스크 점수·추천사유가 CSV 결측값 대신 실제 값을 쓰게.
                    cand = {**cand, **enrich_with_db_metrics(cand["disclosure_id"])}
                    final.append(
                        {
                            **cand,
                            "brand_risk_score": compute_brand_risk_score(cand),
                            "fit_reasons": brand_fit_reasons(cand, derived),
                            "agent_reasoning": r.get("reasoning"),
                            "detail": build_brand_detail(cand["disclosure_id"]),
                        }
                    )
                if not final:
                    return {"status": "no_final_candidates", "final_candidates": []}
                return {"final_candidates": _to_native(final[:FINAL_MAX_CANDIDATES]), "status": "final_ready"}

            if name == "ask_customer":
                if asked >= max_questions:
                    result = {"error": f"질문 한도({max_questions}개)를 초과했습니다. 지금까지 조사한 내용으로 판단하세요."}
                else:
                    asked += 1
                    value = interrupt({"type": "agent_question", "question_id": call["id"], "text": args.get("question", "")})
                    result = {"answer": value}
            else:
                tool_obj = TOOLS_BY_NAME.get(name)
                result = tool_obj.invoke(args) if tool_obj else {"error": "unknown tool"}

            messages.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return {"status": "no_final_candidates", "final_candidates": []}


def comparison_node(state: AgentState) -> dict:
    """최종 후보(1~2개)를 하나로 고르기 전에, 후보 전부의 재무 리포트를 미리 계산해서 비교
    표+차트가 담긴 PDF를 만들어둔다. 브랜드를 아직 안 골랐어도 후보끼리 나란히 비교해볼 수
    있게 select_brand 화면에서 바로 다운로드할 수 있게 한다."""
    profile_obj = profile_from_dict(state["profile"])
    derived_obj = derived_from_dict(state["derived"])
    entries = []
    for cand in state["final_candidates"]:
        report = _to_native(build_financial_report(cand["disclosure_id"], cand, profile_obj, derived_obj))
        entries.append({"candidate": cand, "report": report})

    if len(entries) < 2:
        return {"comparison_pdf_path": None}

    import tempfile

    from .pdf_report import build_comparison_pdf

    ids = "_".join(e["candidate"]["disclosure_id"] for e in entries)
    output_path = str(Path(tempfile.gettempdir()) / f"franchise_comparison_{ids}.pdf")
    try:
        build_comparison_pdf(entries, output_path)
        return {"comparison_pdf_path": output_path}
    except Exception:
        return {"comparison_pdf_path": None}


def select_brand_node(state: AgentState) -> dict:
    disclosure_id = interrupt(
        {
            "type": "select_brand",
            "candidates": state["final_candidates"],
            "comparison_pdf_path": state.get("comparison_pdf_path"),
        }
    )
    chosen = next(c for c in state["final_candidates"] if c["disclosure_id"] == disclosure_id)
    return {"selected_disclosure_id": disclosure_id, "selected_candidate": chosen}


def report_node(state: AgentState) -> dict:
    profile_obj = profile_from_dict(state["profile"])
    derived_obj = derived_from_dict(state["derived"])
    report = _to_native(
        build_financial_report(state["selected_disclosure_id"], state["selected_candidate"], profile_obj, derived_obj)
    )
    try:
        summary = generate_report_narrative(state["selected_candidate"], report)
    except Exception:
        summary = None
    return {"report": report, "report_summary": summary, "status": "report_ready"}


def loop_decision_node(state: AgentState) -> dict:
    choice = interrupt(
        {
            "type": "loop_decision",
            "needs_recommendation_loop": state["report"].get("needs_recommendation_loop", False),
            "final_candidates": state["final_candidates"],
            "selected_disclosure_id": state["selected_disclosure_id"],
        }
    )
    return {"loop_choice": choice}


def reselect_node(state: AgentState) -> dict:
    chosen = next(c for c in state["final_candidates"] if c["disclosure_id"] == state["loop_choice"])
    return {"selected_disclosure_id": chosen["disclosure_id"], "selected_candidate": chosen, "loop_choice": None}


def pdf_node(state: AgentState) -> dict:
    """고객이 "이 브랜드로 확정"을 고르면 1페이지 PDF 리포트를 만든다. 세션 스레드별로 파일을
    구분해야 해서 thread_id 대신 selected_disclosure_id + 임시 파일명을 쓴다(그래프 상태에는
    thread_id가 없어서 여기서 알 방법이 없음 — 파일명이 겹쳐도 매번 새로 덮어써서 문제는 없다)."""
    import tempfile

    from .pdf_report import build_pdf_report

    output_path = str(Path(tempfile.gettempdir()) / f"franchise_report_{state['selected_disclosure_id']}.pdf")
    try:
        build_pdf_report(state["selected_candidate"], state["report"], state.get("report_summary"), output_path)
        return {"pdf_path": output_path, "status": "report_ready"}
    except Exception:
        # PDF 생성 실패해도 화면에 이미 나온 리포트는 그대로 보여줄 수 있어야 하니 조용히 넘어간다
        return {"pdf_path": None, "status": "report_ready"}


def route_after_recommend(state: AgentState) -> str:
    return "present_candidates" if state["status"] == "candidates_ready" else "end"


def route_after_agent(state: AgentState) -> str:
    return "comparison" if state["status"] == "final_ready" else "end"


def route_after_loop_decision(state: AgentState) -> str:
    choice = state.get("loop_choice")
    if choice == "accept":
        return "pdf"
    if choice == "relax":
        # 예전엔 예산·업종 조건을 완화해서 SQL로 다시 검색했지만, 지금은 모델이 프로필을 보고
        # 정해진 후보를 내놓는 구조라 "완화"라는 개념이 없다 — 모델 추천을 다시 한 번 요청하는
        # 정도로만 대응한다(같은 프로필이면 대체로 같은 결과가 나올 수 있음, 알려진 한계).
        return "recommend"
    return "reselect"  # choice는 다른 최종후보의 disclosure_id


@lru_cache(maxsize=1)
def get_graph():
    graph = StateGraph(AgentState)
    graph.add_node("recommend", recommend_node)
    graph.add_node("present_candidates", present_candidates_node)
    graph.add_node("agent", agent_driven_node)
    graph.add_node("comparison", comparison_node)
    graph.add_node("select_brand", select_brand_node)
    graph.add_node("report", report_node)
    graph.add_node("loop_decision", loop_decision_node)
    graph.add_node("reselect", reselect_node)
    graph.add_node("pdf", pdf_node)

    graph.add_edge(START, "recommend")
    graph.add_conditional_edges("recommend", route_after_recommend, {"present_candidates": "present_candidates", "end": END})
    graph.add_edge("present_candidates", "agent")
    graph.add_conditional_edges("agent", route_after_agent, {"comparison": "comparison", "end": END})
    graph.add_edge("comparison", "select_brand")
    graph.add_edge("select_brand", "report")
    graph.add_edge("report", "loop_decision")
    graph.add_conditional_edges(
        "loop_decision", route_after_loop_decision, {"pdf": "pdf", "recommend": "recommend", "reselect": "reselect"}
    )
    graph.add_edge("reselect", "report")
    graph.add_edge("pdf", END)

    return graph.compile(checkpointer=MemorySaver())


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def start_session(thread_id: str, profile: dict, derived: dict) -> dict[str, Any]:
    return get_graph().invoke({"profile": profile, "derived": derived, "relax_level": 0}, _config(thread_id))


def submit_answer(thread_id: str, value) -> dict[str, Any]:
    return get_graph().invoke(Command(resume=value), _config(thread_id))


def pending_interrupt(result: dict[str, Any]) -> dict | None:
    interrupts = result.get("__interrupt__")
    if not interrupts:
        return None
    return interrupts[0].value
