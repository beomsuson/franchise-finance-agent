import uuid
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.finance import REPAYMENT_METHOD_OPTIONS
from src.graph import pending_interrupt, start_session, submit_answer
from src.recommender import model_industries
from src.profile import (
    CustomerProfile,
    ExistingDebt,
    LOAN_STATUS_OPTIONS,
    NewLoanPlan,
    OPERATION_PERIOD_OPTIONS,
    PlannedExpense,
    STARTUP_TIMING_OPTIONS,
    compute_derived_metrics,
)
from src.viz_theme import CATEGORICAL, INK, SCENARIO_COLORS, SCENARIO_LABELS, STATUS

st.set_page_config(page_title="프랜차이즈 금융 에이전트", layout="wide")

if "app_stage" not in st.session_state:
    st.session_state.app_stage = "survey"
    st.session_state.thread_id = str(uuid.uuid4())

STAGE_LABELS = {
    "survey": "① 설문 응답 수집",
    "candidates": "② 1차 데이터 필터링",
    "risk": "③ 2차 감수 진단",
    "select_brand": "③ 최종 후보",
    "report": "④ 금융 시나리오 제공",
}


def _chart_layout(fig, **kwargs):
    fig.update_layout(
        plot_bgcolor=INK["surface"],
        paper_bgcolor=INK["surface"],
        font_color=INK["primary"],
        legend_title_text="",
        margin=dict(l=10, r=10, t=30, b=10),
        **kwargs,
    )
    fig.update_xaxes(gridcolor=INK["gridline"], linecolor=INK["baseline"])
    fig.update_yaxes(gridcolor=INK["gridline"], linecolor=INK["baseline"])
    return fig


def render_progress(current: str):
    st.caption(" → ".join(f"**{v}**" if k == current else v for k, v in STAGE_LABELS.items()))


def submit(value):
    with st.spinner("에이전트가 처리하는 중..."):
        st.session_state.graph_result = submit_answer(st.session_state.thread_id, value)
    st.rerun()


def restart():
    st.session_state.clear()
    st.session_state.app_stage = "survey"
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()


# ---------- ① 설문 ----------

def render_survey():
    st.header("① 창업 재무 설문")
    # st.form을 안 쓴다 — "대출 가능 여부"를 "2. 대출" 섹션 안, 문서 순서 그대로에 두면서도
    # 선택 즉시 아래 대출 상세 입력칸을 비활성화하려면(폼 안 위젯은 제출 전까지 서로 반응하지
    # 않음) 폼 자체를 없애고 일반 위젯 + 버튼으로 구성하는 게 제일 간단하다.

    st.subheader("희망 업종")
    desired_industries = st.multiselect(
        "업종(중분류, 최대 3개까지 선택 가능)", model_industries(), max_selections=3
    )

    st.subheader("1. 창업 자금")
    liquid_capital = st.number_input("즉시(1개월 이내) 가용 총자금 — 부동산·보증금 제외 (만원)", min_value=0, value=8000, step=100)
    operating_reserve = st.number_input("운영 예비자금 (만원)", min_value=0, value=2000, step=100)

    st.subheader("2. 대출")
    loan_status = st.selectbox("대출 가능 여부", LOAN_STATUS_OPTIONS, key="loan_status_select")
    no_loan = loan_status != "대출 가능"
    desired_amount = st.number_input("희망 대출액 (만원)", min_value=0, value=4000, step=100, disabled=no_loan)
    expected_rate = st.number_input("예상금리 (연 %)", min_value=0.0, value=5.5, step=0.1, disabled=no_loan)
    repayment_months = st.number_input("상환기간 (개월)", min_value=0, value=60, step=6, disabled=no_loan)
    repayment_method = st.selectbox("상환방식", REPAYMENT_METHOD_OPTIONS, disabled=no_loan)

    existing_debt_total = st.number_input("현재 부채 총액 (만원)", min_value=0, value=0, step=100)
    existing_debt_monthly = st.number_input("기존 부채 매월 납부 원리금 (만원)", min_value=0, value=0, step=10)

    st.subheader("3. 생계비")
    min_living = st.number_input("월 최소 생활비 (만원)", min_value=0, value=250, step=10)
    maintained_income = st.number_input("창업 후에도 유지되는 가구 월 세후소득 (만원)", min_value=0, value=200, step=10)

    st.caption("향후 3년 내 예정된 큰 지출 (선택)")
    expenses_df = st.data_editor(
        pd.DataFrame(columns=["목적", "예상금액(만원)", "예상시기"]),
        num_rows="dynamic",
        key="expenses_editor",
        use_container_width=True,
    )

    st.subheader("4. 기타 사항")
    planned_period = st.selectbox("새 점포 운영 계획 기간 *", OPERATION_PERIOD_OPTIONS)
    target_income = st.number_input("목표 세후 월소득 (만원) *", min_value=0, value=0, step=10)
    target_payback_years = st.number_input("목표 투자금 회수기간 (년) *", min_value=0.0, value=0.0, step=0.5)
    startup_timing = st.selectbox("실제 창업 가능 시기", STARTUP_TIMING_OPTIONS)
    existing_store_count = st.number_input("현재 운영 중인 프랜차이즈 점포 수", min_value=0, value=0, step=1)
    existing_store_cashflow = st.number_input(
        "기존 점포 월평균 순현금흐름 (만원, 적자면 음수)", value=0, step=10, disabled=existing_store_count == 0
    )

    submitted = st.button("에이전트 시작 →", use_container_width=True)

    if submitted:
        missing = []
        if planned_period == "아직 정하지 않음":
            missing.append("새 점포 운영 계획 기간")
        if target_income == 0:
            missing.append("목표 세후 월소득")
        if target_payback_years == 0:
            missing.append("목표 투자금 회수기간")
        if missing:
            st.error(
                "금융 시나리오 계산을 위해 다음 항목을 입력해주세요: " + ", ".join(missing)
            )
            st.stop()

        expenses = [
            PlannedExpense(row["목적"], row["예상금액(만원)"], row["예상시기"])
            for _, row in expenses_df.iterrows()
            if row.get("목적")
        ]
        profile = CustomerProfile(
            liquid_capital=liquid_capital,
            operating_reserve=operating_reserve,
            new_loan=NewLoanPlan(
                desired_amount=None if no_loan else (desired_amount or None),
                status=loan_status,
                expected_interest_rate=None if no_loan else expected_rate,
                repayment_period_months=None if no_loan else repayment_months,
                repayment_method=None if no_loan else repayment_method,
            ),
            existing_debt=ExistingDebt(total_amount=existing_debt_total, monthly_payment=existing_debt_monthly),
            min_monthly_living_cost=min_living,
            maintained_monthly_income=maintained_income,
            planned_major_expenses=expenses,
            existing_store_count=existing_store_count,
            existing_store_monthly_cashflow=existing_store_cashflow if existing_store_count > 0 else None,
            planned_operation_period=planned_period,
            target_monthly_income=target_income or None,
            target_payback_period_years=target_payback_years or None,
            startup_timing=startup_timing,
            desired_industries=desired_industries,
        )
        derived = compute_derived_metrics(profile)
        with st.spinner("에이전트가 후보 브랜드를 찾고 있습니다..."):
            st.session_state.graph_result = start_session(st.session_state.thread_id, asdict(profile), asdict(derived))
        st.session_state.app_stage = "agent"
        st.rerun()


# ---------- 리포트 차트 (공용) ----------

def render_report_charts(report: dict):
    funding = report["funding"]
    costs = report["costs"]
    shortfall = report["shortfall"]

    st.markdown("#### 자금 조달 구조")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["조달"], x=[funding["equity"]], name="자기자본", orientation="h", marker_color=CATEGORICAL[0]))
    fig.add_trace(go.Bar(y=["조달"], x=[funding["debt"]], name="대출", orientation="h", marker_color=CATEGORICAL[1]))
    if shortfall:
        fig.add_trace(go.Bar(y=["조달"], x=[shortfall], name="부족분", orientation="h", marker_color=STATUS["critical"]))
    fig.update_layout(barmode="stack", height=160, xaxis_title="만원")
    st.plotly_chart(_chart_layout(fig), use_container_width=True)

    m1, m2, m3 = st.columns(3)
    if costs["has_data"]:
        m1.metric("총 창업비용", f"{costs['total']:,.0f}만원")
        m3.metric("부족분", f"{shortfall:,.0f}만원", delta_color="inverse" if shortfall > 0 else "off")
    else:
        m1.metric("총 창업비용", "정보 없음")
        m3.metric("부족분", "판단 불가")
        st.caption("⚠ 이 브랜드는 창업비용 원문을 숫자로 읽을 수 없어 부족분을 계산하지 못했습니다. 정보공개서 원문을 직접 확인하세요.")
    m2.metric("조달 가능액", f"{funding['total']:,.0f}만원")

    if report["loan_schedule"]:
        st.markdown("#### 대출 상환 스케줄")
        sched_df = pd.DataFrame(report["loan_schedule"])
        fig2 = go.Figure()
        fig2.add_trace(
            go.Scatter(x=sched_df["month"], y=sched_df["principal_paid"], name="원금", stackgroup="one", line=dict(width=0.5, color=CATEGORICAL[0]))
        )
        fig2.add_trace(
            go.Scatter(x=sched_df["month"], y=sched_df["interest"], name="이자", stackgroup="one", line=dict(width=0.5, color=CATEGORICAL[1]))
        )
        fig2.update_layout(xaxis_title="개월", yaxis_title="만원")
        st.plotly_chart(_chart_layout(fig2), use_container_width=True)
        principal_total = report["loan_summary"]["total_payment"] - report["loan_summary"]["total_interest"]
        st.caption(
            f"총 상환액 {report['loan_summary']['total_payment']:,.0f}만원 "
            f"(원금 {principal_total:,.0f}만원 + 이자 {report['loan_summary']['total_interest']:,.0f}만원)"
        )
    else:
        st.info("신규 대출 계획이 없거나 상환방식이 확정되지 않아 상환 스케줄을 계산하지 않았습니다.")

    st.markdown("#### 월 현금흐름 시뮬레이션 (3종 시나리오)")
    st.caption("영업이익률 15% 가정, 개점 후 6개월 매출 램프업 반영. 실제 원가 구조는 브랜드별로 다를 수 있습니다.")
    fig3 = go.Figure()
    has_any = False
    for key in ["optimistic", "base", "pessimistic"]:
        rows = report["scenarios"][key]["rows"]
        if not rows:
            continue
        has_any = True
        rdf = pd.DataFrame(rows)
        fig3.add_trace(go.Scatter(x=rdf["month"], y=rdf["reserve_balance"], name=SCENARIO_LABELS[key], line=dict(color=SCENARIO_COLORS[key], width=2)))
    if has_any:
        fig3.add_hline(y=0, line_dash="dot", line_color=INK["baseline"])
        fig3.update_layout(xaxis_title="개월", yaxis_title="가구 예비자금 잔액 (만원)")
        st.plotly_chart(_chart_layout(fig3), use_container_width=True)
    else:
        st.warning("이 브랜드는 매출 데이터가 없어 현금흐름 시뮬레이션을 만들 수 없습니다.")

    st.markdown("#### 시나리오별 핵심 지표")
    cols = st.columns(3)
    for col, key in zip(cols, ["optimistic", "base", "pessimistic"]):
        s = report["scenarios"][key]
        with col:
            st.markdown(f"**{SCENARIO_LABELS[key]}**")
            st.metric("손익분기 도달", f"{s['breakeven_month']}개월" if s["breakeven_month"] else "36개월 내 미도달")
            st.metric("투자금 회수", f"{s['payback_month']}개월" if s["payback_month"] else "36개월 내 미회수")
            if s["runway_month"]:
                st.error(f"⚠ {s['runway_month']}개월차 예비자금 소진 경고")
            elif s["rows"]:
                st.success("✓ 예비자금 소진 위험 없음")


# ---------- 에이전트 흐름 (②③④ + 루프, 전부 LangGraph가 주도) ----------

def render_candidates_preview(intr: dict):
    st.header("② 1차 후보 브랜드")
    render_progress("candidates")
    candidates = intr["candidates"]
    st.caption(f"🤖 추천 모델이 고객 프로필을 보고 {len(candidates)}개 브랜드를 골랐습니다.")

    for i, c in enumerate(candidates, 1):
        with st.container(border=True):
            st.markdown(f"**후보 {i}. {c['brand_name']}** · {c.get('industry_middle', '-')}")
            cols = st.columns(3)
            cols[0].metric("창업비용", f"{c['maximum_startup_total']:,.0f}만원" if c.get("maximum_startup_total") is not None else "정보 없음")
            cols[1].metric("평당(3.3㎡) 매출", f"{c['average_sales_per_3_3sqm']:,.0f}만원" if c.get("average_sales_per_3_3sqm") is not None else "정보 없음")
            cols[2].metric("가맹점 수", f"{c['store_count']:,.0f}개" if c.get("store_count") is not None else "정보 없음")
            if c.get("cost_is_estimated"):
                st.warning("⚠️ 이 창업비용은 실제 값과 차이가 있을 수 있어 재확인이 필요합니다.")
            if c.get("model_reason"):
                st.caption(f"1차 선정 이유: {c['model_reason']}")

    st.markdown("#### 이제 이 후보들의 정보공개서를 근거로 감수질문을 시작합니다")
    if st.button("③ 감수질문 시작하기 →", use_container_width=True):
        submit("start")


def render_brand_detail(c: dict):
    """추천 근거 + 정보공개서 상세(비용/매출/지원제도/계약위험/운영부담)를 사람이 읽기 쉽게 보여준다."""
    if c.get("agent_reasoning"):
        st.markdown(f"**에이전트 판단:** {c['agent_reasoning']}")
    for reason in c.get("fit_reasons", []):
        st.markdown(f"- {reason}")

    detail = c.get("detail")
    if not detail:
        return

    cost = detail.get("cost") or {}
    if cost.get("has_data"):
        fees = cost.get("fees", {})
        st.markdown(
            f"**창업비용 내역:** 가맹비 {fees.get('INITIAL_FRANCHISE_FEE', 0):,.0f}만원 · "
            f"보증금 {fees.get('DEPOSIT', 0):,.0f}만원 · 기타비용(인테리어·설비 등) {cost.get('other_cost', 0):,.0f}만원 "
            f"= 총 {cost.get('total', 0):,.0f}만원"
        )

    sales_rows = [r for r in detail.get("sales_by_region", []) if r.get("region_name")]
    if sales_rows:
        st.markdown("**지역별 평균 연매출**")
        sales_df = pd.DataFrame(sales_rows).rename(
            columns={"region_name": "지역", "average_annual_sales": "평균 연매출(만원)", "store_count": "가맹점 수"}
        )
        st.dataframe(sales_df, use_container_width=True, hide_index=True)

    if detail.get("support"):
        st.markdown("**가맹본부 지원제도:** " + " · ".join(detail["support"]))

    if detail.get("contract_risks"):
        st.markdown("**계약·해지 관련 조건**")
        for item in detail["contract_risks"]:
            st.caption(f"· {item['label']}: {item['text']}")

    if detail.get("operating_burdens"):
        st.markdown("**지속적 운영 부담**")
        for item in detail["operating_burdens"]:
            st.caption(f"· {item['label']}: {item['text']}")


def render_agent_question(intr: dict):
    st.header("③ 에이전트가 직접 조사하며 질문 중")
    render_progress("risk")
    st.caption("🤖 에이전트가 후보 브랜드를 조사하다가 고객 판단이 필요해서 직접 묻는 질문입니다.")
    st.subheader(intr["text"])
    st.caption("감수 가능 여부를 묻는 질문이면 슬라이더로, 후보 선호도처럼 척도로 답하기 애매하면 아래 텍스트로 답하세요.")
    value = st.slider("0 = 전혀 감수 불가 · 10 = 충분히 감수 가능", 0, 10, 5, key=f"agentq_slider_{intr['question_id']}")
    text_value = st.text_input("또는 직접 답변 입력 (예: '후보 2가 더 좋아요, 이유는...')", key=f"agentq_text_{intr['question_id']}")
    if st.button("답변하고 계속 →", use_container_width=True):
        submit(text_value.strip() if text_value.strip() else value)


def render_select_brand(result: dict, intr: dict):
    st.header("③ 최종 후보 브랜드")
    render_progress("select_brand")
    st.caption("🤖 에이전트가 직접 후보를 검색·조사하고 필요시 질문한 뒤 골라낸 최종 후보입니다.")

    candidates = intr["candidates"]
    # average_sales_per_3_3sqm(평당매출)은 브랜드데이터.csv 값 — SQL의 average_annual_sales는
    # 결측·이상치가 잦아서(사용자 확인) 화면 대표 지표로는 CSV 쪽을 쓴다.
    cols = ["brand_name", "brand_risk_score", "maximum_startup_total", "average_sales_per_3_3sqm", "cost_is_estimated"]
    df = pd.DataFrame(candidates)
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df["cost_is_estimated"] = df["cost_is_estimated"].map({True: "⚠️ 재확인 필요", False: "-"}).fillna("-")
    st.dataframe(
        df[cols].rename(
            columns={
                "brand_name": "브랜드",
                "brand_risk_score": "브랜드 리스크",
                "maximum_startup_total": "창업비용(만원)",
                "average_sales_per_3_3sqm": "평당(3.3㎡) 매출(만원)",
                "cost_is_estimated": "창업비용 신뢰도",
            }
        ),
        use_container_width=True,
    )
    if any(c.get("cost_is_estimated") for c in candidates):
        st.caption("⚠️ 표시된 브랜드는 창업비용이 실제 값과 차이가 있을 수 있어 재확인이 필요합니다.")

    comparison_pdf_path = intr.get("comparison_pdf_path")
    if comparison_pdf_path and Path(comparison_pdf_path).exists():
        with open(comparison_pdf_path, "rb") as f:
            st.download_button(
                "📊 후보 비교 PDF 다운로드 (표+차트)",
                data=f.read(),
                file_name="브랜드_비교_리포트.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    st.markdown("#### 왜 이 순서로 추천됐나요?")
    for c in candidates:
        with st.expander(f"{c['brand_name']} — 리스크 {c.get('brand_risk_score')}점"):
            render_brand_detail(c)

    names = [c["brand_name"] for c in candidates]
    choice = st.selectbox("금융 시나리오를 볼 브랜드", names)
    if st.button("④ 금융 시나리오 보기 →", use_container_width=True):
        chosen_id = next(c["disclosure_id"] for c in candidates if c["brand_name"] == choice)
        submit(chosen_id)


def render_loop_decision(result: dict, intr: dict):
    st.header("④ 금융 시나리오 리포트")
    render_progress("report")

    candidate = result["selected_candidate"]
    report = result["report"]
    st.subheader(candidate["brand_name"])
    if result.get("report_summary"):
        st.info(result["report_summary"])

    with st.expander(f"왜 이 브랜드인가요? (리스크 {candidate.get('brand_risk_score')}점)", expanded=True):
        render_brand_detail(candidate)

    render_report_charts(report)

    if intr["needs_recommendation_loop"]:
        st.warning("모든 시나리오에서 감당이 어려운 것으로 보입니다.")

    st.markdown("#### 이 브랜드로 진행하시겠어요?")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("✅ 이 브랜드로 확정", use_container_width=True):
            submit("accept")
    with c2:
        others = [c for c in intr["final_candidates"] if c["disclosure_id"] != intr["selected_disclosure_id"]]
        if others:
            other_choice = st.selectbox("다른 최종후보", [c["brand_name"] for c in others], key="other_pick", label_visibility="collapsed")
            if st.button("이 브랜드로 다시 보기", use_container_width=True):
                chosen_id = next(c["disclosure_id"] for c in others if c["brand_name"] == other_choice)
                submit(chosen_id)
        else:
            st.caption("다른 최종후보가 없습니다.")
    with c3:
        if st.button("🔄 조건 완화하고 다시 찾기", use_container_width=True):
            submit("relax")


def render_terminal(result: dict):
    if result.get("status") == "report_ready" and result.get("report"):
        st.header("④ 금융 시나리오 — 확정")
        render_progress("report")
        candidate = result["selected_candidate"]
        st.success(f"'{candidate['brand_name']}' 브랜드로 확정하셨습니다.")
        if result.get("report_summary"):
            st.info(result["report_summary"])
        with st.expander(f"왜 이 브랜드인가요? (리스크 {candidate.get('brand_risk_score')}점)"):
            render_brand_detail(candidate)
        render_report_charts(result["report"])

        pdf_path = result.get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as f:
                st.download_button(
                    "📄 1페이지 리포트 PDF 다운로드",
                    data=f.read(),
                    file_name=f"{candidate['brand_name']}_창업리포트.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.caption("PDF 리포트를 만들지 못했습니다 — 화면의 내용은 그대로 유효합니다.")
    else:
        st.header("조건에 맞는 브랜드를 찾지 못했습니다")
        st.warning("예산·업종·감수 조건을 최대한 완화해도 맞는 브랜드가 없습니다.")

    if st.button("처음부터 다시 시작하기", use_container_width=True):
        restart()


def render_agent():
    result = st.session_state.graph_result
    intr = pending_interrupt(result)

    if intr is None:
        render_terminal(result)
        return

    dispatch = {
        "candidates_preview": lambda: render_candidates_preview(intr),
        "agent_question": lambda: render_agent_question(intr),
        "select_brand": lambda: render_select_brand(result, intr),
        "loop_decision": lambda: render_loop_decision(result, intr),
    }
    dispatch[intr["type"]]()


PAGES = {"survey": render_survey, "agent": render_agent}
PAGES[st.session_state.app_stage]()
