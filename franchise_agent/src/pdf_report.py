"""④단계 확정 리포트를 1페이지 PDF로 만든다. matplotlib로 차트를 PNG로 그려서 fpdf2에 붙여넣는
방식 — plotly는 정적 이미지 추출에 kaleido가 따로 필요해서, 이미 설치돼 있는 matplotlib를 쓴다.
숫자는 전부 build_financial_report()가 이미 계산해둔 값 그대로 쓰고, 여기서는 새로 계산하지 않는다
(신뢰성 있는 숫자만 보여준다는 프로젝트 원칙 유지)."""

import tempfile
from pathlib import Path

import matplotlib
import pandas as pd
from fpdf import FPDF

matplotlib.use("Agg")  # 서버 환경에서 화면 없이 그림만 렌더링
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

# 클라우드(Streamlit Community Cloud 등) 배포 시엔 Linux라 Windows 폰트가 없다. 그래서 먼저
# 프로젝트에 같이 배포한 나눔고딕(fonts/ 폴더, SIL Open Font License — 자유 재배포 가능. 맑은
# 고딕은 MS 라이선스라 레포에 그대로 올리면 안 됨)을 찾고, 로컬 개발 환경(Windows)에서는
# 그게 없으면 맑은 고딕으로 대체한다.
_BUNDLED_FONT_DIR = Path(__file__).resolve().parent.parent.parent / "fonts"
_BUNDLED_REGULAR = _BUNDLED_FONT_DIR / "NanumGothic.ttf"
_BUNDLED_BOLD = _BUNDLED_FONT_DIR / "NanumGothicBold.ttf"
_WINDOWS_REGULAR = Path(r"C:\Windows\Fonts\malgun.ttf")
_WINDOWS_BOLD = Path(r"C:\Windows\Fonts\malgunbd.ttf")

_FONT_PATH = _BUNDLED_REGULAR if _BUNDLED_REGULAR.exists() else _WINDOWS_REGULAR
_FONT_BOLD_PATH = _BUNDLED_BOLD if _BUNDLED_BOLD.exists() else _WINDOWS_BOLD

try:
    fm.fontManager.addfont(str(_FONT_PATH))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(_FONT_PATH)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass

_SCENARIO_LABELS = {"optimistic": "낙관", "base": "기본", "pessimistic": "비관"}
_SCENARIO_COLORS = {"optimistic": "#2E7D32", "base": "#1565C0", "pessimistic": "#C62828"}


def _draw_cashflow_chart(report: dict) -> str:
    """3종 시나리오의 가구 예비자금 잔액 추이를 그려서 임시 PNG 파일 경로를 반환한다."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=150)
    has_any = False
    for key, label in _SCENARIO_LABELS.items():
        rows = report["scenarios"][key]["rows"]
        if not rows:
            continue
        has_any = True
        df = pd.DataFrame(rows)
        ax.plot(df["month"], df["reserve_balance"], label=label, color=_SCENARIO_COLORS[key], linewidth=2)
    ax.axhline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("개월")
    ax.set_ylabel("가구 예비자금 잔액(만원)")
    if has_any:
        ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name)
    plt.close(fig)
    return tmp.name


def _draw_funding_chart(report: dict) -> str:
    """자금조달 구조(자기자본/대출/부족분)를 가로 막대로 그린다."""
    funding = report["funding"]
    costs = report["costs"]
    shortfall = report.get("shortfall")

    fig, ax = plt.subplots(figsize=(7.2, 1.2), dpi=150)
    left = 0
    for label, value, color in [
        ("자기자본", funding["equity"], "#1565C0"),
        ("대출", funding["debt"], "#42A5F5"),
        ("부족분", shortfall or 0, "#C62828"),
    ]:
        if value and value > 0:
            ax.barh(["자금조달"], [value], left=left, color=color, label=label)
            left += value
    ax.set_xlabel("만원")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=3, fontsize=9, frameon=False)
    ax.set_yticks([])
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name)
    plt.close(fig)
    return tmp.name


class _ReportPDF(FPDF):
    def header(self):
        pass

    def footer(self):
        self.set_y(-12)
        self.set_font("Malgun", "", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 8, "본 리포트는 공정거래위원회 정보공개서 데이터를 바탕으로 자동 생성되었습니다. 실제 계약 전 정식 정보공개서를 반드시 확인하세요.", align="C")


def build_pdf_report(candidate: dict, report: dict, report_summary: str | None, output_path: str) -> str:
    pdf = _ReportPDF(format="A4", unit="mm")
    pdf.add_font("Malgun", "", str(_FONT_PATH))
    pdf.add_font("Malgun", "B", str(_FONT_BOLD_PATH) if _FONT_BOLD_PATH.exists() else str(_FONT_PATH))
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Malgun", "B", 18)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, f"{candidate.get('brand_name', '')} 창업 재무 리포트", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Malgun", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, f"업종: {candidate.get('industry_middle', '-')}  |  브랜드 리스크: {candidate.get('brand_risk_score', '-')}점", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    if report_summary:
        pdf.set_font("Malgun", "", 10)
        pdf.set_text_color(30, 30, 30)
        pdf.multi_cell(0, 5.5, report_summary)
        pdf.ln(2)

    # ---- 핵심 지표 ----
    costs = report["costs"]
    funding = report["funding"]
    shortfall = report.get("shortfall")
    pdf.set_font("Malgun", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, "핵심 재무 지표", new_x="LMARGIN", new_y="NEXT")

    pdf.set_font("Malgun", "", 10)
    if costs.get("has_data"):
        cost_text = f"{costs['total']:,.0f}만원"
    else:
        cost_text = "정보 없음"
    shortfall_text = f"{shortfall:,.0f}만원" if shortfall is not None else "판단 불가"
    metrics = [
        ("총 창업비용", cost_text),
        ("조달 가능액", f"{funding['total']:,.0f}만원"),
        ("부족분", shortfall_text),
    ]
    col_w = 190 / 3
    for label, value in metrics:
        x = pdf.get_x()
        y = pdf.get_y()
        pdf.set_font("Malgun", "", 9)
        pdf.set_text_color(120, 120, 120)
        pdf.cell(col_w, 5, label)
        pdf.set_xy(x, y + 5)
        pdf.set_font("Malgun", "B", 13)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(col_w, 7, value)
        pdf.set_xy(x + col_w, y)
    pdf.set_xy(10, pdf.get_y() + 14)

    funding_chart = _draw_funding_chart(report)
    pdf.image(funding_chart, x=10, w=190)
    pdf.ln(2)

    # ---- 시나리오별 지표 ----
    pdf.set_font("Malgun", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, "시나리오별 손익분기·투자금 회수", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Malgun", "", 9)
    col_w = 190 / 3
    for key, label in _SCENARIO_LABELS.items():
        s = report["scenarios"][key]
        be = f"{s['breakeven_month']}개월" if s["breakeven_month"] else "36개월 내 미도달"
        pb = f"{s['payback_month']}개월" if s["payback_month"] else "36개월 내 미회수"
        x, y = pdf.get_x(), pdf.get_y()
        pdf.set_font("Malgun", "B", 10)
        pdf.cell(col_w, 5, label)
        pdf.set_xy(x, y + 5)
        pdf.set_font("Malgun", "", 8)
        pdf.multi_cell(col_w, 4.5, f"손익분기 {be}\n투자금회수 {pb}")
        pdf.set_xy(x + col_w, y)
    pdf.set_xy(10, pdf.get_y() + 16)

    cashflow_chart = _draw_cashflow_chart(report)
    pdf.image(cashflow_chart, x=10, w=190)
    pdf.ln(2)

    # ---- 왜 이 브랜드인가 ----
    if candidate.get("fit_reasons") or candidate.get("agent_reasoning"):
        pdf.set_font("Malgun", "B", 12)
        pdf.set_text_color(20, 20, 20)
        pdf.cell(0, 8, "왜 이 브랜드인가요", new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Malgun", "", 9)
        pdf.set_text_color(40, 40, 40)
        if candidate.get("agent_reasoning"):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"에이전트 판단: {candidate['agent_reasoning']}", new_x="LMARGIN", new_y="NEXT")
        for reason in candidate.get("fit_reasons", []):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"- {reason}", new_x="LMARGIN", new_y="NEXT")

    pdf.output(output_path)
    return output_path


_COMPARE_COLORS = ["#1565C0", "#C62828", "#2E7D32", "#F9A825"]


def _draw_compare_cost_chart(entries: list[dict]) -> str:
    """후보들의 창업비용을 막대그래프로 나란히 비교한다."""
    names = [e["candidate"].get("brand_name", "") for e in entries]
    costs = [e["report"]["costs"]["total"] if e["report"]["costs"].get("has_data") else 0 for e in entries]

    fig, ax = plt.subplots(figsize=(7.2, 2.2), dpi=150)
    bars = ax.bar(names, costs, color=_COMPARE_COLORS[: len(entries)])
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{cost:,.0f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("총 창업비용(만원)")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name)
    plt.close(fig)
    return tmp.name


def _draw_compare_cashflow_chart(entries: list[dict], scenario_key: str = "base") -> str:
    """후보들의 '기본' 시나리오 현금흐름을 한 그래프에 겹쳐서 비교한다."""
    fig, ax = plt.subplots(figsize=(7.2, 3.0), dpi=150)
    has_any = False
    for i, e in enumerate(entries):
        rows = e["report"]["scenarios"][scenario_key]["rows"]
        if not rows:
            continue
        has_any = True
        df = pd.DataFrame(rows)
        ax.plot(
            df["month"], df["reserve_balance"],
            label=e["candidate"].get("brand_name", f"후보{i+1}"),
            color=_COMPARE_COLORS[i % len(_COMPARE_COLORS)], linewidth=2,
        )
    ax.axhline(0, color="#888888", linestyle="--", linewidth=1)
    ax.set_xlabel("개월")
    ax.set_ylabel("가구 예비자금 잔액(만원) — 기본 시나리오")
    if has_any:
        ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name)
    plt.close(fig)
    return tmp.name


def build_comparison_pdf(entries: list[dict], output_path: str) -> str:
    """entries: [{"candidate": ..., "report": ...}, ...] (finalize_recommendation이 정한
    최종 후보 전부, 보통 1~2개). 최종 브랜드를 하나로 고르기 전에, 후보들을 나란히 비교해볼
    수 있는 표+차트 PDF를 만든다. 개별 브랜드 PDF(build_pdf_report)와는 별개 산출물이다."""
    pdf = _ReportPDF(format="A4", unit="mm")
    pdf.add_font("Malgun", "", str(_FONT_PATH))
    pdf.add_font("Malgun", "B", str(_FONT_BOLD_PATH) if _FONT_BOLD_PATH.exists() else str(_FONT_PATH))
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Malgun", "B", 18)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 10, "최종 후보 브랜드 비교", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Malgun", "", 10)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 6, f"{len(entries)}개 후보 비교", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # ---- 비교 표 ----
    pdf.set_font("Malgun", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, "핵심 지표 비교", new_x="LMARGIN", new_y="NEXT")

    row_labels = ["브랜드", "업종", "브랜드 리스크", "총 창업비용", "조달 가능액", "부족분", "기본 손익분기", "기본 투자금회수"]
    label_w = 40
    col_w = (190 - label_w) / max(len(entries), 1)

    def _row_values(e):
        cand = e["candidate"]
        report = e["report"]
        costs = report["costs"]
        base = report["scenarios"]["base"]
        cost_text = f"{costs['total']:,.0f}만원" if costs.get("has_data") else "정보 없음"
        shortfall = report.get("shortfall")
        shortfall_text = f"{shortfall:,.0f}만원" if shortfall is not None else "판단 불가"
        be = f"{base['breakeven_month']}개월" if base["breakeven_month"] else "미도달"
        pb = f"{base['payback_month']}개월" if base["payback_month"] else "미회수"
        return [
            cand.get("brand_name", "-"),
            cand.get("industry_middle", "-") or "-",
            f"{cand.get('brand_risk_score', '-')}점",
            cost_text,
            f"{report['funding']['total']:,.0f}만원",
            shortfall_text,
            be,
            pb,
        ]

    table_values = [_row_values(e) for e in entries]

    pdf.set_font("Malgun", "", 9)
    for r, label in enumerate(row_labels):
        y = pdf.get_y()
        pdf.set_text_color(120, 120, 120)
        pdf.cell(label_w, 7, label, border="B")
        pdf.set_text_color(20, 20, 20)
        for c in range(len(entries)):
            pdf.set_xy(10 + label_w + c * col_w, y)
            pdf.cell(col_w, 7, str(table_values[c][r]), border="B")
        pdf.set_xy(10, y + 7)
    pdf.ln(4)

    cost_chart = _draw_compare_cost_chart(entries)
    pdf.image(cost_chart, x=10, w=190)
    pdf.ln(2)

    cashflow_chart = _draw_compare_cashflow_chart(entries)
    pdf.image(cashflow_chart, x=10, w=190)
    pdf.ln(2)

    # ---- 후보별 추천 근거 ----
    pdf.set_font("Malgun", "B", 12)
    pdf.set_text_color(20, 20, 20)
    pdf.cell(0, 8, "후보별 추천 근거", new_x="LMARGIN", new_y="NEXT")
    for e in entries:
        cand = e["candidate"]
        pdf.set_font("Malgun", "B", 10)
        pdf.set_text_color(20, 20, 20)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 6, cand.get("brand_name", ""), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Malgun", "", 9)
        pdf.set_text_color(40, 40, 40)
        if cand.get("agent_reasoning"):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"에이전트 판단: {cand['agent_reasoning']}", new_x="LMARGIN", new_y="NEXT")
        for reason in cand.get("fit_reasons", []):
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"- {reason}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    pdf.output(output_path)
    return output_path
