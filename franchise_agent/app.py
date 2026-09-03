import re
import threading
import time
import uuid
from dataclasses import asdict
from html import escape
from pathlib import Path

import openai
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

st.set_page_config(
    page_title="프랜차이즈 금융 에이전트",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------- 화면 전용 스타일 / 컴포넌트 ----------

APP_CSS = """
<style>
:root {
    --ff-brand: #3157d5;
    --ff-brand-deep: #213a9b;
    --ff-brand-soft: #edf2ff;
    --ff-ink: #172033;
    --ff-ink-soft: #526078;
    --ff-muted: #7d8799;
    --ff-line: #e1e6ef;
    --ff-surface: #ffffff;
    --ff-surface-soft: #f7f9fc;
    --ff-success: #16845b;
    --ff-warning: #b36b00;
    --ff-danger: #c64747;
    --ff-shadow: 0 12px 34px rgba(27, 43, 83, 0.08);
}

html, body, [class*="css"] {
    font-family: "Pretendard", "Noto Sans KR", "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
    color: var(--ff-ink);
}

html { scroll-behavior: smooth; }

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(circle at 8% -8%, rgba(49, 87, 213, 0.13), transparent 27rem),
        radial-gradient(circle at 96% 2%, rgba(38, 167, 128, 0.08), transparent 23rem),
        #f5f7fb;
}

[data-testid="stHeader"] {
    background: rgba(245, 247, 251, 0.82);
    backdrop-filter: blur(14px);
    -webkit-backdrop-filter: blur(14px);
}

[data-testid="stDecoration"] { display: none; }
#MainMenu, footer { visibility: hidden; }

.block-container {
    width: min(1180px, calc(100% - 2rem));
    max-width: 1180px;
    padding-top: 1.25rem;
    padding-bottom: 5rem;
}

/* 앱 상단 */
.ff-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 0.25rem 0.1rem 1.05rem;
    margin-bottom: 0.25rem;
}

.ff-brand-group {
    display: flex;
    align-items: center;
    gap: 0.78rem;
}

.ff-brand-mark {
    display: grid;
    place-items: center;
    width: 2.7rem;
    height: 2.7rem;
    border-radius: 0.9rem;
    background: linear-gradient(145deg, var(--ff-brand), var(--ff-brand-deep));
    color: #fff;
    font-size: 1.3rem;
    box-shadow: 0 8px 20px rgba(49, 87, 213, 0.24);
}

.ff-brand-name {
    color: var(--ff-ink);
    font-size: 0.98rem;
    font-weight: 850;
    letter-spacing: -0.02em;
}

.ff-brand-sub {
    color: var(--ff-muted);
    font-size: 0.72rem;
    font-weight: 650;
    margin-top: 0.08rem;
}

.ff-top-tags {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.ff-top-tag {
    display: inline-flex;
    align-items: center;
    min-height: 1.8rem;
    padding: 0.3rem 0.68rem;
    border: 1px solid #dbe2f1;
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.76);
    color: #526078;
    font-size: 0.72rem;
    font-weight: 750;
}

/* 페이지 히어로 */
.ff-hero {
    position: relative;
    overflow: hidden;
    padding: 1.75rem 1.9rem;
    margin: 0 0 1rem;
    border: 1px solid rgba(49, 87, 213, 0.16);
    border-radius: 1.5rem;
    background:
        linear-gradient(120deg, rgba(255, 255, 255, 0.98), rgba(244, 247, 255, 0.96)),
        var(--ff-surface);
    box-shadow: var(--ff-shadow);
}

.ff-hero::after {
    content: "";
    position: absolute;
    width: 13rem;
    height: 13rem;
    right: -4rem;
    top: -6rem;
    border-radius: 50%;
    background: linear-gradient(145deg, rgba(49, 87, 213, 0.17), rgba(49, 87, 213, 0.02));
}

.ff-kicker {
    position: relative;
    z-index: 1;
    color: var(--ff-brand);
    font-size: 0.72rem;
    font-weight: 850;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}

.ff-hero h1 {
    position: relative;
    z-index: 1;
    margin: 0.45rem 0 0.55rem;
    color: var(--ff-ink);
    font-size: clamp(1.75rem, 3.4vw, 2.45rem);
    line-height: 1.23;
    letter-spacing: -0.045em;
}

.ff-hero p {
    position: relative;
    z-index: 1;
    max-width: 48rem;
    margin: 0;
    color: var(--ff-ink-soft);
    font-size: 0.96rem;
    line-height: 1.72;
}

/* 진행 단계 */
.ff-progress {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 0.65rem;
    margin: 0 0 1.25rem;
}

.ff-step {
    display: flex;
    align-items: center;
    gap: 0.65rem;
    min-width: 0;
    padding: 0.75rem 0.82rem;
    border: 1px solid var(--ff-line);
    border-radius: 0.95rem;
    background: rgba(255, 255, 255, 0.78);
}

.ff-step.is-complete {
    border-color: #cbd7f7;
    background: #f1f5ff;
}

.ff-step.is-current {
    border-color: #263f9f;
    background: linear-gradient(135deg, #243e9f, #3157d5);
    box-shadow: 0 9px 22px rgba(49, 87, 213, 0.22);
}

.ff-step-number {
    display: grid;
    flex: 0 0 auto;
    place-items: center;
    width: 1.72rem;
    height: 1.72rem;
    border-radius: 50%;
    background: #eef1f7;
    color: #667085;
    font-size: 0.76rem;
    font-weight: 850;
}

.ff-step.is-complete .ff-step-number {
    background: #dfe8ff;
    color: var(--ff-brand-deep);
}

.ff-step.is-current .ff-step-number {
    background: rgba(255, 255, 255, 0.18);
    color: #fff;
}

.ff-step-copy { min-width: 0; }
.ff-step-title {
    overflow: hidden;
    color: var(--ff-ink);
    font-size: 0.79rem;
    font-weight: 820;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ff-step-sub {
    overflow: hidden;
    margin-top: 0.05rem;
    color: var(--ff-muted);
    font-size: 0.66rem;
    font-weight: 620;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ff-step.is-current .ff-step-title,
.ff-step.is-current .ff-step-sub { color: #fff; }
.ff-step.is-current .ff-step-sub { opacity: 0.78; }

/* 섹션 */
.ff-section-heading {
    display: flex;
    align-items: flex-start;
    gap: 0.72rem;
    margin: 0.05rem 0 0.95rem;
}

.ff-section-index {
    display: grid;
    flex: 0 0 auto;
    place-items: center;
    width: 2.05rem;
    height: 2.05rem;
    border-radius: 0.72rem;
    background: var(--ff-brand-soft);
    color: var(--ff-brand);
    font-size: 0.76rem;
    font-weight: 900;
}

.ff-section-index.is-icon { font-size: 1rem; }
.ff-section-copy { min-width: 0; }
.ff-section-title {
    color: var(--ff-ink);
    font-size: 1.03rem;
    font-weight: 850;
    letter-spacing: -0.025em;
    line-height: 1.35;
}
.ff-section-desc {
    margin-top: 0.18rem;
    color: var(--ff-muted);
    font-size: 0.76rem;
    line-height: 1.55;
}

.ff-subsection-label {
    margin: 0.95rem 0 0.45rem;
    padding-top: 0.9rem;
    border-top: 1px solid #edf0f5;
    color: #5f6b7e;
    font-size: 0.75rem;
    font-weight: 820;
    letter-spacing: 0.02em;
}

.ff-required-note,
.ff-inline-note {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    padding: 0.75rem 0.85rem;
    border: 1px solid #dfe5f2;
    border-radius: 0.9rem;
    background: rgba(255, 255, 255, 0.72);
    color: #5a6679;
    font-size: 0.76rem;
    line-height: 1.55;
}

.ff-required-note { margin: -0.25rem 0 1rem; }
.ff-inline-note { margin: 0.25rem 0 0.85rem; }

/* Streamlit 기본 카드 */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: var(--ff-line) !important;
    border-radius: 1.15rem !important;
    background: rgba(255, 255, 255, 0.94) !important;
    box-shadow: 0 8px 26px rgba(27, 43, 83, 0.055);
}

/* 입력 요소 */
[data-testid="stWidgetLabel"] p {
    color: #344054;
    font-size: 0.82rem;
    font-weight: 760;
    line-height: 1.45;
}

[data-baseweb="input"] > div,
[data-baseweb="select"] > div,
[data-baseweb="textarea"] > div {
    border-color: #d9e0eb !important;
    border-radius: 0.75rem !important;
    background: #fbfcff !important;
    box-shadow: none !important;
}

[data-baseweb="input"] > div:focus-within,
[data-baseweb="select"] > div:focus-within,
[data-baseweb="textarea"] > div:focus-within {
    border-color: var(--ff-brand) !important;
    box-shadow: 0 0 0 3px rgba(49, 87, 213, 0.1) !important;
}

[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    border-radius: 999px;
    background: #e8eeff;
    color: var(--ff-brand-deep);
    font-weight: 750;
}

[data-testid="stNumberInput"] button {
    border-color: #e0e5ee;
    background: #f4f6fa;
}

/* 버튼 */
.stButton > button,
[data-testid="stDownloadButton"] > button {
    min-height: 3rem;
    border-radius: 0.78rem;
    font-weight: 820;
    letter-spacing: -0.015em;
    transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
}

.stButton > button:hover,
[data-testid="stDownloadButton"] > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(27, 43, 83, 0.11);
}

[data-testid="stBaseButton-primary"] {
    border: 0 !important;
    background: linear-gradient(135deg, var(--ff-brand-deep), var(--ff-brand)) !important;
    color: #fff !important;
    box-shadow: 0 8px 20px rgba(49, 87, 213, 0.22);
}

[data-testid="stBaseButton-secondary"],
[data-testid="stDownloadButton"] > button {
    border-color: #cfd7e6 !important;
    background: #fff !important;
    color: #2e3a50 !important;
}

/* 지표 / 표 / 알림 */
[data-testid="stMetric"] {
    min-height: 5.3rem;
    padding: 0.82rem 0.9rem;
    border: 1px solid #e2e7f0;
    border-radius: 0.9rem;
    background: #f8faff;
}

[data-testid="stMetricLabel"] p {
    color: #667085;
    font-size: 0.73rem;
    font-weight: 760;
}

[data-testid="stMetricValue"] {
    color: var(--ff-ink);
    font-size: clamp(1.08rem, 2vw, 1.38rem);
    font-weight: 860;
    letter-spacing: -0.035em;
}

[data-testid="stDataFrame"] {
    overflow: hidden;
    border: 1px solid #e0e5ee;
    border-radius: 0.9rem;
}

[data-testid="stAlert"] {
    border-radius: 0.95rem;
    border-width: 1px;
}

[data-testid="stExpander"] {
    overflow: hidden;
    border: 1px solid #dfe5ee;
    border-radius: 0.95rem;
    background: rgba(255, 255, 255, 0.92);
}

[data-testid="stExpander"] summary {
    font-weight: 780;
}

[data-testid="stProgress"] > div > div {
    border-radius: 999px;
}

/* 요약 배너 */
.ff-summary-banner {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 1rem 1.05rem;
    margin-bottom: 1rem;
    border: 1px solid #d9e2f6;
    border-radius: 1rem;
    background: linear-gradient(135deg, #f2f6ff, #ffffff);
}

.ff-summary-banner strong {
    display: block;
    color: var(--ff-ink);
    font-size: 0.94rem;
    font-weight: 850;
}
.ff-summary-banner span {
    display: block;
    margin-top: 0.18rem;
    color: var(--ff-ink-soft);
    font-size: 0.76rem;
    line-height: 1.48;
}
.ff-summary-count {
    flex: 0 0 auto;
    min-width: 4.4rem;
    padding: 0.58rem 0.8rem;
    border-radius: 0.85rem;
    background: var(--ff-brand);
    color: #fff;
    text-align: center;
    font-size: 0.74rem;
    font-weight: 760;
}
.ff-summary-count b {
    display: block;
    margin-bottom: 0.05rem;
    font-size: 1.32rem;
    line-height: 1;
}

/* 후보 카드 */
.ff-candidate-card {
    height: 100%;
    min-height: 15.5rem;
    padding: 1.15rem;
    margin-bottom: 0.25rem;
    border: 1px solid #dfe5ef;
    border-radius: 1.15rem;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 8px 24px rgba(27, 43, 83, 0.055);
    transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
}
.ff-candidate-card:hover {
    transform: translateY(-2px);
    border-color: #bdcaeb;
    box-shadow: 0 13px 28px rgba(27, 43, 83, 0.09);
}
.ff-candidate-top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.65rem;
}
.ff-chip-group {
    display: flex;
    align-items: center;
    gap: 0.38rem;
    min-width: 0;
    flex-wrap: wrap;
}
.ff-rank-chip,
.ff-industry-chip,
.ff-warning-chip {
    display: inline-flex;
    align-items: center;
    min-height: 1.72rem;
    padding: 0.3rem 0.58rem;
    border-radius: 999px;
    font-size: 0.69rem;
    font-weight: 810;
    white-space: nowrap;
}
.ff-rank-chip { background: var(--ff-brand); color: #fff; }
.ff-industry-chip { background: #f0f3f8; color: #5a6678; }
.ff-warning-chip { background: #fff2d8; color: #985c00; }
.ff-candidate-name {
    margin: 0.9rem 0 0.85rem;
    color: var(--ff-ink);
    font-size: 1.32rem;
    font-weight: 900;
    letter-spacing: -0.045em;
    line-height: 1.25;
}
.ff-candidate-metrics {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.48rem;
}
.ff-mini-metric {
    min-width: 0;
    padding: 0.72rem 0.68rem;
    border: 1px solid #e5e9f1;
    border-radius: 0.78rem;
    background: #f8faff;
}
.ff-mini-label {
    overflow: hidden;
    color: #798397;
    font-size: 0.64rem;
    font-weight: 740;
    line-height: 1.3;
    text-overflow: ellipsis;
    white-space: nowrap;
}
.ff-mini-value {
    overflow-wrap: anywhere;
    margin-top: 0.32rem;
    color: #243047;
    font-size: 0.88rem;
    font-weight: 870;
    letter-spacing: -0.03em;
    line-height: 1.25;
}
.ff-candidate-reason {
    margin-top: 0.82rem;
    padding: 0.76rem 0.8rem;
    border-left: 3px solid #a9baf0;
    border-radius: 0.3rem 0.72rem 0.72rem 0.3rem;
    background: #f7f9fe;
    color: #536078;
    font-size: 0.73rem;
    line-height: 1.6;
}
.ff-candidate-reason b {
    color: #344054;
    font-size: 0.68rem;
}
.ff-risk-row {
    display: flex;
    align-items: center;
    gap: 0.55rem;
    margin-top: 0.8rem;
}
.ff-risk-label {
    color: #687386;
    font-size: 0.66rem;
    font-weight: 760;
    white-space: nowrap;
}
.ff-risk-track {
    flex: 1;
    height: 0.42rem;
    overflow: hidden;
    border-radius: 999px;
    background: #e9edf3;
}
.ff-risk-fill {
    height: 100%;
    border-radius: inherit;
    background: linear-gradient(90deg, #28a779, #e3a127, #d54c4c);
}
.ff-risk-score {
    color: #344054;
    font-size: 0.69rem;
    font-weight: 850;
    white-space: nowrap;
}

/* 상세 근거 */
.ff-reasoning-box {
    padding: 0.9rem 0.95rem;
    margin-bottom: 0.85rem;
    border: 1px solid #dbe3f5;
    border-radius: 0.9rem;
    background: #f4f7ff;
    color: #40506b;
    font-size: 0.78rem;
    line-height: 1.65;
}
.ff-reasoning-box b { color: #2e3d59; }
.ff-detail-title {
    margin: 0.82rem 0 0.48rem;
    color: #344054;
    font-size: 0.79rem;
    font-weight: 850;
}
.ff-check-list {
    display: grid;
    gap: 0.42rem;
    margin-bottom: 0.65rem;
}
.ff-check-item {
    display: flex;
    align-items: flex-start;
    gap: 0.48rem;
    color: #526078;
    font-size: 0.75rem;
    line-height: 1.52;
}
.ff-check-icon {
    display: grid;
    flex: 0 0 auto;
    place-items: center;
    width: 1.15rem;
    height: 1.15rem;
    margin-top: 0.08rem;
    border-radius: 50%;
    background: #e7f6ef;
    color: #16845b;
    font-size: 0.66rem;
    font-weight: 900;
}
.ff-pill-list {
    display: flex;
    flex-wrap: wrap;
    gap: 0.42rem;
    margin-bottom: 0.65rem;
}
.ff-pill {
    padding: 0.38rem 0.62rem;
    border: 1px solid #dce5f6;
    border-radius: 999px;
    background: #f4f7fd;
    color: #4b5a74;
    font-size: 0.7rem;
    font-weight: 740;
}
.ff-detail-list {
    display: grid;
    gap: 0.48rem;
}
.ff-detail-item {
    padding: 0.67rem 0.72rem;
    border: 1px solid #e5e9f0;
    border-radius: 0.72rem;
    background: #fafbfc;
    color: #566277;
    font-size: 0.72rem;
    line-height: 1.55;
}
.ff-detail-item b {
    display: block;
    margin-bottom: 0.12rem;
    color: #344054;
    font-size: 0.72rem;
}

/* 질문 */
.ff-question-card {
    padding: 1.25rem 1.3rem;
    margin-bottom: 1rem;
    border: 1px solid #d7e0f4;
    border-left: 4px solid var(--ff-brand);
    border-radius: 1rem;
    background: linear-gradient(135deg, #f4f7ff, #ffffff);
    box-shadow: 0 8px 22px rgba(27, 43, 83, 0.055);
}
.ff-question-label {
    color: var(--ff-brand);
    font-size: 0.68rem;
    font-weight: 880;
    letter-spacing: 0.09em;
}
.ff-question-card p {
    margin: 0.72rem 0 0;
    color: #2f3c55;
    font-size: 0.94rem;
    font-weight: 700;
    line-height: 1.75;
}
.ff-scale-legend {
    display: flex;
    justify-content: space-between;
    margin: -0.25rem 0 0.4rem;
    color: #7b8597;
    font-size: 0.68rem;
    font-weight: 680;
}

/* 리포트 / 상태 */
.ff-report-brand {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
    padding: 1rem 1.05rem;
    margin-bottom: 0.9rem;
    border: 1px solid #dbe2f1;
    border-radius: 1rem;
    background: #fff;
}
.ff-report-brand-name {
    color: var(--ff-ink);
    font-size: 1.2rem;
    font-weight: 900;
    letter-spacing: -0.04em;
}
.ff-report-brand-sub {
    margin-top: 0.18rem;
    color: #758095;
    font-size: 0.72rem;
}
.ff-score-badge {
    flex: 0 0 auto;
    padding: 0.48rem 0.68rem;
    border-radius: 0.75rem;
    background: #f1f4fa;
    color: #455168;
    font-size: 0.72rem;
    font-weight: 800;
}
.ff-status-strip {
    margin-top: 0.55rem;
    padding: 0.62rem 0.68rem;
    border-radius: 0.72rem;
    font-size: 0.7rem;
    font-weight: 790;
    line-height: 1.45;
}
.ff-status-strip.is-good { background: #eaf7f1; color: #147550; }
.ff-status-strip.is-danger { background: #fff0f0; color: #b33c3c; }
.ff-status-strip.is-neutral { background: #f1f3f7; color: #697386; }
.ff-scenario-name {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.6rem;
    color: #344054;
    font-size: 0.88rem;
    font-weight: 870;
}
.ff-scenario-dot {
    width: 0.58rem;
    height: 0.58rem;
    border-radius: 50%;
}

.ff-action-title {
    margin-bottom: 0.25rem;
    color: #344054;
    font-size: 0.9rem;
    font-weight: 850;
}
.ff-action-desc {
    min-height: 2.4rem;
    margin-bottom: 0.7rem;
    color: #798397;
    font-size: 0.71rem;
    line-height: 1.48;
}

/* 세부 여백 정리 */
[data-testid="stMarkdownContainer"] h4 {
    margin-top: 1.35rem;
    color: #2d3850;
    font-size: 1.02rem;
    letter-spacing: -0.025em;
}

hr {
    border-color: #e7eaf0 !important;
}

@media (max-width: 820px) {
    .block-container {
        width: min(100% - 1rem, 1180px);
        padding-top: 0.65rem;
    }
    .ff-top-tags { display: none; }
    .ff-hero { padding: 1.3rem 1.2rem; border-radius: 1.15rem; }
    .ff-progress { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .ff-candidate-metrics { grid-template-columns: 1fr; }
    .ff-candidate-card { min-height: 0; }
    .ff-summary-banner { align-items: flex-start; }
}

@media (max-width: 520px) {
    .ff-brand-sub { display: none; }
    .ff-progress { gap: 0.45rem; }
    .ff-step { padding: 0.62rem 0.65rem; }
    .ff-step-sub { display: none; }
    .ff-summary-banner { display: block; }
    .ff-summary-count { width: fit-content; margin-top: 0.7rem; }
    .ff-hero h1 { font-size: 1.62rem; }
}
</style>
"""

PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
}

STEP_DEFINITIONS = [
    (1, "조건 입력", "재무 설문"),
    (2, "후보 탐색", "데이터 필터링"),
    (3, "감수·선정", "조건 확인"),
    (4, "금융 검토", "시나리오"),
]

STAGE_STEP = {
    "survey": 1,
    "candidates": 2,
    "risk": 3,
    "select_brand": 3,
    "report": 4,
}


def _html_text(value, fallback: str = "-") -> str:
    if value is None:
        return escape(fallback)
    try:
        if bool(pd.isna(value)):
            return escape(fallback)
    except (TypeError, ValueError):
        pass
    return escape(str(value)).replace("\n", "<br>")


def _format_number(value, suffix: str = "", fallback: str = "정보 없음") -> str:
    if value is None:
        return fallback
    try:
        if bool(pd.isna(value)):
            return fallback
    except (TypeError, ValueError):
        pass
    try:
        return f"{float(value):,.0f}{suffix}"
    except (TypeError, ValueError):
        return f"{value}{suffix}"


def render_topbar():
    render_html(
        """
        <div class="ff-topbar">
            <div class="ff-brand-group">
                <div class="ff-brand-mark">🏪</div>
                <div>
                    <div class="ff-brand-name">프랜차이즈 금융 에이전트</div>
                    <div class="ff-brand-sub">창업 조건부터 현금흐름까지 한 화면에서 검토</div>
                </div>
            </div>
            <div class="ff-top-tags">
                <span class="ff-top-tag">정보공개서 기반</span>
                <span class="ff-top-tag">4단계 검토</span>
            </div>
        </div>
        """,
        
    )


def render_page_header(kicker: str, title: str, description: str):
    render_html(
        f"""
        <section class="ff-hero">
            <div class="ff-kicker">{escape(kicker)}</div>
            <h1>{escape(title)}</h1>
            <p>{escape(description)}</p>
        </section>
        """,
        
    )


def render_html(html: str):
    """unsafe_allow_html=True로 HTML을 그릴 때, 여러 줄 f-string이 들여쓰기가 깊어지면
    Streamlit의 마크다운 파서가 4칸 이상 들여쓴 줄을 코드블록으로 오인식해서 태그가 그대로
    화면에 텍스트로 노출되는 문제가 있었다. 렌더링 직전에 각 줄 앞쪽 공백을 지워서 원천 차단한다."""
    st.markdown("\n".join(line.strip() for line in html.splitlines()), unsafe_allow_html=True)


def render_progress(current: str):
    current_step = STAGE_STEP[current]
    cards = []
    for number, title, subtitle in STEP_DEFINITIONS:
        if number < current_step:
            state = "is-complete"
            number_text = "✓"
        elif number == current_step:
            state = "is-current"
            number_text = str(number)
        else:
            state = ""
            number_text = str(number)
        cards.append(
            f"""
            <div class="ff-step {state}">
                <div class="ff-step-number">{number_text}</div>
                <div class="ff-step-copy">
                    <div class="ff-step-title">{escape(title)}</div>
                    <div class="ff-step-sub">{escape(subtitle)}</div>
                </div>
            </div>
            """
        )
    render_html(f'<div class="ff-progress">{"".join(cards)}</div>')


def render_section_heading(index: str, title: str, description: str = "", icon: bool = False):
    index_class = "ff-section-index is-icon" if icon else "ff-section-index"
    desc_html = f'<div class="ff-section-desc">{escape(description)}</div>' if description else ""
    render_html(
        f"""
        <div class="ff-section-heading">
            <div class="{index_class}">{escape(index)}</div>
            <div class="ff-section-copy">
                <div class="ff-section-title">{escape(title)}</div>
                {desc_html}
            </div>
        </div>
        """,
        
    )


def render_summary_banner(title: str, description: str, count: int | None = None, count_label: str = "후보"):
    count_html = ""
    if count is not None:
        count_html = (
            f'<div class="ff-summary-count"><b>{count}</b>{escape(count_label)}</div>'
        )
    render_html(
        f"""
        <div class="ff-summary-banner">
            <div>
                <strong>{escape(title)}</strong>
                <span>{escape(description)}</span>
            </div>
            {count_html}
        </div>
        """,
        
    )


def _candidate_card_html(candidate: dict, rank: int, *, show_risk: bool = False) -> str:
    brand_name = _html_text(candidate.get("brand_name"), "브랜드명 없음")
    industry = _html_text(candidate.get("industry_middle"), "업종 정보 없음")
    startup_cost = escape(_format_number(candidate.get("maximum_startup_total"), "만원"))
    sales = escape(_format_number(candidate.get("average_sales_per_3_3sqm"), "만원"))
    store_count = escape(_format_number(candidate.get("store_count"), "개"))
    estimated = bool(candidate.get("cost_is_estimated"))
    warning_chip = '<span class="ff-warning-chip">비용 재확인</span>' if estimated else ""

    if show_risk:
        risk_score = candidate.get("brand_risk_score")
        try:
            risk_value = max(0.0, min(float(risk_score), 100.0))
            risk_text = f"{risk_value:,.1f}점"
            risk_width = f"{risk_value:.1f}%"
        except (TypeError, ValueError):
            risk_text = "정보 없음"
            risk_width = "0%"
        metric_three_label = "브랜드 리스크"
        metric_three_value = escape(risk_text)
        risk_html = f"""
            <div class="ff-risk-row">
                <span class="ff-risk-label">낮을수록 안정적</span>
                <div class="ff-risk-track"><div class="ff-risk-fill" style="width:{risk_width}"></div></div>
                <span class="ff-risk-score">{escape(risk_text)}</span>
            </div>
        """
    else:
        metric_three_label = "가맹점 수"
        metric_three_value = store_count
        risk_html = ""

    reason = candidate.get("model_reason")
    reason_html = ""
    if reason:
        reason_html = (
            '<div class="ff-candidate-reason"><b>선정 이유</b><br>'
            f'{_html_text(reason)}</div>'
        )

    return f"""
        <article class="ff-candidate-card">
            <div class="ff-candidate-top">
                <div class="ff-chip-group">
                    <span class="ff-rank-chip">후보 {rank}</span>
                    <span class="ff-industry-chip">{industry}</span>
                </div>
                {warning_chip}
            </div>
            <div class="ff-candidate-name">{brand_name}</div>
            <div class="ff-candidate-metrics">
                <div class="ff-mini-metric">
                    <div class="ff-mini-label">창업비용</div>
                    <div class="ff-mini-value">{startup_cost}</div>
                </div>
                <div class="ff-mini-metric">
                    <div class="ff-mini-label">평당 연매출</div>
                    <div class="ff-mini-value">{sales}</div>
                </div>
                <div class="ff-mini-metric">
                    <div class="ff-mini-label">{escape(metric_three_label)}</div>
                    <div class="ff-mini-value">{metric_three_value}</div>
                </div>
            </div>
            {risk_html}
            {reason_html}
        </article>
    """


def render_candidate_grid(candidates: list[dict], *, show_risk: bool = False):
    if len(candidates) == 1:
        left, center, right = st.columns([1, 1.8, 1])
        with center:
            render_html(
                _candidate_card_html(candidates[0], 1, show_risk=show_risk),
                
            )
        return

    for start in range(0, len(candidates), 2):
        row = candidates[start : start + 2]
        columns = st.columns(2, gap="large")
        for offset, candidate in enumerate(row):
            with columns[offset]:
                render_html(
                    _candidate_card_html(candidate, start + offset + 1, show_risk=show_risk),
                    
                )


def _chart_layout(fig, **kwargs):
    layout = {
        "plot_bgcolor": "#ffffff",
        "paper_bgcolor": "#ffffff",
        "font": {
            "family": "Pretendard, Noto Sans KR, Apple SD Gothic Neo, Malgun Gothic, sans-serif",
            "color": INK["primary"],
            "size": 13,
        },
        "hoverlabel": {"bgcolor": "#172033", "font_color": "#ffffff"},
        "hovermode": "x unified",
        "legend_title_text": "",
        "legend": {
            "font_color": INK["secondary"],
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "left",
            "x": 0,
        },
        "margin": {"l": 12, "r": 12, "t": 42, "b": 12},
    }
    layout.update(kwargs)
    fig.update_layout(**layout)
    fig.update_xaxes(
        gridcolor=INK["gridline"],
        linecolor=INK["baseline"],
        tickformat=",",
        tickfont_color=INK["secondary"],
        title_font_color=INK["secondary"],
        automargin=True,
        zeroline=False,
    )
    fig.update_yaxes(
        gridcolor=INK["gridline"],
        linecolor=INK["baseline"],
        tickformat=",",
        tickfont_color=INK["secondary"],
        title_font_color=INK["secondary"],
        automargin=True,
        zeroline=False,
    )
    return fig


def render_plotly(fig):
    st.plotly_chart(
        _chart_layout(fig),
        use_container_width=True,
        config=PLOTLY_CONFIG,
        theme=None,
    )


render_html(APP_CSS)

if "app_stage" not in st.session_state:
    st.session_state.app_stage = "survey"
    st.session_state.thread_id = str(uuid.uuid4())


def format_agent_question(text: str) -> str:
    """후보별 설명이 한 문단에 붙지 않도록 후보 번호 등장 시점마다 나눈다."""
    if not text:
        return text
    parts = re.split(r"(?=후보\s?\d)", text)
    return "\n\n".join(p.strip() for p in parts if p.strip())


def md_safe(text: str) -> str:
    """자유 텍스트의 마크다운 예약문자가 의도치 않게 서식으로 해석되지 않게 한다."""
    if not text:
        return text
    for ch in ("~", "*", "_", "`"):
        text = text.replace(ch, "\\" + ch)
    return text


AGENT_STEP_ESTIMATE_SECONDS = 18


def submit(value):
    outcome: dict = {}
    # st.session_state는 메인 스크립트 스레드에서만 접근 가능하므로 미리 값을 꺼낸다.
    thread_id = st.session_state.thread_id

    def _worker():
        try:
            outcome["result"] = submit_answer(thread_id, value)
        except openai.RateLimitError:
            outcome["rate_limited"] = True
        except Exception as e:  # noqa: BLE001 - 워커 예외를 메인 스레드에서 다시 처리
            outcome["error"] = e

    thread = threading.Thread(target=_worker)
    thread.start()

    bar = st.progress(0, text="에이전트가 답변을 검토하는 중...")
    elapsed = 0.0
    while thread.is_alive():
        time.sleep(0.3)
        elapsed += 0.3
        pct = min(int(elapsed / AGENT_STEP_ESTIMATE_SECONDS * 90), 90)
        bar.progress(pct, text=f"에이전트가 답변을 검토하는 중... ({elapsed:.0f}초 경과)")
    thread.join()
    bar.progress(100, text="완료")

    if outcome.get("rate_limited"):
        st.error(
            "지금 요청이 몰려서 잠시 처리할 수 없어요 (OpenAI 분당 사용량 한도 초과). "
            "10~20초 정도 기다렸다가 방금 답변을 다시 제출해주세요."
        )
        st.stop()
    if "error" in outcome:
        raise outcome["error"]

    st.session_state.graph_result = outcome["result"]
    st.rerun()


def restart():
    st.session_state.clear()
    st.session_state.app_stage = "survey"
    st.session_state.thread_id = str(uuid.uuid4())
    st.rerun()


# ---------- ① 설문 ----------


def render_survey():
    render_page_header(
        "STEP 1 · FINANCIAL PROFILE",
        "창업 조건을 알려주세요",
        "입력 항목을 영역별로 나눠 한눈에 확인할 수 있도록 구성했습니다. 필수 목표값을 포함해 현재 자금 상황을 입력해주세요.",
    )
    render_progress("survey")
    render_html(
        """
        <div class="ff-required-note">
            <span>●</span>
            <span><b>* 표시 항목은 필수입니다.</b> 금액은 모두 만원 단위이며, 입력값은 기존 분석 흐름 그대로 후보 검토와 금융 시나리오에 사용됩니다.</span>
        </div>
        """,
        
    )

    with st.container(border=True):
        render_section_heading("선호", "희망 업종", "관심 있는 중분류 업종을 최대 3개까지 선택할 수 있습니다.")
        desired_industries = st.multiselect(
            "업종(중분류, 최대 3개까지 선택 가능)",
            model_industries(),
            max_selections=3,
            placeholder="관심 업종을 선택하세요",
        )

    left, right = st.columns(2, gap="large")

    with left:
        with st.container(border=True):
            render_section_heading("01", "창업 자금", "즉시 투입할 수 있는 자금과 별도로 남겨둘 운영 예비자금을 입력하세요.")
            c1, c2 = st.columns(2)
            liquid_capital = c1.number_input(
                "즉시(1개월 이내) 가용 총자금 — 부동산·보증금 제외 (만원)",
                min_value=0,
                value=8000,
                step=100,
            )
            operating_reserve = c2.number_input(
                "운영 예비자금 (만원)",
                min_value=0,
                value=2000,
                step=100,
            )

        with st.container(border=True):
            render_section_heading("02", "대출 및 기존 부채", "신규 대출 계획과 현재 부담 중인 원리금을 구분해 입력하세요.")
            loan_status = st.selectbox(
                "대출 가능 여부",
                LOAN_STATUS_OPTIONS,
                key="loan_status_select",
            )
            no_loan = loan_status != "대출 가능"

            c1, c2 = st.columns(2)
            desired_amount = c1.number_input(
                "희망 대출액 (만원)",
                min_value=0,
                value=4000,
                step=100,
                disabled=no_loan,
            )
            expected_rate = c2.number_input(
                "예상금리 (연 %)",
                min_value=0.0,
                value=5.5,
                step=0.1,
                disabled=no_loan,
            )
            c1, c2 = st.columns(2)
            repayment_months = c1.number_input(
                "상환기간 (개월)",
                min_value=0,
                value=60,
                step=6,
                disabled=no_loan,
            )
            repayment_method = c2.selectbox(
                "상환방식",
                REPAYMENT_METHOD_OPTIONS,
                disabled=no_loan,
            )

            render_html('<div class="ff-subsection-label">현재 보유 부채</div>')
            c1, c2 = st.columns(2)
            existing_debt_total = c1.number_input(
                "현재 부채 총액 (만원)",
                min_value=0,
                value=0,
                step=100,
            )
            existing_debt_monthly = c2.number_input(
                "기존 부채 매월 납부 원리금 (만원)",
                min_value=0,
                value=0,
                step=10,
            )

    with right:
        with st.container(border=True):
            render_section_heading("03", "생활비 및 예정 지출", "창업 후 가구에 남는 소득과 반드시 지켜야 할 생활비를 입력하세요.")
            c1, c2 = st.columns(2)
            min_living = c1.number_input(
                "월 최소 생활비 (만원)",
                min_value=0,
                value=250,
                step=10,
            )
            maintained_income = c2.number_input(
                "창업 후에도 유지되는 가구 월 세후소득 (만원)",
                min_value=0,
                value=200,
                step=10,
            )

            render_html('<div class="ff-subsection-label">향후 3년 내 예정된 큰 지출 · 선택</div>')
            expenses_df = st.data_editor(
                pd.DataFrame(columns=["목적", "예상금액(만원)", "예상시기"]),
                num_rows="dynamic",
                key="expenses_editor",
                use_container_width=True,
                hide_index=True,
                height=176,
            )

        with st.container(border=True):
            render_section_heading("04", "운영 계획 및 목표", "점포 운영 기간과 원하는 소득·회수기간을 기준으로 시나리오를 검토합니다.")
            c1, c2 = st.columns(2)
            planned_period = c1.selectbox(
                "새 점포 운영 계획 기간 *",
                OPERATION_PERIOD_OPTIONS,
            )
            target_income = c2.number_input(
                "목표 세후 월소득 (만원) *",
                min_value=0,
                value=0,
                step=10,
            )
            c1, c2 = st.columns(2)
            target_payback_years = c1.number_input(
                "목표 투자금 회수기간 (년) *",
                min_value=0.0,
                value=0.0,
                step=0.5,
            )
            startup_timing = c2.selectbox(
                "실제 창업 가능 시기",
                STARTUP_TIMING_OPTIONS,
            )
            c1, c2 = st.columns(2)
            existing_store_count = c1.number_input(
                "현재 운영 중인 프랜차이즈 점포 수",
                min_value=0,
                value=0,
                step=1,
            )
            existing_store_cashflow = c2.number_input(
                "기존 점포 월평균 순현금흐름 (만원, 적자면 음수)",
                value=0,
                step=10,
                disabled=existing_store_count == 0,
            )

    render_html(
        """
        <div class="ff-inline-note">
            <span>✓</span>
            <span>입력 내용을 확인한 뒤 시작하세요. 이후 단계의 계산·추천 로직은 기존 코드와 동일하게 실행됩니다.</span>
        </div>
        """,
        
    )
    submitted = st.button("입력 완료 · 에이전트 시작 →", use_container_width=True, type="primary")

    if submitted:
        missing = []
        if planned_period == "아직 정하지 않음":
            missing.append("새 점포 운영 계획 기간")
        if target_income == 0:
            missing.append("목표 세후 월소득")
        if target_payback_years == 0:
            missing.append("목표 투자금 회수기간")
        if missing:
            st.error("금융 시나리오 계산을 위해 다음 항목을 입력해주세요: " + ", ".join(missing))
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
            existing_debt=ExistingDebt(
                total_amount=existing_debt_total,
                monthly_payment=existing_debt_monthly,
            ),
            min_monthly_living_cost=min_living,
            maintained_monthly_income=maintained_income,
            planned_major_expenses=expenses,
            existing_store_count=existing_store_count,
            existing_store_monthly_cashflow=(
                existing_store_cashflow if existing_store_count > 0 else None
            ),
            planned_operation_period=planned_period,
            target_monthly_income=target_income or None,
            target_payback_period_years=target_payback_years or None,
            startup_timing=startup_timing,
            desired_industries=desired_industries,
        )
        derived = compute_derived_metrics(profile)
        with st.spinner("에이전트가 후보 브랜드를 찾고 있습니다..."):
            st.session_state.graph_result = start_session(
                st.session_state.thread_id,
                asdict(profile),
                asdict(derived),
            )
        st.session_state.app_stage = "agent"
        st.rerun()


# ---------- 리포트 차트 (공용) ----------


def render_report_charts(report: dict):
    funding = report["funding"]
    costs = report["costs"]
    shortfall = report["shortfall"]

    render_section_heading("₩", "자금 조달 구조", "총 창업비용과 자기자본·대출·부족분을 함께 비교합니다.", icon=True)
    with st.container(border=True):
        m1, m2, m3 = st.columns(3)
        if costs["has_data"]:
            m1.metric("총 창업비용", f"{costs['total']:,.0f}만원")
            m3.metric(
                "부족분",
                f"{shortfall:,.0f}만원",
                delta_color="inverse" if shortfall > 0 else "off",
            )
        else:
            m1.metric("총 창업비용", "정보 없음")
            m3.metric("부족분", "판단 불가")
        m2.metric("조달 가능액", f"{funding['total']:,.0f}만원")

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                y=["조달"],
                x=[funding["equity"]],
                name="자기자본",
                orientation="h",
                marker_color=CATEGORICAL[0],
                hovertemplate="%{x:,.0f}만원<extra>자기자본</extra>",
            )
        )
        fig.add_trace(
            go.Bar(
                y=["조달"],
                x=[funding["debt"]],
                name="대출",
                orientation="h",
                marker_color=CATEGORICAL[1],
                hovertemplate="%{x:,.0f}만원<extra>대출</extra>",
            )
        )
        if shortfall:
            fig.add_trace(
                go.Bar(
                    y=["조달"],
                    x=[shortfall],
                    name="부족분",
                    orientation="h",
                    marker_color=STATUS["critical"],
                    hovertemplate="%{x:,.0f}만원<extra>부족분</extra>",
                )
            )
        fig.update_layout(barmode="stack", height=185, xaxis_title="만원", hovermode="y unified")
        fig.update_yaxes(showticklabels=False, title_text=None)
        render_plotly(fig)

        if not costs["has_data"]:
            st.caption(
                "⚠ 이 브랜드는 창업비용 원문을 숫자로 읽을 수 없어 부족분을 계산하지 못했습니다. 정보공개서 원문을 직접 확인하세요."
            )

    render_section_heading("↘", "대출 상환 스케줄", "월별 원금과 이자 부담의 변화를 확인합니다.", icon=True)
    with st.container(border=True):
        if report["loan_schedule"]:
            sched_df = pd.DataFrame(report["loan_schedule"])
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(
                    x=sched_df["month"],
                    y=sched_df["principal_paid"],
                    name="원금",
                    stackgroup="one",
                    line=dict(width=0.8, color=CATEGORICAL[0]),
                    hovertemplate="%{y:,.1f}만원<extra>원금</extra>",
                )
            )
            fig2.add_trace(
                go.Scatter(
                    x=sched_df["month"],
                    y=sched_df["interest"],
                    name="이자",
                    stackgroup="one",
                    line=dict(width=0.8, color=CATEGORICAL[1]),
                    hovertemplate="%{y:,.1f}만원<extra>이자</extra>",
                )
            )
            fig2.update_layout(height=335, xaxis_title="개월", yaxis_title="만원")
            render_plotly(fig2)
            principal_total = (
                report["loan_summary"]["total_payment"]
                - report["loan_summary"]["total_interest"]
            )
            st.caption(
                f"총 상환액 {report['loan_summary']['total_payment']:,.0f}만원 "
                f"(원금 {principal_total:,.0f}만원 + 이자 {report['loan_summary']['total_interest']:,.0f}만원)"
            )
        else:
            st.info("신규 대출 계획이 없거나 상환방식이 확정되지 않아 상환 스케줄을 계산하지 않았습니다.")

    render_section_heading("▥", "월 현금흐름 시뮬레이션", "낙관·기본·비관 시나리오의 36개월 예비자금 흐름을 비교합니다.", icon=True)
    st.caption("영업이익률 15% 가정, 개점 후 6개월 매출 램프업 반영. 실제 원가 구조는 브랜드별로 다를 수 있습니다.")
    with st.container(border=True):
        fig3 = go.Figure()
        has_any = False
        for key in ["optimistic", "base", "pessimistic"]:
            rows = report["scenarios"][key]["rows"]
            if not rows:
                continue
            has_any = True
            rdf = pd.DataFrame(rows)
            fig3.add_trace(
                go.Scatter(
                    x=rdf["month"],
                    y=rdf["reserve_balance"],
                    name=SCENARIO_LABELS[key],
                    line=dict(color=SCENARIO_COLORS[key], width=2.6),
                    hovertemplate="%{y:,.0f}만원<extra>" + SCENARIO_LABELS[key] + "</extra>",
                )
            )
        if has_any:
            fig3.add_hline(y=0, line_dash="dot", line_color=INK["baseline"])
            fig3.update_layout(
                height=390,
                xaxis_title="개월",
                yaxis_title="가구 예비자금 잔액 (만원)",
            )
            render_plotly(fig3)
        else:
            st.warning("이 브랜드는 매출 데이터가 없어 현금흐름 시뮬레이션을 만들 수 없습니다.")

    render_section_heading("✓", "시나리오별 핵심 지표", "손익분기·투자금 회수·예비자금 소진 위험을 나란히 확인합니다.", icon=True)
    cols = st.columns(3, gap="large")
    for col, key in zip(cols, ["optimistic", "base", "pessimistic"]):
        scenario = report["scenarios"][key]
        with col:
            with st.container(border=True):
                render_html(
                    f"""
                    <div class="ff-scenario-name">
                        <span>{escape(SCENARIO_LABELS[key])} 시나리오</span>
                        <span class="ff-scenario-dot" style="background:{SCENARIO_COLORS[key]}"></span>
                    </div>
                    """,
                    
                )
                st.metric(
                    "손익분기 도달",
                    f"{scenario['breakeven_month']}개월"
                    if scenario["breakeven_month"]
                    else "36개월 내 미도달",
                )
                st.metric(
                    "투자금 회수",
                    f"{scenario['payback_month']}개월"
                    if scenario["payback_month"]
                    else "36개월 내 미회수",
                )
                if scenario["runway_month"]:
                    status_html = (
                        '<div class="ff-status-strip is-danger">'
                        f"⚠ {scenario['runway_month']}개월차 예비자금 소진 경고"
                        "</div>"
                    )
                elif scenario["rows"]:
                    status_html = '<div class="ff-status-strip is-good">✓ 예비자금 소진 위험 없음</div>'
                else:
                    status_html = '<div class="ff-status-strip is-neutral">시뮬레이션 데이터 없음</div>'
                render_html(status_html)


# ---------- 에이전트 흐름 (②③④ + 루프, 전부 LangGraph가 주도) ----------


def render_candidates_preview(intr: dict):
    render_page_header(
        "STEP 2 · CANDIDATE DISCOVERY",
        "1차 후보 브랜드",
        "추천 모델이 입력 조건을 바탕으로 골라낸 후보입니다. 주요 수치를 카드에서 먼저 비교한 뒤 감수 진단으로 이동하세요.",
    )
    render_progress("candidates")
    candidates = intr["candidates"]
    render_summary_banner(
        "입력 조건과 가까운 후보를 찾았습니다",
        "카드의 창업비용·평당 연매출·가맹점 수를 비교하고, 선정 이유를 함께 확인하세요.",
        len(candidates),
        "개 후보",
    )

    if any(c.get("cost_is_estimated") for c in candidates):
        st.warning("⚠️ '비용 재확인' 표시가 있는 후보는 창업비용이 실제 값과 차이가 있을 수 있습니다.")

    render_candidate_grid(candidates)

    with st.container(border=True):
        render_section_heading("다음", "정보공개서 기반 감수 진단", "후보별 계약·운영 조건을 확인하면서 필요한 판단만 질문합니다.")
        if st.button("감수 진단 시작하기 →", use_container_width=True, type="primary"):
            submit("start")


def _render_detail_items(items: list[dict]) -> str:
    blocks = []
    for item in items:
        blocks.append(
            '<div class="ff-detail-item">'
            f'<b>{_html_text(item.get("label"), "항목")}</b>'
            f'{_html_text(item.get("text"), "내용 없음")}'
            "</div>"
        )
    return f'<div class="ff-detail-list">{"".join(blocks)}</div>'


def render_brand_detail(candidate: dict):
    """추천 근거와 정보공개서 상세를 화면에서 빠르게 읽을 수 있게 배치한다."""
    if candidate.get("agent_reasoning"):
        render_html(
            f"""
            <div class="ff-reasoning-box">
                <b>에이전트 판단</b><br>
                {_html_text(candidate['agent_reasoning'])}
            </div>
            """,
            
        )

    fit_reasons = candidate.get("fit_reasons", [])
    if fit_reasons:
        reason_items = "".join(
            f'<div class="ff-check-item"><span class="ff-check-icon">✓</span><span>{_html_text(reason)}</span></div>'
            for reason in fit_reasons
        )
        render_html(
            '<div class="ff-detail-title">추천 적합 근거</div>'
            f'<div class="ff-check-list">{reason_items}</div>',
            
        )

    detail = candidate.get("detail")
    if not detail:
        return

    cost = detail.get("cost") or {}
    if cost.get("has_data"):
        fees = cost.get("fees", {})
        render_html('<div class="ff-detail-title">창업비용 내역</div>')
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("가맹비", f"{fees.get('INITIAL_FRANCHISE_FEE', 0):,.0f}만원")
        c2.metric("보증금", f"{fees.get('DEPOSIT', 0):,.0f}만원")
        c3.metric("기타비용", f"{cost.get('other_cost', 0):,.0f}만원")
        c4.metric("총 비용", f"{cost.get('total', 0):,.0f}만원")

    sales_rows = [row for row in detail.get("sales_by_region", []) if row.get("region_name")]
    if sales_rows:
        render_html('<div class="ff-detail-title">지역별 평균 연매출</div>')
        sales_df = pd.DataFrame(sales_rows).rename(
            columns={
                "region_name": "지역",
                "average_annual_sales": "평균 연매출(만원)",
                "store_count": "가맹점 수",
            }
        )
        st.dataframe(sales_df, use_container_width=True, hide_index=True)

    support = detail.get("support")
    if support:
        pills = "".join(f'<span class="ff-pill">{_html_text(item)}</span>' for item in support)
        render_html(
            '<div class="ff-detail-title">가맹본부 지원제도</div>'
            f'<div class="ff-pill-list">{pills}</div>',
            
        )

    contract_risks = detail.get("contract_risks") or []
    operating_burdens = detail.get("operating_burdens") or []
    if contract_risks and operating_burdens:
        left, right = st.columns(2, gap="large")
        with left:
            render_html('<div class="ff-detail-title">계약·해지 관련 조건</div>')
            render_html(_render_detail_items(contract_risks))
        with right:
            render_html('<div class="ff-detail-title">지속적 운영 부담</div>')
            render_html(_render_detail_items(operating_burdens))
    elif contract_risks:
        render_html('<div class="ff-detail-title">계약·해지 관련 조건</div>')
        render_html(_render_detail_items(contract_risks))
    elif operating_burdens:
        render_html('<div class="ff-detail-title">지속적 운영 부담</div>')
        render_html(_render_detail_items(operating_burdens))


def render_agent_question(intr: dict, result: dict):
    render_page_header(
        "STEP 3 · RISK REVIEW",
        "후보 조건을 확인하고 있습니다",
        "정보공개서에서 확인한 계약·운영 조건 중 고객 판단이 필요한 항목만 질문합니다.",
    )
    render_progress("risk")

    paragraphs = format_agent_question(intr["text"]).split("\n\n")
    question_body = "".join(f"<p>{_html_text(paragraph)}</p>" for paragraph in paragraphs)
    render_html(
        f"""
        <div class="ff-question-card">
            <div class="ff-question-label">AGENT QUESTION</div>
            {question_body}
        </div>
        """,
        
    )

    candidates = result.get("model_candidates") or []
    with st.container(border=True):
        render_section_heading("응답", "판단을 알려주세요", "선택한 응답은 기존 에이전트 흐름에 그대로 전달됩니다.")
        if "선호" in intr["text"] and candidates:
            options = [f"후보 {i + 1}" for i in range(len(candidates))]
            choice = st.radio(
                "가장 끌리는 후보를 선택하세요",
                options,
                key=f"agentq_pref_{intr['question_id']}",
                horizontal=True,
            )
            if st.button("답변하고 계속 →", use_container_width=True, type="primary"):
                submit(choice)
            return

        render_html(
            '<div class="ff-scale-legend"><span>0 · 전혀 감수 불가</span><span>10 · 충분히 감수 가능</span></div>',
            
        )
        value = st.slider(
            "감수 가능 정도",
            0,
            10,
            5,
            key=f"agentq_slider_{intr['question_id']}",
        )
        text_value = st.text_input(
            "직접 답변 입력 · 선택",
            placeholder="예: 후보 2가 더 좋아요. 이유는…",
            key=f"agentq_text_{intr['question_id']}",
        )
        st.caption("직접 입력한 답변이 있으면 슬라이더 값 대신 해당 문장이 제출됩니다.")
        if st.button("답변하고 계속 →", use_container_width=True, type="primary"):
            submit(text_value.strip() if text_value.strip() else value)


def render_select_brand(result: dict, intr: dict):
    render_page_header(
        "STEP 3 · FINAL SHORTLIST",
        "최종 후보 브랜드",
        "조사 결과와 고객 응답을 반영해 좁힌 후보입니다. 핵심 수치와 상세 근거를 확인한 뒤 금융 시나리오를 볼 브랜드를 선택하세요.",
    )
    render_progress("select_brand")

    candidates = intr["candidates"]
    render_summary_banner(
        "최종 검토 후보가 정리되었습니다",
        "리스크 점수는 낮을수록 안정적입니다. 비용 재확인 표시와 상세 근거도 함께 확인하세요.",
        len(candidates),
        "개 후보",
    )
    render_candidate_grid(candidates, show_risk=True)

    if any(candidate.get("cost_is_estimated") for candidate in candidates):
        st.warning("⚠️ '비용 재확인' 표시가 있는 브랜드는 창업비용 원문을 별도로 확인하는 것이 좋습니다.")

    cols = [
        "brand_name",
        "brand_risk_score",
        "maximum_startup_total",
        "average_sales_per_3_3sqm",
        "cost_is_estimated",
    ]
    df = pd.DataFrame(candidates)
    for column in cols:
        if column not in df.columns:
            df[column] = None
    df["cost_is_estimated"] = (
        df["cost_is_estimated"].map({True: "⚠️ 재확인 필요", False: "-"}).fillna("-")
    )

    with st.expander("수치 비교표 펼쳐보기"):
        st.dataframe(
            df[cols].rename(
                columns={
                    "brand_name": "브랜드",
                    "brand_risk_score": "브랜드 리스크",
                    "maximum_startup_total": "창업비용(만원)",
                    "average_sales_per_3_3sqm": "평당(3.3㎡) 연매출(만원)",
                    "cost_is_estimated": "창업비용 신뢰도",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )

    comparison_pdf_path = intr.get("comparison_pdf_path")
    if comparison_pdf_path and Path(comparison_pdf_path).exists():
        with open(comparison_pdf_path, "rb") as file:
            st.download_button(
                "후보 비교 PDF 다운로드 · 표와 차트",
                data=file.read(),
                file_name="브랜드_비교_리포트.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

    render_section_heading("근거", "왜 이 순서로 추천됐나요?", "브랜드별 추천 이유와 정보공개서 상세 내용을 펼쳐서 확인하세요.")
    for candidate in candidates:
        with st.expander(
            f"{candidate['brand_name']} · 리스크 {candidate.get('brand_risk_score')}점"
        ):
            render_brand_detail(candidate)

    with st.container(border=True):
        render_section_heading("선택", "금융 시나리오를 확인할 브랜드", "선택 후 자금 조달·대출 상환·36개월 현금흐름을 보여드립니다.")
        names = [candidate["brand_name"] for candidate in candidates]
        choice = st.selectbox("브랜드 선택", names)
        if st.button("금융 시나리오 보기 →", use_container_width=True, type="primary"):
            chosen_id = next(
                candidate["disclosure_id"]
                for candidate in candidates
                if candidate["brand_name"] == choice
            )
            submit(chosen_id)


def _render_report_brand(candidate: dict):
    risk = candidate.get("brand_risk_score")
    risk_text = f"리스크 {risk}점" if risk is not None else "리스크 정보 없음"
    render_html(
        f"""
        <div class="ff-report-brand">
            <div>
                <div class="ff-report-brand-name">{_html_text(candidate.get('brand_name'), '브랜드명 없음')}</div>
                <div class="ff-report-brand-sub">{_html_text(candidate.get('industry_middle'), '업종 정보 없음')} · 선택 브랜드 금융 검토</div>
            </div>
            <div class="ff-score-badge">{escape(risk_text)}</div>
        </div>
        """,
        
    )


def render_loop_decision(result: dict, intr: dict):
    render_page_header(
        "STEP 4 · FINANCIAL SCENARIOS",
        "금융 시나리오 리포트",
        "선택 브랜드의 조달 구조와 대출 상환, 36개월 현금흐름을 시나리오별로 검토합니다.",
    )
    render_progress("report")

    candidate = result["selected_candidate"]
    report = result["report"]
    _render_report_brand(candidate)

    if result.get("report_summary"):
        st.info(md_safe(result["report_summary"]))

    with st.expander(
        f"선택 근거 자세히 보기 · 리스크 {candidate.get('brand_risk_score')}점",
        expanded=False,
    ):
        render_brand_detail(candidate)

    render_report_charts(report)

    if intr["needs_recommendation_loop"]:
        st.warning("모든 시나리오에서 감당이 어려운 것으로 보입니다.")

    render_section_heading("결정", "이 브랜드로 진행하시겠어요?", "확정하거나 다른 최종후보를 비교하고, 필요하면 조건을 완화해 다시 탐색할 수 있습니다.")
    c1, c2, c3 = st.columns(3, gap="large")
    with c1:
        with st.container(border=True):
            render_html('<div class="ff-action-title">브랜드 확정</div>')
            render_html('<div class="ff-action-desc">현재 리포트와 선택 브랜드를 최종 결과로 확정합니다.</div>')
            if st.button("이 브랜드로 확정", use_container_width=True, type="primary"):
                submit("accept")

    with c2:
        with st.container(border=True):
            render_html('<div class="ff-action-title">다른 후보 비교</div>')
            render_html('<div class="ff-action-desc">남아 있는 최종후보의 금융 시나리오로 전환합니다.</div>')
            others = [
                candidate_item
                for candidate_item in intr["final_candidates"]
                if candidate_item["disclosure_id"] != intr["selected_disclosure_id"]
            ]
            if others:
                other_choice = st.selectbox(
                    "다른 최종후보",
                    [candidate_item["brand_name"] for candidate_item in others],
                    key="other_pick",
                    label_visibility="collapsed",
                )
                if st.button("선택 후보 다시 보기", use_container_width=True):
                    chosen_id = next(
                        candidate_item["disclosure_id"]
                        for candidate_item in others
                        if candidate_item["brand_name"] == other_choice
                    )
                    submit(chosen_id)
            else:
                st.caption("다른 최종후보가 없습니다.")

    with c3:
        with st.container(border=True):
            render_html('<div class="ff-action-title">조건 완화</div>')
            render_html('<div class="ff-action-desc">현재 조건을 완화한 뒤 후보 탐색을 다시 진행합니다.</div>')
            if st.button("조건 완화하고 다시 찾기", use_container_width=True):
                submit("relax")


def render_terminal(result: dict):
    if result.get("status") == "report_ready" and result.get("report"):
        render_page_header(
            "COMPLETE · FINAL REPORT",
            "금융 시나리오가 확정되었습니다",
            "선택 브랜드의 최종 리포트를 화면에서 확인하고 PDF로 내려받을 수 있습니다.",
        )
        render_progress("report")
        candidate = result["selected_candidate"]
        _render_report_brand(candidate)
        st.success(f"'{candidate['brand_name']}' 브랜드로 확정하셨습니다.")

        if result.get("report_summary"):
            st.info(result["report_summary"])
        with st.expander(
            f"선택 근거 자세히 보기 · 리스크 {candidate.get('brand_risk_score')}점"
        ):
            render_brand_detail(candidate)
        render_report_charts(result["report"])

        pdf_path = result.get("pdf_path")
        if pdf_path and Path(pdf_path).exists():
            with open(pdf_path, "rb") as file:
                st.download_button(
                    "1페이지 리포트 PDF 다운로드",
                    data=file.read(),
                    file_name=f"{candidate['brand_name']}_창업리포트.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        else:
            st.caption("PDF 리포트를 만들지 못했습니다 — 화면의 내용은 그대로 유효합니다.")
    else:
        render_page_header(
            "SEARCH RESULT",
            "조건에 맞는 브랜드를 찾지 못했습니다",
            "예산·업종·감수 조건을 최대한 완화했지만 현재 조건과 맞는 후보가 없습니다.",
        )
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
        "agent_question": lambda: render_agent_question(intr, result),
        "select_brand": lambda: render_select_brand(result, intr),
        "loop_decision": lambda: render_loop_decision(result, intr),
    }
    dispatch[intr["type"]]()


render_topbar()
PAGES = {"survey": render_survey, "agent": render_agent}
PAGES[st.session_state.app_stage]()
