# dataviz 스킬의 검증된 기본 팔레트. 카테고리 색은 반드시 이 순서 그대로 사용한다(순환 생성 금지).
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

INK = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "gridline": "#e1e0d9",
    "baseline": "#c3c2b7",
    "surface": "#fcfcfb",
}

SCENARIO_LABELS = {"optimistic": "낙관", "base": "기본", "pessimistic": "비관"}
SCENARIO_COLORS = {"optimistic": CATEGORICAL[0], "base": CATEGORICAL[2], "pessimistic": CATEGORICAL[1]}
