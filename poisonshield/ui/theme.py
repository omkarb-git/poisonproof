"""PoisonShield visual theme and reusable styling."""

from __future__ import annotations

import streamlit as st

CSS = """
<style>
:root {
  --ps-navy: #07111f;
  --ps-deep: #0d2236;
  --ps-teal: #0e7490;
  --ps-cyan: #06b6d4;
  --ps-ink: #16202f;
  --ps-muted: #5b7385;
  --ps-line: #dbe6f0;
  --ps-surface: #ffffff;
  --ps-bg: #f4f7fb;
}

.stApp {
  background: linear-gradient(180deg, #f4f7fb 0%, #eef5f8 100%);
}

.block-container {
  padding-top: 2.1rem;
  padding-bottom: 3rem;
  max-width: 1420px;
}

.ps-hero {
  background: linear-gradient(125deg, #07111f 0%, #0d2236 55%, #0b3b44 100%);
  color: #ecfeff;
  border-radius: 20px;
  padding: 28px 34px;
  margin: 0 0 22px;
  box-shadow: 0 18px 40px rgba(7, 17, 31, 0.18);
}
.ps-hero h1 {
  margin: 0;
  font-size: 2rem;
  letter-spacing: 0.08em;
  font-weight: 750;
}
.ps-hero p {
  margin: 7px 0 0;
  color: #a5f3fc;
  font-size: 0.98rem;
}
.ps-hero .ps-kicker {
  color: #67e8f9;
  font-size: 0.72rem;
  letter-spacing: 0.22em;
  text-transform: uppercase;
  margin-bottom: 8px;
}

.ps-card {
  background: var(--ps-surface);
  border: 1px solid var(--ps-line);
  border-radius: 15px;
  padding: 18px 20px;
  margin: 10px 0 16px;
  box-shadow: 0 6px 18px rgba(15, 61, 74, 0.04);
}
.ps-card h3 { margin: 0 0 8px; color: var(--ps-teal); font-size: 1.02rem; }
.ps-card p { margin: 4px 0; }

.ps-badge {
  display: inline-block;
  border-radius: 999px;
  padding: 3px 10px;
  font-size: 0.72rem;
  font-weight: 650;
  letter-spacing: 0.04em;
  border: 1px solid transparent;
}
.ps-low { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.ps-moderate { background: #fffbeb; color: #92400e; border-color: #fde68a; }
.ps-high { background: #fff7ed; color: #9a3412; border-color: #fdba74; }
.ps-critical { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.ps-info { background: #ecfeff; color: #155e75; border-color: #a5f3fc; }
.ps-neutral { background: #f1f5f9; color: #334155; border-color: #e2e8f0; }

.ps-callout {
  border-left: 4px solid #f59e0b;
  background: #fffbeb;
  color: #7c4a03;
  padding: 12px 15px;
  border-radius: 9px;
  margin: 10px 0;
  font-size: 0.92rem;
}
.ps-callout.danger { border-left-color: #ef4444; background: #fef2f2; color: #7f1d1d; }
.ps-callout.ok { border-left-color: #10b981; background: #ecfdf5; color: #065f46; }
.ps-callout.info { border-left-color: #0e7490; background: #ecfeff; color: #155e75; }

.ps-muted { color: var(--ps-muted); font-size: 0.86rem; }
.ps-mono { font-family: 'Cascadia Code', 'Consolas', monospace; font-size: 0.82rem; }
.ps-section-title { margin: 4px 0 4px; color: var(--ps-ink); font-weight: 700; }
.ps-divider { height: 1px; background: var(--ps-line); margin: 18px 0; }

div[data-testid="stMetric"] {
  background: #ffffff;
  border: 1px solid var(--ps-line);
  border-radius: 13px;
  padding: 12px 14px;
  box-shadow: 0 4px 12px rgba(15, 61, 74, 0.035);
}
div[data-testid="stMetricLabel"] { color: var(--ps-muted); }
div[data-testid="stMetricValue"] { color: var(--ps-teal); }

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #07111f 0%, #0b1f2f 100%);
  border-right: 1px solid #123447;
  color: #e6f6fa;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] h4,
section[data-testid="stSidebar"] h5,
section[data-testid="stSidebar"] h6,
section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] {
  color: #e6f6fa;
}
section[data-testid="stSidebar"] .ps-muted,
section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
  color: #8fb3c2 !important;
}

/* Sidebar Metric Box: sleek dark container with clear bright numbers */
section[data-testid="stSidebar"] div[data-testid="stMetric"],
[data-testid="stSidebar"] [data-testid="stMetric"],
[data-testid="stSidebarContent"] [data-testid="stMetric"] {
  background: #0d2236 !important;
  border: 1px solid #1a3d54 !important;
  border-radius: 12px !important;
  padding: 12px 14px !important;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25) !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricLabel"],
section[data-testid="stSidebar"] [data-testid="stMetricLabel"] *,
[data-testid="stSidebar"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stMetricLabel"] * {
  color: #7dd3fc !important;
  font-size: 0.82rem !important;
  font-weight: 600 !important;
}
section[data-testid="stSidebar"] [data-testid="stMetricValue"],
section[data-testid="stSidebar"] [data-testid="stMetricValue"] *,
[data-testid="stSidebar"] [data-testid="stMetricValue"],
[data-testid="stSidebar"] [data-testid="stMetricValue"] * {
  color: #38bdf8 !important;
  font-weight: 800 !important;
}

/* Sidebar Expander (Session Management) */
section[data-testid="stSidebar"] div[data-testid="stExpander"],
[data-testid="stSidebar"] [data-testid="stExpander"],
[data-testid="stSidebar"] details {
  background: #0d2236 !important;
  border: 1px solid #1a3d54 !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] details summary {
  background: #0d2236 !important;
  color: #e6f6fa !important;
  border-radius: 10px !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary *,
[data-testid="stSidebar"] [data-testid="stExpander"] summary *,
[data-testid="stSidebar"] details summary * {
  color: #e6f6fa !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover,
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover,
[data-testid="stSidebar"] details summary:hover {
  background: #132f4c !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary:hover *,
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover *,
[data-testid="stSidebar"] details summary:hover * {
  color: #38bdf8 !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] summary svg,
[data-testid="stSidebar"] [data-testid="stExpander"] summary svg {
  fill: #7dd3fc !important;
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] div[data-testid="stExpanderDetails"],
[data-testid="stSidebar"] [data-testid="stExpanderDetails"] {
  background: #0a1b2b !important;
  border-top: 1px solid #1a3d54 !important;
}

/* Sidebar Buttons (Clear analysis, Clear dataset) - ALWAYS visible & crisp */
section[data-testid="stSidebar"] button,
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] [data-testid="baseButton-secondary"],
section[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
section[data-testid="stSidebar"] .stButton > button,
[data-testid="stSidebar"] button,
[data-testid="stSidebar"] button[kind="secondary"],
[data-testid="stSidebar"] [data-testid="baseButton-secondary"],
[data-testid="stSidebar"] [data-testid="stBaseButton-secondary"],
[data-testid="stSidebar"] .stButton > button {
  background: #11283d !important;
  color: #e0f2fe !important;
  border: 1px solid #1e4968 !important;
  border-radius: 8px !important;
  font-weight: 600 !important;
  transition: all 0.15s ease-in-out;
}
section[data-testid="stSidebar"] button *,
[data-testid="stSidebar"] button *,
[data-testid="stSidebar"] .stButton > button * {
  color: #e0f2fe !important;
}
section[data-testid="stSidebar"] button:hover,
section[data-testid="stSidebar"] button[kind="secondary"]:hover,
section[data-testid="stSidebar"] .stButton > button:hover,
[data-testid="stSidebar"] button:hover,
[data-testid="stSidebar"] .stButton > button:hover {
  background: #163856 !important;
  border-color: #38bdf8 !important;
  box-shadow: 0 0 10px rgba(56, 189, 248, 0.25) !important;
}
section[data-testid="stSidebar"] button:hover *,
[data-testid="stSidebar"] button:hover *,
[data-testid="stSidebar"] .stButton > button:hover * {
  color: #ffffff !important;
}

/* Sidebar Text Input (Reviewer name) */
section[data-testid="stSidebar"] div[data-baseweb="input"] {
  background: #0d2236 !important;
  border: 1px solid #1a3d54 !important;
  border-radius: 8px !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"] input {
  color: #e6f6fa !important;
  background: transparent !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"] input::placeholder {
  color: #64748b !important;
}
section[data-testid="stSidebar"] div[data-baseweb="input"]:focus-within {
  border-color: #38bdf8 !important;
  box-shadow: 0 0 0 1px #38bdf8 !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label {
  padding: 4px 2px;
  border-radius: 8px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
  background: rgba(14, 116, 144, 0.25);
}

.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] { border-radius: 8px 8px 0 0; padding: 8px 14px; }

.ps-flag-list { margin: 6px 0 0; padding-left: 18px; }
.ps-flag-list li { margin: 3px 0; }
</style>
"""


def inject_theme() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, kicker: str = "PoisonShield") -> None:
    st.markdown(
        f"""
        <div class="ps-hero">
          <div class="ps-kicker">{kicker}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def callout(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="ps-callout {kind}">{text}</div>', unsafe_allow_html=True)


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="ps-badge ps-{kind}">{text}</span>'
