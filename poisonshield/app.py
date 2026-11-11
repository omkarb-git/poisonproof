"""PoisonShield application entry point."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
CLEANLAB_ROOT = PROJECT_ROOT.parent / "cleanlab-master"
if CLEANLAB_ROOT.exists() and str(CLEANLAB_ROOT) not in sys.path:
    sys.path.insert(0, str(CLEANLAB_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from backend.config import APP_NAME, APP_TITLE

st.set_page_config(
    page_title=APP_NAME,
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"About": APP_TITLE},
)

from ui.app import main  # noqa: E402 - page config must run first

main()
