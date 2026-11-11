"""Streamlit shell: sidebar navigation and section dispatch."""

from __future__ import annotations

import html

import streamlit as st

from backend.config import APP_NAME, TAGLINE, VERSION

from . import state
from .sections import SECTIONS, render_section
from .theme import inject_theme


def main() -> None:
    inject_theme()
    state.init_state()
    _render_sidebar()
    from .components import render_flash

    render_flash()
    section = st.session_state.get("ps_nav", "Overview")
    try:
        render_section(section)
    except Exception as exc:  # noqa: BLE001 - keep the app usable and expose the real error
        st.error("The page could not be rendered.")
        st.exception(exc)


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 8px 2px 14px;">
              <div style="font-size: 1.65rem; font-weight: 800; letter-spacing: .08em; color: #ecfeff;">🛡️ POISONSHIELD</div>
              <div style="color:#67e8f9; font-size:.74rem; letter-spacing:.16em; text-transform:uppercase;">Secure the Data. Protect the Model.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if "ps_nav_pending" in st.session_state:
            st.session_state["ps_nav"] = st.session_state.pop("ps_nav_pending")
        st.radio("Navigate", SECTIONS, key="ps_nav", label_visibility="collapsed")
        st.divider()

        reviewer = st.text_input(
            "Reviewer name",
            key="ps_reviewer",
            placeholder="Used on review decisions",
        )
        store = state.current_store()
        if store is not None:
            store.reviewer = reviewer

        dataset = state.current_dataset()
        bundle = state.current_bundle()
        st.markdown("#### Session")
        if dataset is None:
            st.caption("No dataset loaded.")
        else:
            st.markdown(f"**{html.escape(dataset.source_name)}**")
            st.caption(
                f"{dataset.modality.title()} · {len(dataset.sample_ids):,} samples · "
                f"{'labelled' if dataset.has_labels else 'unlabelled'}"
            )
        if bundle is not None:
            quality = bundle.risk.quality
            st.caption(
                f"Session {bundle.session_id} · detectors {quality.get('detector_completion', '0/0')}"
            )
            failed = [status.issue for status in bundle.audit.statuses if status.status == "failed"]
            if failed:
                st.warning("Failed detectors: " + ", ".join(failed))
            if bundle.warnings:
                st.caption(f"{len(bundle.warnings)} recorded warning(s)")
            st.metric("Quality index", f"{quality.get('quality_index', 0):.1f}" if quality.get("quality_index") is not None else "n/a")
        else:
            st.caption("No analysis run yet.")

        st.divider()
        with st.expander("Session management", expanded=False):
            if bundle is not None and st.button("Clear analysis (keep dataset)", key="ps_sidebar_clear_analysis"):
                state.clear_analysis(keep_dataset=True)
                st.rerun()
            if dataset is not None and st.button("Clear dataset", key="ps_sidebar_clear_dataset"):
                state.reset_all()
                st.rerun()
            st.caption("Clearing removes in-memory results. Workspace session files are retained on disk.")
        st.caption(f"{APP_NAME} v{VERSION} · {TAGLINE}")
