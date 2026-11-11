"""Human label-review page."""

from __future__ import annotations

import html
from typing import Any

import pandas as pd
import streamlit as st

from backend.intelligence.reviewer import ACTIONS
from backend.types import AnalysisBundle

from .. import state
from ..components import (
    callout,
    fmt,
    render_no_analysis,
    render_no_dataset,
    render_sample_detail,
)
from ..theme import hero


def render() -> None:
    hero(
        "Label Review",
        "Record human decisions on a traceable evidence trail. Original data is never modified.",
        kicker="PoisonShield · Human review",
    )
    dataset = state.current_dataset()
    if dataset is None:
        render_no_dataset()
        return
    bundle = state.current_bundle()
    store = state.current_store()
    if bundle is None or store is None:
        render_no_analysis(dataset)
        return

    candidates = _candidate_ids(bundle, labelled_only=dataset.has_labels)
    if dataset.has_labels:
        st.caption(
            "Queue includes Cleanlab label issues, neighbourhood conflicts and samples with a "
            "suggested correction. A flag is a review priority, not proof of poisoning."
        )
    else:
        callout(
            "This dataset has no label column, so label corrections are unavailable. You can still "
            "mark feature-only flags for removal or uncertainty.",
            "info",
        )

    summary = store.summary()
    reviewed_in_queue = len([sid for sid in candidates if store.get(sid) is not None])
    columns = st.columns(5)
    with columns[0]:
        st.metric("Review queue", f"{len(candidates):,}")
    with columns[1]:
        st.metric("Reviewed from queue", f"{reviewed_in_queue:,}")
    with columns[2]:
        st.metric("Pending from queue", f"{max(0, len(candidates) - reviewed_in_queue):,}")
    with columns[3]:
        st.metric("Corrections approved", f"{summary.get('approve_correction', 0):,}")
    with columns[4]:
        st.metric("Marked for removal", f"{summary.get('remove', 0):,}")

    filters = st.columns([1.4, 1.4])
    with filters[0]:
        status = st.selectbox(
            "Queue filter",
            ["Pending only", "All queue samples", "Reviewed only"],
            key="ps_review_status_filter",
        )
    with filters[1]:
        search = st.text_input("Sample ID contains", key="ps_review_search")

    pending_ids = [sid for sid in candidates if store.get(sid) is None]
    reviewed_ids = [sid for sid in candidates if store.get(sid) is not None]
    options = candidates
    if status == "Pending only":
        options = pending_ids
    elif status == "Reviewed only":
        options = reviewed_ids
    if search.strip():
        options = [sid for sid in options if search.strip().lower() in sid.lower()]

    if not options:
        if status == "Pending only":
            st.success("No pending samples match the current filters. Review decisions are recorded below.")
        else:
            st.info("No samples match the current filters.")
    else:
        preselect = st.session_state.pop("ps_review_preselect", None)
        index = options.index(preselect) if preselect in options else 0
        selected = st.selectbox("Select a sample to review", options, index=index, key="ps_review_sample")
        if selected:
            _render_review_panel(bundle, store, selected, dataset.has_labels)

    st.divider()
    tabs = st.tabs(["Review history", "Decision summary", "Changes for sanitisation"])
    with tabs[0]:
        history = store.history_frame(limit=500)
        if history.empty:
            st.caption("No decisions recorded yet.")
        else:
            st.dataframe(history, width="stretch", hide_index=True)
    with tabs[1]:
        summary_rows = pd.DataFrame(
            [{"action": action, "label": ACTIONS.get(action, action), "count": summary.get(action, 0)} for action in ACTIONS if action != "clear"]
        )
        st.dataframe(summary_rows, width="stretch", hide_index=True)
        st.caption(f"Decisions recorded: {len(store.decisions):,}. History entries: {len(store.history):,}.")
    with tabs[2]:
        changes = store.changes_frame()
        if changes.empty:
            st.caption("No changes will be applied to the cleaned copy yet.")
        else:
            st.dataframe(changes, width="stretch", hide_index=True)
            st.caption("Open the Sanitisation page to generate the cleaned copy from these decisions.")


def _candidate_ids(bundle: AnalysisBundle, labelled_only: bool) -> list[str]:
    scores = bundle.risk.scores
    mask = (
        scores["label_issue"].fillna(False).astype(bool)
        | scores["relationship_conflict"].fillna(False).astype(bool)
        | scores["suggested_label"].notna()
    )
    if not labelled_only:
        mask = (
            scores["label_issue"].fillna(False).astype(bool)
            | scores["outlier"].fillna(False).astype(bool)
            | scores["duplicate"].fillna(False).astype(bool)
            | scores["exact_duplicate"].fillna(False).astype(bool)
            | scores["relationship_conflict"].fillna(False).astype(bool)
        )
    return scores[mask].sort_values("risk_score", ascending=False).index.tolist()


def _render_review_panel(bundle: AnalysisBundle, store, sample_id: str, labelled: bool) -> None:
    scores = bundle.risk.scores
    row = scores.loc[sample_id]
    original = row.get("label")
    original = None if original is None or pd.isna(original) else str(original)
    suggested = row.get("suggested_label")
    suggested = None if suggested is None or pd.isna(suggested) else str(suggested)
    decision = store.get(sample_id)

    render_sample_detail(bundle, sample_id)
    st.markdown("#### Record a decision")
    if decision is not None:
        st.info(
            f"Current decision: **{ACTIONS.get(decision.action, decision.action)}**"
            + (f" → `{decision.corrected_label}`" if decision.corrected_label is not None else "")
            + f" · reviewer: {decision.reviewer or 'n/a'} · {decision.timestamp}"
        )

    corrected_default = suggested or original or ""
    corrected_key = f"ps_corrected_{sample_id}"
    notes_key = f"ps_notes_{sample_id}"
    if corrected_key not in st.session_state:
        st.session_state[corrected_key] = corrected_default
    notes = st.text_area(
        "Reviewer notes (optional)",
        key=notes_key,
        placeholder="Why is this decision appropriate? Record what you checked.",
    )
    corrected_label = None
    if labelled:
        corrected_label = st.text_input(
            "Corrected label",
            key=corrected_key,
            help="Approve correction uses this value. The original table is never changed in place.",
        )

    actions = st.columns(6)
    with actions[0]:
        if labelled and st.button("Approve correction", type="primary", key=f"ps_approve_{sample_id}"):
            if not corrected_label.strip():
                st.error("A corrected label is required before approving a correction.")
            else:
                state.set_decision(
                    sample_id,
                    "approve_correction",
                    original_label=original,
                    suggested_label=suggested,
                    corrected_label=corrected_label.strip(),
                    notes=notes,
                )
                st.session_state["ps_flash"] = {
                    "message": f"Correction approved for {sample_id}.",
                    "kind": "success",
                }
                st.rerun()
    with actions[1]:
        if labelled and st.button("Reject correction", key=f"ps_reject_{sample_id}"):
            state.set_decision(
                sample_id,
                "reject_correction",
                original_label=original,
                suggested_label=suggested,
                notes=notes,
            )
            st.rerun()
    with actions[2]:
        if labelled and st.button("Retain original", key=f"ps_retain_{sample_id}"):
            state.set_decision(
                sample_id,
                "retain_original",
                original_label=original,
                suggested_label=suggested,
                notes=notes,
            )
            st.rerun()
    with actions[3]:
        if st.button("Mark uncertain", key=f"ps_uncertain_{sample_id}"):
            state.set_decision(
                sample_id,
                "uncertain",
                original_label=original,
                suggested_label=suggested,
                notes=notes,
            )
            st.rerun()
    with actions[4]:
        if st.button("Remove from cleaned set", key=f"ps_remove_{sample_id}"):
            state.set_decision(
                sample_id,
                "remove",
                original_label=original,
                suggested_label=suggested,
                notes=notes,
            )
            st.rerun()
    with actions[5]:
        if st.button("Clear decision", key=f"ps_clear_{sample_id}"):
            state.set_decision(sample_id, "clear", original_label=original)
            st.rerun()

    st.caption(
        "Removal affects only the exported cleaned copy. The uploaded dataset and original files stay "
        "unchanged, and every action is timestamped with the reviewer name."
    )
