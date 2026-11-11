"""Overview page."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.utils import format_duration, human_bytes

from .. import state
from ..components import (
    callout,
    fmt,
    provenance_status,
    render_dataset_line,
    render_detector_status,
    render_quality_banner,
    render_quality_cards,
    render_rescore_controls,
    render_risk_distribution,
    render_run_controls,
    risk_tag,
    sample_table,
    table_with_risk,
)
from ..theme import hero


def render() -> None:
    hero(
        "Command Overview",
        "One screen for dataset health, detector coverage, risk priorities and next actions.",
        kicker="PoisonShield · Audit overview",
    )
    dataset = state.current_dataset()
    bundle = state.current_bundle()
    if dataset is None:
        _render_welcome()
        return
    if bundle is None:
        _render_loaded_only(dataset)
        return

    render_quality_banner(bundle)
    render_dataset_line(dataset)
    quality = bundle.risk.quality
    render_quality_cards(bundle)

    st.divider()
    left, right = st.columns([1.05, 1.45])
    with left:
        st.markdown("#### Risk distribution")
        render_risk_distribution(bundle)
        st.caption(
            f"Mean review risk {fmt(quality.get('mean_risk'), 3)} · "
            f"mean evidence coverage {fmt(quality.get('mean_evidence_coverage'), 2)}"
        )
    with right:
        st.markdown("#### Highest-priority samples")
        table_with_risk(sample_table(bundle, limit=15), limit=15)

    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Requiring review", f"{quality.get('n_requiring_review', 0):,}")
        st.caption("High/Critical risk or at least one detector flag")
    with col2:
        st.metric("Relationship conflicts", f"{quality.get('n_relationship_conflicts', 0):,}")
        st.caption("Neighbourhood label disagreement")
    with col3:
        st.metric("Detector completion", str(quality.get("detector_completion", "0/0")))
        st.caption("Failed or skipped detectors are reported explicitly")

    with st.expander("Detector status and data-quality details", expanded=False):
        render_detector_status(bundle)
        st.markdown("##### Dataset-level findings")
        class_imbalance = quality.get("class_imbalance", {})
        non_iid = quality.get("non_iid", {})
        st.write(
            {
                "class-imbalance quality score": class_imbalance.get("score"),
                "non-IID quality score": non_iid.get("score"),
                "class-imbalance details": class_imbalance.get("info"),
                "non-IID details": non_iid.get("info"),
            }
        )
        st.caption(
            "Class-imbalance and non-IID are dataset-level findings and are excluded from "
            "per-sample risk by design."
        )
        if bundle.risk.missing_evidence:
            st.warning("Unavailable evidence: " + ", ".join(bundle.risk.missing_evidence))

    render_rescore_controls(bundle)

    st.divider()
    st.markdown("#### Processing and provenance")
    processing = st.columns(4)
    with processing[0]:
        st.metric("Total time", format_duration(bundle.timings.get("total")))
    with processing[1]:
        st.metric("Feature build", format_duration(bundle.timings.get("feature_build")))
    with processing[2]:
        st.metric("Cleanlab audit", format_duration(bundle.timings.get("cleanlab_audit")))
    with processing[3]:
        st.metric("Peak traced memory", fmt(bundle.memory.get("tracemalloc_peak_mb"), 1, " MB"))
    status, message = provenance_status(dataset)
    if status is True:
        callout(html.escape(message), "ok")
    elif status is False:
        callout(html.escape(message), "danger")
    else:
        st.caption(message)
    st.caption(f"Session: {bundle.session_id} · workspace: {bundle.workspace}")

    if bundle.warnings:
        with st.expander(f"Warnings ({len(bundle.warnings)})", expanded=False):
            for warning in bundle.warnings:
                st.markdown(f"- {html.escape(str(warning))}")

    st.markdown("#### Next actions")
    action_columns = st.columns(4)
    with action_columns[0]:
        if st.button("Explore suspicious samples", key="ps_go_suspicious"):
            state.go_to("Suspicious Samples")
    with action_columns[1]:
        if st.button("Review labels", key="ps_go_review"):
            state.go_to("Label Review")
    with action_columns[2]:
        if st.button("Open relationship map", key="ps_go_rel"):
            state.go_to("Relationship Explorer")
    with action_columns[3]:
        if st.button("Export report", key="ps_go_report"):
            state.go_to("Reports / Export")


def _render_welcome() -> None:
    st.markdown("### Start an audit")
    columns = st.columns(3)
    cards = [
        (
            "1 · Upload",
            "Add a CSV/XLSX table or an image ZIP. A label column is optional. Originals are copied "
            "into the workspace unchanged.",
        ),
        (
            "2 · Analyse",
            "Cleanlab detectors and PoisonShield relationships run on real data. Every detector "
            "status, warning and timing is recorded.",
        ),
        (
            "3 · Review & export",
            "Inspect evidence, record human decisions, generate a sanitised copy and export a "
            "traceable report.",
        ),
    ]
    for column, (title, body) in zip(columns, cards, strict=True):
        with column:
            st.markdown(
                f'<div class="ps-card"><h3>{html.escape(title)}</h3><p>{html.escape(body)}</p></div>',
                unsafe_allow_html=True,
            )
    if st.button("Open Upload", type="primary", key="ps_welcome_upload"):
        state.go_to("Upload")
    st.markdown("#### What PoisonShield does not do")
    st.markdown(
        "- It does not simulate poisoning in the normal workflow or prefill demo metrics.\n"
        "- It does not claim that a risk flag proves poisoning.\n"
        "- It does not modify uploaded files; sanitisation is written to a new copy.\n"
        "- It does not claim sanitisation always improves model accuracy."
    )


def _render_loaded_only(dataset) -> None:
    render_dataset_line(dataset)
    profile = dataset.profile
    columns = st.columns(4)
    with columns[0]:
        st.metric("Samples", f"{profile.n_samples:,}")
    with columns[1]:
        st.metric("Columns", f"{profile.n_columns:,}")
    with columns[2]:
        st.metric("Labels", "yes" if dataset.has_labels else "no")
    with columns[3]:
        st.metric("Task", dataset.task.title())
    callout(
        "No analysis has been run for this dataset yet. Run the audit below, or open Upload to "
        "change the label/task configuration.",
        "info",
    )
    render_run_controls(dataset)
