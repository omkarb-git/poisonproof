"""Suspicious-sample triage page."""

from __future__ import annotations

import html

import pandas as pd
import streamlit as st

from .. import state
from ..components import (
    callout,
    fmt,
    render_no_analysis,
    render_no_dataset,
    render_rescore_controls,
    render_sample_detail,
    risk_tag,
    sample_table,
    table_with_risk,
)
from ..theme import hero


def render() -> None:
    hero(
        "Suspicious Samples",
        "Filter every flag, inspect traceable evidence and triage the review queue.",
        kicker="PoisonShield · Triage",
    )
    dataset = state.current_dataset()
    if dataset is None:
        render_no_dataset()
        return
    bundle = state.current_bundle()
    if bundle is None:
        render_no_analysis(dataset)
        return
    scores = bundle.risk.scores

    render_rescore_controls(bundle)

    st.markdown("#### Filters")
    filters = st.columns([1.2, 1.1, 1.3, 1.0])
    with filters[0]:
        categories = st.multiselect(
            "Risk category",
            ["Critical", "High", "Moderate", "Low"],
            default=["Critical", "High", "Moderate", "Low"],
            key="ps_sus_categories",
        )
    with filters[1]:
        min_risk = st.slider("Minimum risk score", 0.0, 1.0, 0.0, 0.01, key="ps_sus_min_risk")
    with filters[2]:
        flags = st.multiselect(
            "At least one flag",
            ["Label issue", "Outlier", "Near duplicate", "Exact duplicate", "Neighbourhood conflict", "Integrity"],
            default=[],
            key="ps_sus_flags",
        )
    with filters[3]:
        search = st.text_input("Sample ID contains", key="ps_sus_search")

    mask = scores["risk_category"].isin(categories) & (scores["risk_score"] >= float(min_risk))
    flag_columns = {
        "Label issue": "label_issue",
        "Outlier": "outlier",
        "Near duplicate": "duplicate",
        "Exact duplicate": "exact_duplicate",
        "Neighbourhood conflict": "relationship_conflict",
    }
    for flag in flags:
        if flag == "Integrity":
            mask &= scores["integrity_score"].fillna(0) >= 0.25
        else:
            mask &= scores[flag_columns[flag]].fillna(False).astype(bool)
    if search.strip():
        mask &= pd.Series(scores.index, index=scores.index).str.contains(search.strip(), case=False, regex=False)
    filtered = scores[mask].sort_values("risk_score", ascending=False)
    st.caption(f"{len(filtered):,} of {len(scores):,} samples match the current filters.")

    tabs = st.tabs(["Triage table", "Inspector", "Flag overview"])
    with tabs[0]:
        table_with_risk(sample_table(bundle, ids=filtered.index.tolist(), limit=500), limit=500)
        if not filtered.empty:
            st.download_button(
                "Download filtered risk table (CSV)",
                data=sample_table(bundle, ids=filtered.index.tolist(), limit=100000)
                .to_csv(index=False)
                .encode("utf-8-sig"),
                file_name="poisonshield_filtered_samples.csv",
                mime="text/csv",
            )

    with tabs[1]:
        if filtered.empty:
            st.info("No samples match the filters.")
            return
        options = filtered.index.tolist()[:1000]
        selected = st.selectbox("Select a sample", options, key="ps_sus_sample")
        if selected:
            render_sample_detail(bundle, selected)
            action_row = st.columns([1, 1, 2])
            with action_row[0]:
                if st.button("Open in Label Review", key="ps_sus_to_review"):
                    st.session_state["ps_review_preselect"] = selected
                    state.go_to("Label Review")
            with action_row[1]:
                if st.button("Open relationship map", key="ps_sus_to_rel"):
                    st.session_state["ps_rel_sample"] = selected
                    state.go_to("Relationship Explorer")

    with tabs[2]:
        summary = pd.DataFrame(
            {
                "Flag": [
                    "Label issues",
                    "Outliers",
                    "Near duplicates",
                    "Exact duplicates",
                    "Neighbourhood conflicts",
                    "Integrity concerns",
                    "High/Critical risk",
                ],
                "Count": [
                    int(scores["label_issue"].sum()),
                    int(scores["outlier"].sum()),
                    int(scores["duplicate"].sum()),
                    int(scores["exact_duplicate"].sum()),
                    int(scores["relationship_conflict"].sum()),
                    int((scores["integrity_score"].fillna(0) >= 0.25).sum()),
                    int(scores["risk_category"].isin(["High", "Critical"]).sum()),
                ],
            }
        )
        st.dataframe(summary, width="stretch", hide_index=True)
        st.markdown("#### Risk group availability")
        coverage = pd.DataFrame(
            {
                "Group": ["Label + neighbourhood", "Outlier", "Duplicate", "Integrity"],
                "Available samples": [
                    int(scores["label_check_available"].fillna(False).sum())
                    if "label_check_available" in scores
                    else 0,
                    int(scores["outlier_check_available"].fillna(False).sum())
                    if "outlier_check_available" in scores
                    else 0,
                    int(scores["duplicate_check_available"].fillna(False).sum())
                    if "duplicate_check_available" in scores
                    else 0,
                    int(scores["integrity_score"].notna().sum()),
                ],
            }
        )
        st.dataframe(coverage, width="stretch", hide_index=True)
        st.caption(
            "A sample's risk denominator excludes evidence groups that were unavailable for it. "
            "The per-sample evidence coverage is shown in the inspector."
        )
        if bundle.risk.missing_evidence:
            callout(
                "Unavailable evidence types: " + html.escape(", ".join(bundle.risk.missing_evidence)),
                "info",
            )
        if bundle.warnings:
            st.markdown("#### Warnings")
            for warning in bundle.warnings:
                st.markdown(f"- {html.escape(str(warning))}")
