"""Tabular analysis page."""

from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.utils import format_duration

from .. import state
from ..components import (
    callout,
    fmt,
    render_detector_status,
    render_no_analysis,
    render_quality_banner,
    render_risk_distribution,
    risk_histogram,
    sample_table,
    table_with_risk,
)
from ..theme import hero


def render() -> None:
    hero(
        "Tabular Analysis",
        "Cleanlab Datalab detectors, out-of-sample probabilities and per-sample risk evidence.",
        kicker="PoisonShield · Tabular",
    )
    dataset = state.current_dataset()
    if dataset is None:
        from ..components import render_no_dataset

        render_no_dataset()
        return
    if dataset.modality != "tabular":
        callout(
            "This page is for tabular datasets. Open <strong>Image Analysis</strong> for the current "
            "image dataset.",
            "info",
        )
        return
    bundle = state.current_bundle()
    if bundle is None:
        render_no_analysis(dataset)
        return

    render_quality_banner(bundle)
    tabs = st.tabs(["Detectors", "Risk signals", "Flagged samples", "Feature representation"])

    with tabs[0]:
        st.markdown("#### Cleanlab Datalab status")
        render_detector_status(bundle)
        st.markdown("#### Issue summary")
        summary = bundle.audit.summary.copy()
        if summary.empty:
            callout("Cleanlab returned no issue summary rows.", "danger")
        else:
            st.dataframe(summary, width="stretch", hide_index=True)
            st.caption(
                "Cleanlab issue scores are quality scores: higher is better. PoisonShield converts "
                "them to risk signals with (1 − quality)."
            )
        st.markdown("#### Out-of-sample prediction probabilities")
        meta = bundle.audit.pred_probs_meta or {}
        st.write(
            {
                "method": meta.get("method"),
                "classes": meta.get("classes"),
                "samples": meta.get("n_samples"),
                "logistic_error": meta.get("logistic_error"),
                "notes": meta.get("notes"),
            }
        )
        st.caption(
            "Out-of-sample probabilities are generated with cross-fitting. They are model-based "
            "evidence and not ground truth."
        )
        st.markdown("#### Processing time")
        st.write(
            {
                key: format_duration(value)
                for key, value in bundle.timings.items()
                if key in {"feature_build", "predicted_probabilities", "cleanlab_audit", "total"}
            }
        )

    with tabs[1]:
        scores = bundle.risk.scores
        columns = st.columns(3)
        with columns[0]:
            render_risk_distribution(bundle)
        with columns[1]:
            risk_histogram(bundle, "label_quality", "Cleanlab label quality")
        with columns[2]:
            risk_histogram(bundle, "outlier_quality", "Cleanlab outlier quality", "#f97316")

        st.markdown("#### Risk signal overview")
        numeric = [
            "label_quality",
            "outlier_quality",
            "near_duplicate_quality",
            "relationship_consistency",
            "integrity_score",
            "risk_score",
        ]
        available = [column for column in numeric if column in scores.columns]
        st.dataframe(
            scores[available].describe().T.reset_index().rename(columns={"index": "signal"}),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Label quality versus unified risk")
        scatter_data = scores.dropna(subset=["label_quality", "risk_score"]).copy()
        if not scatter_data.empty:
            scatter_data["label"] = scatter_data["label"].astype(str)
            scatter_data = scatter_data.reset_index()
            figure = px.scatter(
                scatter_data,
                x="label_quality",
                y="risk_score",
                color="risk_category",
                color_discrete_map={
                    "Low": "#10b981",
                    "Moderate": "#f59e0b",
                    "High": "#f97316",
                    "Critical": "#ef4444",
                },
                hover_data=["sample_id", "label", "label_issue", "outlier"],
                labels={"label_quality": "Cleanlab label quality (higher is better)"},
            )
            figure.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=20, b=10),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Risk",
            )
            st.plotly_chart(figure, width="stretch")
        else:
            st.caption("Label quality is unavailable for this dataset.")

    with tabs[2]:
        scores = bundle.risk.scores
        issue_columns = st.columns(4)
        with issue_columns[0]:
            st.metric("Label issues", int(scores["label_issue"].sum()))
        with issue_columns[1]:
            st.metric("Outliers", int(scores["outlier"].sum()))
        with issue_columns[2]:
            st.metric("Near duplicates", int(scores["duplicate"].sum()))
        with issue_columns[3]:
            st.metric("Exact duplicate rows", int(scores["exact_duplicate"].sum()))

        st.markdown("#### Cleanlab label issues")
        label_issues = scores[scores["label_issue"]].sort_values("risk_score", ascending=False)
        table_with_risk(sample_table(bundle, ids=label_issues.index.tolist(), limit=100), limit=100)
        st.markdown("#### Outliers")
        outliers = scores[scores["outlier"]].sort_values("risk_score", ascending=False)
        table_with_risk(sample_table(bundle, ids=outliers.index.tolist(), limit=100), limit=100)
        st.markdown("#### Near duplicates")
        duplicates = scores[scores["duplicate"] | scores["exact_duplicate"]].sort_values("risk_score", ascending=False)
        table_with_risk(sample_table(bundle, ids=duplicates.index.tolist(), limit=100), limit=100)

    with tabs[3]:
        meta = bundle.features.meta or {}
        st.markdown("#### Feature Representation Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Matrix Rows", f"{bundle.features.matrix.shape[0]:,}")
            st.caption("Analyzable samples")
        with c2:
            st.metric("Feature Dim", f"{bundle.features.matrix.shape[1]:,}")
            st.caption(f"SVD components: {meta.get('svd_components', 'N/A')}")
        with c3:
            num_cols = len(meta.get("numeric_columns", []))
            st.metric("Numeric Columns", f"{num_cols:,}")
            st.caption(f"{len(meta.get('categorical_columns', [])):,} categorical")
        with c4:
            st.metric("Build Time", format_duration(bundle.timings.get("feature_build")))
            st.caption(f"Probabilities: {format_duration(bundle.timings.get('predicted_probabilities'))}")

        with st.expander("Technical metadata & raw details", expanded=False):
            st.json(meta)
        st.markdown("#### Excluded columns and integrity notes")
        if bundle.risk.missing_evidence:
            st.warning("Unavailable evidence: " + ", ".join(bundle.risk.missing_evidence))
        st.markdown(
            "- Identifier-like, constant and entirely missing columns are excluded from features.\n"
            "- The selected evaluation ground-truth column is excluded to avoid leakage.\n"
            "- Text missing sentinels are treated as missing only in the feature representation.\n"
            "- The original dataframe is retained unchanged for display and export."
        )
        st.markdown("#### Warnings")
        for warning in bundle.warnings:
            st.markdown(f"- {html.escape(str(warning))}")
