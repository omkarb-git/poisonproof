"""Shared UI components for PoisonShield."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from backend.types import AnalysisBundle, Explanation, LoadedDataset
from backend.utils import format_duration, human_bytes, path_fingerprint

from . import state
from .theme import badge, callout

RISK_COLORS = {
    "Low": "#10b981",
    "Moderate": "#f59e0b",
    "High": "#f97316",
    "Critical": "#ef4444",
}
STATUS_ICONS = {"completed": "✅ Completed", "skipped": "⏭️ Skipped", "failed": "❌ Failed"}


def flash(message: str, kind: str = "success") -> None:
    st.session_state["ps_flash"] = {"message": message, "kind": kind}


def render_flash() -> None:
    payload = st.session_state.get("ps_flash")
    if not payload:
        return
    st.session_state["ps_flash"] = None
    message = payload["message"]
    kind = payload["kind"]
    if kind == "error":
        st.error(message)
    elif kind == "warning":
        st.warning(message)
    elif kind == "info":
        st.info(message)
    else:
        st.success(message)


def fmt(value: Any, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if np.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}{suffix}"


def pct(value: Any, digits: int = 1) -> str:
    return fmt(value, digits, "%")


def risk_tag(category: str) -> str:
    return badge(str(category), str(category).lower() if category in RISK_COLORS else "neutral")


def detector_frame(bundle: AnalysisBundle) -> pd.DataFrame:
    rows = []
    for status in bundle.audit.statuses:
        rows.append(
            {
                "Detector": status.issue,
                "Status": STATUS_ICONS.get(status.status, status.status),
                "Flagged": status.n_flagged if status.n_flagged is not None else "n/a",
                "Duration": format_duration(status.duration_s),
                "Detail": status.detail,
            }
        )
    return pd.DataFrame(rows)


def render_detector_status(bundle: AnalysisBundle) -> None:
    failed = [s for s in bundle.audit.statuses if s.status == "failed"]
    skipped = [s for s in bundle.audit.statuses if s.status == "skipped"]
    if failed:
        names = ", ".join(s.issue for s in failed)
        callout(
            f"<strong>Detector failure:</strong> {html.escape(names)}. Failed detectors are not "
            "evidence of a clean dataset; their issue types are excluded from per-sample risk.",
            "danger",
        )
    if skipped:
        names = ", ".join(s.issue for s in skipped)
        callout(
            f"<strong>Not applicable / skipped:</strong> {html.escape(names)}. See detector details.",
            "info",
        )
    frame = detector_frame(bundle)
    if frame.empty:
        callout("No detector statuses were recorded.", "danger")
    else:
        st.dataframe(frame, width="stretch", hide_index=True)


def render_dataset_line(dataset: LoadedDataset) -> None:
    st.markdown(
        f"**{html.escape(dataset.source_name)}** &nbsp; "
        f"{badge(dataset.modality.upper(), 'info')} &nbsp; "
        f"{badge(dataset.task.replace('_', ' ').title(), 'neutral')} &nbsp; "
        f"{badge(f'{len(dataset.sample_ids):,} samples', 'neutral')}",
        unsafe_allow_html=True,
    )
    if dataset.label_column:
        st.caption(f"Label column: {dataset.label_column}")
    else:
        st.caption("No label column: feature-only auditing will run.")


def render_no_dataset() -> None:
    callout(
        "No dataset is loaded. Open <strong>Upload</strong> in the sidebar to add a labelled or "
        "unlabelled tabular dataset or an image dataset.",
        "info",
    )


def render_no_analysis(dataset: LoadedDataset | None) -> None:
    if dataset is None:
        render_no_dataset()
        return
    callout(
        "The dataset is loaded but has not been analysed yet. Configure the settings below and "
        "run the audit, or open <strong>Upload</strong> to review the dataset configuration.",
        "info",
    )


def render_analysis_settings(dataset: LoadedDataset, show_header: bool = True) -> Any:
    """Render tuning controls and return an AnalysisConfig (unvalidated)."""

    if show_header:
        st.markdown("#### Analysis settings")
    with st.expander("Detection and risk settings", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.slider(
                "Neighbours (k)",
                min_value=3,
                max_value=50,
                step=1,
                key="ps_k",
                help="Neighbourhood size used by the relationship engine and KNN probabilities.",
            )
            st.number_input("Random seed", min_value=0, max_value=2**31 - 1, step=1, key="ps_seed")
        with col2:
            if dataset.modality == "image":
                extractor_options = ["auto"] + [
                    option["key"] for option in _extractor_options()
                ]
                labels = {"auto": "Auto (DINOv2 → ResNet-50 → classic)"}
                labels.update({option["key"]: option["label"] for option in _extractor_options()})
                st.selectbox(
                    "Image feature extractor",
                    options=extractor_options,
                    key="ps_extractor",
                    format_func=lambda key: labels.get(key, key),
                )
                st.slider(
                    "Embedding batch size",
                    min_value=1,
                    max_value=64,
                    step=1,
                    key="ps_batch",
                )
            else:
                st.caption(
                    "Tabular features are built from numeric, categorical and datetime columns; "
                    "identifier-like, constant and entirely missing columns are excluded."
                )
                st.metric("Configured k", st.session_state.get("ps_k", 15))
        with col3:
            st.caption(
                "Risk is a weighted average of available evidence groups. Correlated label and "
                "neighbourhood signals are fused into one group; unavailable groups are excluded "
                "and coverage is reported."
            )
            weights = [
                float(st.session_state.get("ps_weight_label", 0.5)),
                float(st.session_state.get("ps_weight_outlier", 0.2)),
                float(st.session_state.get("ps_weight_duplicate", 0.1)),
                float(st.session_state.get("ps_weight_integrity", 0.2)),
            ]
            total = sum(weights)
            if total <= 0:
                st.error("At least one risk weight must be greater than zero.")
            else:
                st.caption(
                    "Normalised weights: "
                    + " · ".join(
                        f"{name} {weight / total:.2f}"
                        for name, weight in zip(
                            ["label", "outlier", "duplicate", "integrity"], weights, strict=True
                        )
                    )
                )

        st.markdown("##### Risk evidence weights")
        w1, w2, w3, w4 = st.columns(4)
        with w1:
            st.slider("Label + neighbourhood", 0.0, 1.0, step=0.05, key="ps_weight_label")
        with w2:
            st.slider("Outlier", 0.0, 1.0, step=0.05, key="ps_weight_outlier")
        with w3:
            st.slider("Duplicate", 0.0, 1.0, step=0.05, key="ps_weight_duplicate")
        with w4:
            st.slider("Integrity", 0.0, 1.0, step=0.05, key="ps_weight_integrity")

        st.markdown("##### Risk category thresholds")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.slider("Low upper bound", 0.05, 0.95, step=0.05, key="ps_thr_low")
        with t2:
            st.slider("Moderate upper bound", 0.05, 0.95, step=0.05, key="ps_thr_moderate")
        with t3:
            st.slider("High upper bound", 0.05, 0.95, step=0.05, key="ps_thr_high")
        if not (
            float(st.session_state.get("ps_thr_low", 0.3))
            < float(st.session_state.get("ps_thr_moderate", 0.55))
            < float(st.session_state.get("ps_thr_high", 0.75))
        ):
            st.error("Thresholds must satisfy Low < Moderate < High.")

    return state.build_analysis_config(dataset)


def _extractor_options() -> list[dict[str, Any]]:
    from backend.features.image_features import available_extractors

    return available_extractors()


def render_run_controls(dataset: LoadedDataset) -> None:
    """Settings + run button with live progress."""

    config = render_analysis_settings(dataset)
    existing = state.current_bundle()
    store = state.current_store()
    has_decisions = bool(store and store.decisions)
    confirm_rerun = True
    if existing and has_decisions:
        callout(
            "Re-running the analysis starts a <strong>new session</strong>. Review decisions recorded "
            "in the current session will not be copied to the new one. Export them first if needed.",
            "info",
        )
        confirm_rerun = st.checkbox(
            "I understand that re-analysis creates a new review session",
            key="ps_confirm_rerun",
        )

    label = "Run PoisonShield analysis" if not existing else "Re-run analysis (new session)"
    if not st.button(label, type="primary", key="ps_run_analysis", disabled=not confirm_rerun):
        if existing:
            st.caption(
                "The current analysis is still loaded. Use the sidebar pages to inspect it, or "
                "re-run with new settings above."
            )
        return

    try:
        config.validated()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Invalid analysis settings: {exc}")
        return

    progress_bar = st.progress(0.0)
    progress_text = st.empty()

    def report(fraction: float, message: str) -> None:
        progress_bar.progress(float(max(0.0, min(1.0, fraction))))
        progress_text.caption(message)

    from backend.pipeline import AnalysisError, analyse_dataset

    try:
        bundle = analyse_dataset(dataset, config, progress=report)
    except AnalysisError as exc:
        progress_text.empty()
        st.session_state["ps_last_run_error"] = str(exc)
        st.error(f"Analysis failed: {exc}")
        st.caption(
            "The dataset remains loaded. Failed detectors are never reported as a clean dataset."
        )
        return
    except Exception as exc:  # noqa: BLE001
        progress_text.empty()
        st.session_state["ps_last_run_error"] = f"{type(exc).__name__}: {exc}"
        st.exception(exc)
        return
    finally:
        progress_bar.progress(1.0)

    state.set_bundle(bundle)
    flash(
        f"Analysis complete in {format_duration(bundle.timings.get('total'))}. "
        f"Session {bundle.session_id} is ready.",
        "success",
    )
    st.rerun()


def render_rescore_controls(bundle: AnalysisBundle) -> None:
    """Allow changing risk weights/thresholds without re-running detectors."""

    with st.expander("Re-score risk with different weights or thresholds", expanded=False):
        st.caption(
            "Re-scoring reuses the stored detector evidence. Weights are re-normalised over the "
            "evidence groups that are available for each sample."
        )
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.slider("Label + neighbourhood", 0.0, 1.0, step=0.05, key="ps_weight_label")
        with col2:
            st.slider("Outlier", 0.0, 1.0, step=0.05, key="ps_weight_outlier")
        with col3:
            st.slider("Duplicate", 0.0, 1.0, step=0.05, key="ps_weight_duplicate")
        with col4:
            st.slider("Integrity", 0.0, 1.0, step=0.05, key="ps_weight_integrity")
        t1, t2, t3 = st.columns(3)
        with t1:
            st.slider("Low upper bound", 0.05, 0.95, step=0.05, key="ps_thr_low")
        with t2:
            st.slider("Moderate upper bound", 0.05, 0.95, step=0.05, key="ps_thr_moderate")
        with t3:
            st.slider("High upper bound", 0.05, 0.95, step=0.05, key="ps_thr_high")
        if st.button("Apply risk settings and re-score", type="primary", key="ps_rescore"):
            from backend.config import RiskThresholds, RiskWeights
            from backend.intelligence.risk_engine import rescore_bundle
            from backend.reporting.artifacts import save_session_artifacts

            try:
                weights = RiskWeights(
                    label_evidence=float(st.session_state.get("ps_weight_label", 0.5)),
                    outlier_evidence=float(st.session_state.get("ps_weight_outlier", 0.2)),
                    duplicate_evidence=float(st.session_state.get("ps_weight_duplicate", 0.1)),
                    integrity_evidence=float(st.session_state.get("ps_weight_integrity", 0.2)),
                ).validated()
                thresholds = RiskThresholds(
                    low_max=float(st.session_state.get("ps_thr_low", 0.3)),
                    moderate_max=float(st.session_state.get("ps_thr_moderate", 0.55)),
                    high_max=float(st.session_state.get("ps_thr_high", 0.75)),
                ).validated()
                rescore_bundle(bundle, weights, thresholds)
                try:
                    save_session_artifacts(bundle)
                except Exception:
                    pass
                flash("Risk scores were re-computed from the stored detector evidence.", "success")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not re-score: {exc}")


def render_risk_distribution(bundle: AnalysisBundle) -> None:
    distribution = bundle.risk.quality.get("risk_distribution", {})
    frame = pd.DataFrame(
        {
            "Risk category": list(distribution.keys()),
            "Samples": list(distribution.values()),
        }
    )
    if frame.empty:
        return
    figure = px.bar(
        frame,
        x="Risk category",
        y="Samples",
        color="Risk category",
        color_discrete_map=RISK_COLORS,
        text="Samples",
    )
    figure.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        height=280,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(figure, width="stretch")


def render_quality_cards(bundle: AnalysisBundle) -> None:
    quality = bundle.risk.quality
    columns = st.columns(4)
    with columns[0]:
        st.metric("Quality index", fmt(quality.get("quality_index"), 1))
        st.caption("100 × (1 − mean review risk); screening summary only.")
    with columns[1]:
        st.metric("Label issues", f"{quality.get('n_label_issues', 0):,}")
        st.caption("Cleanlab label quality detector")
    with columns[2]:
        st.metric("Outliers", f"{quality.get('n_outliers', 0):,}")
        st.caption("Cleanlab feature-space outliers")
    with columns[3]:
        st.metric("Duplicates", f"{quality.get('n_near_duplicates', 0):,}")
        st.caption(f"Near duplicates · {quality.get('n_exact_duplicates', 0):,} exact")


def render_quality_banner(bundle: AnalysisBundle) -> None:
    quality = bundle.risk.quality
    completion = quality.get("detector_completion", "0/0")
    failed = [s.issue for s in bundle.audit.statuses if s.status == "failed"]
    missing = bundle.risk.missing_evidence
    if failed:
        callout(
            "Detector completion is "
            f"<strong>{completion}</strong>. Failed: {html.escape(', '.join(failed))}. "
            "Do not interpret the dataset as clean for those issue types.",
            "danger",
        )
    elif missing:
        callout(
            f"Detector completion is <strong>{completion}</strong>. Unavailable evidence: "
            f"{html.escape(', '.join(missing))}.",
            "info",
        )
    else:
        callout(
            f"Detector completion is <strong>{completion}</strong>. Risk scores prioritise manual "
            "review and do not confirm poisoning.",
            "ok",
        )


def factor_frame(bundle: AnalysisBundle, sample_id: str) -> pd.DataFrame:
    scores = bundle.risk.scores
    if sample_id not in scores.index:
        return pd.DataFrame()
    row = scores.loc[sample_id]
    weights = bundle.risk.weights
    evidence = bundle.risk.evidence.reindex([sample_id]) if not bundle.risk.evidence.empty else pd.DataFrame()
    groups = [
        ("Label + neighbourhood", "label_evidence", "label_group"),
        ("Outlier", "outlier_evidence", "outlier_signal"),
        ("Duplicate", "duplicate_evidence", "duplicate_signal"),
        ("Integrity", "integrity_evidence", "integrity_signal"),
    ]
    rows = []
    total_weight = 0.0
    for label, availability_key, signal_key in groups:
        available = bool(evidence[availability_key].iloc[0]) if availability_key in evidence else False
        weight_map = {
            "label_evidence": "label_evidence",
            "outlier_evidence": "outlier_evidence",
            "duplicate_evidence": "duplicate_evidence",
            "integrity_evidence": "integrity_evidence",
        }
        weight = float(weights.get(weight_map[availability_key], 0.0))
        signal = None
        if signal_key in evidence:
            value = evidence[signal_key].iloc[0]
            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                signal = float(value)
        rows.append(
            {
                "Evidence group": label,
                "Available": "yes" if available else "no (excluded)",
                "Signal (0–1)": signal,
                "Weight": weight,
                "Contribution": weight * signal if available and signal is not None else None,
            }
        )
        if available:
            total_weight += weight
    frame = pd.DataFrame(rows)
    if total_weight:
        st.caption(
            f"Risk = weighted mean of available signals = "
            f"{float(row['risk_score']):.3f}. Available weight coverage: {total_weight:.2f}. "
            "The final denominator excludes unavailable groups."
        )
    return frame


def render_sample_detail(bundle: AnalysisBundle, sample_id: str, show_image: bool = True) -> None:
    scores = bundle.risk.scores
    if sample_id not in scores.index:
        st.warning(f"Sample {sample_id} is not present in this analysis.")
        return
    row = scores.loc[sample_id]
    columns = st.columns(5)
    with columns[0]:
        st.metric("Risk score", fmt(row.get("risk_score"), 3))
    with columns[1]:
        st.metric("Category", str(row.get("risk_category", "n/a")))
    with columns[2]:
        st.metric("Original label", str(row.get("label") if pd.notna(row.get("label")) else "unlabelled"))
    with columns[3]:
        label = row.get("suggested_label")
        st.metric("Suggested label", "—" if label is None or pd.isna(label) else str(label))
    with columns[4]:
        st.metric("Evidence coverage", fmt(row.get("evidence_coverage"), 2))

    action = row.get("recommended_action")
    if isinstance(action, str) and action:
        callout(html.escape(action), "info")

    st.markdown("##### Evidence contribution")
    factors = factor_frame(bundle, sample_id)
    if not factors.empty:
        st.dataframe(
            factors,
            width="stretch",
            hide_index=True,
            column_config={
                "Signal (0–1)": st.column_config.NumberColumn(format="%.3f"),
                "Weight": st.column_config.NumberColumn(format="%.2f"),
                "Contribution": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    explanation = bundle.risk.explanations.get(sample_id)
    if explanation is None:
        st.caption("No stored explanation for this sample (it was not flagged).")
    else:
        render_explanation(explanation)

    if show_image and bundle.dataset.modality == "image":
        record = next((r for r in bundle.dataset.images or [] if r.sample_id == sample_id), None)
        if record is not None:
            render_image(record, caption=f"{sample_id} · {record.rel_path}")

    neighbours = bundle.relationships.neighbours_for(sample_id, limit=12)
    if not neighbours.empty:
        st.markdown("##### Nearest neighbours")
        st.dataframe(
            neighbours,
            width="stretch",
            hide_index=True,
            column_config={
                "distance": st.column_config.NumberColumn(format="%.4f"),
                "similarity": st.column_config.NumberColumn(format="%.4f"),
            },
        )

    if sample_id in bundle.relationships.projection.index:
        projection = bundle.relationships.projection.loc[[sample_id]].reset_index()
        st.caption(
            "Projection coordinates: "
            + ", ".join(f"{key}={fmt(value, 3)}" for key, value in projection.iloc[0].items())
        )


def render_explanation(explanation: Explanation) -> None:
    st.markdown("##### Explanation")
    st.markdown(f"**{html.escape(explanation.sample_id)}** — {risk_tag(explanation.risk_category)}", unsafe_allow_html=True)
    st.write(explanation.narrative)
    if explanation.findings:
        frame = pd.DataFrame(
            [
                {
                    "Finding": finding.kind,
                    "Severity": finding.severity,
                    "Text": finding.text,
                }
                for finding in explanation.findings
            ]
        )
        st.dataframe(
            frame,
            width="stretch",
            hide_index=True,
            column_config={"Severity": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f")},
        )
    if explanation.missing_evidence:
        st.caption("Missing evidence: " + ", ".join(explanation.missing_evidence))


def render_image(record: Any, caption: str | None = None, width: int = 320) -> None:
    if not record.readable:
        callout(
            f"<strong>Unreadable image:</strong> {html.escape(record.rel_path)} — "
            f"{html.escape(str(record.error))}",
            "danger",
        )
        return
    try:
        with Image.open(record.path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((width, width))
            st.image(preview, caption=caption or record.rel_path, width=width)
    except Exception as exc:  # noqa: BLE001
        callout(f"Preview unavailable for {html.escape(record.rel_path)}: {html.escape(str(exc))}", "danger")


def risk_histogram(bundle: AnalysisBundle, column: str, title: str, color: str = "#0e7490") -> None:
    values = pd.to_numeric(bundle.risk.scores.get(column), errors="coerce").dropna()
    if values.empty:
        st.caption(f"No {title.lower()} values are available.")
        return
    figure = px.histogram(values, nbins=30, labels={"value": title, "count": "Samples"})
    figure.update_traces(marker_color=color)
    figure.update_layout(
        title=title,
        height=290,
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis_title=title,
        yaxis_title="Samples",
    )
    st.plotly_chart(figure, width="stretch")


def sample_table(bundle: AnalysisBundle, ids: list[str] | None = None, limit: int = 300) -> pd.DataFrame:
    scores = bundle.risk.scores
    if ids is not None:
        scores = scores.loc[[sid for sid in ids if sid in scores.index]]
    columns = [
        "label",
        "risk_score",
        "risk_category",
        "label_issue",
        "outlier",
        "duplicate",
        "exact_duplicate",
        "relationship_conflict",
        "suggested_label",
        "suggested_source",
        "suggested_confidence",
    ]
    available = [column for column in columns if column in scores.columns]
    frame = scores.sort_values("risk_score", ascending=False)[available].head(limit).reset_index()
    return frame


def table_with_risk(frame: pd.DataFrame, limit: int = 300) -> None:
    if frame.empty:
        st.caption("No samples match the current filters.")
        return
    st.dataframe(
        frame.head(limit),
        width="stretch",
        hide_index=True,
        column_config={
            "risk_score": st.column_config.NumberColumn("risk", format="%.3f"),
            "suggested_confidence": st.column_config.NumberColumn("suggestion conf.", format="%.3f"),
        },
    )
    if len(frame) > limit:
        st.caption(f"Showing the first {limit:,} of {len(frame):,} matching samples.")


def provenance_status(dataset: LoadedDataset) -> tuple[bool | None, str]:
    if not dataset.original_copy or not Path(dataset.original_copy).exists():
        return None, "Original upload archive/file is not available for hash verification."
    expected = st.session_state.get("ps_original_hash")
    if not expected:
        return None, "No original hash was recorded."
    actual = path_fingerprint(Path(dataset.original_copy))
    if actual == expected:
        if Path(dataset.original_copy).is_file():
            size = Path(dataset.original_copy).stat().st_size
            return True, f"Original file hash unchanged ({human_bytes(size)})."
        count = sum(1 for item in Path(dataset.original_copy).rglob("*") if item.is_file())
        return True, f"Original upload directory fingerprint unchanged ({count:,} files)."
    return False, "The stored original fingerprint has changed. Do not trust the provenance record."


def scatter_projection(bundle: AnalysisBundle, color_by: str = "risk_category") -> go.Figure | None:
    projection = bundle.relationships.projection.copy()
    if projection.empty or not {"x", "y"}.issubset(projection.columns):
        return None
    scores = bundle.risk.scores
    projection["risk_category"] = scores["risk_category"].reindex(projection.index)
    projection["risk_score"] = scores["risk_score"].reindex(projection.index)
    projection["label"] = scores["label"].reindex(projection.index).astype(str)
    projection["sample_id"] = projection.index.astype(str)
    if color_by == "risk_category":
        figure = px.scatter(
            projection,
            x="x",
            y="y",
            color="risk_category",
            color_discrete_map=RISK_COLORS,
            hover_data=["sample_id", "label", "risk_score"],
            title="Relationship projection (PCA)",
        )
    else:
        figure = px.scatter(
            projection,
            x="x",
            y="y",
            color="label",
            hover_data=["sample_id", "risk_category", "risk_score"],
            title="Relationship projection (PCA)",
        )
    figure.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=45, b=10),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend_title_text="",
    )
    return figure
