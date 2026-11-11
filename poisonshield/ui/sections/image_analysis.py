"""Image analysis page."""

from __future__ import annotations

import html

import pandas as pd
import plotly.express as px
import streamlit as st

from backend.utils import format_duration, human_bytes

from .. import state
from ..components import (
    callout,
    fmt,
    render_detector_status,
    render_image,
    render_no_analysis,
    render_no_dataset,
    render_quality_banner,
    render_risk_distribution,
    risk_histogram,
    sample_table,
    table_with_risk,
)
from ..theme import hero


def render() -> None:
    hero(
        "Image Analysis",
        "Embedding quality, visual relationships and per-image poisoning risk evidence.",
        kicker="PoisonShield · Images",
    )
    dataset = state.current_dataset()
    if dataset is None:
        render_no_dataset()
        return
    if dataset.modality != "image":
        callout(
            "This page is for image datasets. Open <strong>Tabular Analysis</strong> for the current "
            "tabular dataset.",
            "info",
        )
        return
    bundle = state.current_bundle()
    if bundle is None:
        render_no_analysis(dataset)
        return

    render_quality_banner(bundle)
    meta = bundle.features.meta or {}
    extractor = meta.get("extractor", "unknown")
    callout(
        f"<strong>Feature extractor:</strong> {html.escape(str(meta.get('extractor_label', extractor)))} "
        f"({html.escape(str(extractor))}) on {html.escape(str(meta.get('device', 'unknown')))}. "
        f"Embedding dimension: {meta.get('embedding_dim', 'n/a')}. "
        f"Cache hits: {meta.get('cache_hits', 0):,}; misses: {meta.get('cache_misses', 0):,}.",
        "info",
    )
    if meta.get("fallback_used") or str(meta.get("requested_extractor")) not in {"auto", str(extractor)}:
        callout(
            "The requested extractor was not used; the actual extractor above is recorded in the "
            "session metadata and report.",
            "danger",
        )

    tabs = st.tabs(["Detectors", "Risk signals", "Gallery", "Integrity & duplicates"])

    with tabs[0]:
        st.markdown("#### Cleanlab Datalab status")
        render_detector_status(bundle)
        st.markdown("#### Issue summary")
        if bundle.audit.summary.empty:
            callout("Cleanlab returned no issue summary rows.", "danger")
        else:
            st.dataframe(bundle.audit.summary, width="stretch", hide_index=True)
        st.markdown("#### Extractor metadata")
        st.json(meta)
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

    with tabs[2]:
        records = {record.sample_id: record for record in (dataset.images or [])}
        scores = bundle.risk.scores
        readable = [
            record
            for record in (dataset.images or [])
            if record.readable and record.sample_id in scores.index
        ]
        readable.sort(key=lambda record: float(scores.loc[record.sample_id, "risk_score"]), reverse=True)
        page_size = 8
        pages = max(1, (len(readable) + page_size - 1) // page_size)
        page = st.number_input("Gallery page (sorted by risk)", min_value=1, max_value=pages, value=1, step=1)
        start = (int(page) - 1) * page_size
        for index, record in enumerate(readable[start : start + page_size]):
            if index % 4 == 0:
                columns = st.columns(4)
            row = scores.loc[record.sample_id]
            with columns[index % 4]:
                render_image(
                    record,
                    caption=(
                        f"{record.sample_id} · {record.label or 'unlabelled'}\n"
                        f"risk {fmt(row.get('risk_score'), 3)} · {row.get('risk_category')}"
                    ),
                    width=260,
                )
        st.caption(f"Page {page} of {pages}. Images are sorted by unified risk, highest first.")

        st.markdown("#### Highest-priority images")
        top_ids = [record.sample_id for record in readable[:12]]
        table_with_risk(sample_table(bundle, ids=top_ids, limit=12), limit=12)

    with tabs[3]:
        records = dataset.images or []
        stats = dataset.profile.image_stats or {}
        columns = st.columns(4)
        with columns[0]:
            st.metric("Unreadable", f"{stats.get('unreadable', 0):,}")
        with columns[1]:
            st.metric("Exact duplicates", f"{stats.get('exact_duplicates', 0):,}")
        with columns[2]:
            st.metric("Low information", f"{stats.get('low_information', 0):,}")
        with columns[3]:
            st.metric("Total bytes", human_bytes(stats.get("total_bytes", 0)))
        unreadable = [record for record in records if not record.readable]
        if unreadable:
            st.markdown("#### Unreadable images")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "sample_id": record.sample_id,
                            "rel_path": record.rel_path,
                            "error": record.error,
                        }
                        for record in unreadable
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "Unreadable files are retained as integrity findings. They are excluded from embedding-based "
                "detectors but remain part of the dataset and cleaned-export decisions."
            )
        duplicate_groups: dict[str, set[str]] = {}
        for record in records:
            if record.sha256:
                duplicate_groups.setdefault(record.sha256, set()).add(record.sample_id)
        duplicate_ids: set[str] = set()
        for group in duplicate_groups.values():
            if len(group) > 1:
                duplicate_ids.update(group)
        if duplicate_ids:
            st.markdown("#### Byte-identical image files")
            st.dataframe(
                pd.DataFrame(
                    [
                        {"sample_id": record.sample_id, "rel_path": record.rel_path, "duplicate_of": record.exact_duplicate_of}
                        for record in records
                        if record.sample_id in duplicate_ids or record.exact_duplicate_of
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        if bundle.warnings:
            st.markdown("#### Warnings")
            for warning in bundle.warnings:
                st.markdown(f"- {html.escape(str(warning))}")
