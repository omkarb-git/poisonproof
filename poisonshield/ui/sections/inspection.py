"""Dataset inspection page."""

from __future__ import annotations

import html
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from backend.utils import human_bytes, json_safe

from .. import state
from ..components import callout, fmt, render_image, render_no_dataset
from ..theme import hero


def render() -> None:
    hero(
        "Dataset Inspection",
        "Profiling, missingness, duplicates and image integrity before and after analysis.",
        kicker="PoisonShield · Inspection",
    )
    dataset = state.current_dataset()
    if dataset is None:
        render_no_dataset()
        return
    st.markdown(
        f"### {html.escape(dataset.source_name)}  \n"
        f"<span class='ps-muted'>{dataset.modality.title()} · {dataset.task.replace('_', ' ')} · "
        f"{dataset.profile.n_samples:,} samples</span>",
        unsafe_allow_html=True,
    )
    if dataset.modality == "tabular":
        _render_tabular(dataset)
    else:
        _render_images(dataset)
    if dataset.notes:
        st.divider()
        st.markdown("#### Ingestion notes")
        for note in dataset.notes:
            st.markdown(f"- {html.escape(str(note))}")


def _render_tabular(dataset) -> None:
    frame = dataset.dataframe
    assert frame is not None
    profile = dataset.profile
    tabs = st.tabs(["Summary", "Columns", "Missing & invalid", "Duplicates"])

    with tabs[0]:
        columns = st.columns(6)
        with columns[0]:
            st.metric("Rows", f"{profile.n_samples:,}")
        with columns[1]:
            st.metric("Columns", f"{profile.n_columns:,}")
        with columns[2]:
            st.metric("Missing cells", f"{profile.missing_cells:,}")
        with columns[3]:
            st.metric("Missing %", fmt(profile.missing_cell_pct, 2))
        with columns[4]:
            st.metric("Exact extra rows", f"{profile.duplicate_rows:,}")
        with columns[5]:
            st.metric("Classes", f"{len(profile.class_distribution):,}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Class distribution")
            if profile.class_distribution:
                class_frame = pd.DataFrame(
                    {"label": list(profile.class_distribution.keys()), "count": list(profile.class_distribution.values())}
                )
                figure = px.bar(class_frame, x="label", y="count", color="label", text="count")
                figure.update_layout(
                    showlegend=False,
                    height=320,
                    margin=dict(l=10, r=10, t=20, b=10),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(figure, width="stretch")
            else:
                st.caption("No label column was configured.")
        with col2:
            st.markdown("#### Column roles")
            role_rows = [
                {"column": item.get("column"), "role": item.get("role"), "dtype": item.get("dtype")}
                for item in profile.column_stats
            ]
            if role_rows:
                st.dataframe(pd.DataFrame(role_rows), width="stretch", hide_index=True)
            else:
                st.caption("No column statistics were recorded.")
        with st.expander("Raw profile JSON", expanded=False):
            st.json(json_safe(profile.to_dict()))

    with tabs[1]:
        stats = pd.DataFrame(profile.column_stats)
        if stats.empty:
            st.caption("No per-column statistics are available.")
            return
        st.dataframe(stats, width="stretch", hide_index=True)
        groups = st.columns(4)
        with groups[0]:
            st.markdown("**Identifier-like**")
            cols = profile.id_like_columns
            st.markdown(", ".join(f"`{c}`" for c in cols) if cols else "*None*")
        with groups[1]:
            st.markdown("**Constant**")
            cols = profile.constant_columns
            st.markdown(", ".join(f"`{c}`" for c in cols) if cols else "*None*")
        with groups[2]:
            st.markdown("**Entirely missing**")
            cols = profile.all_missing_columns
            st.markdown(", ".join(f"`{c}`" for c in cols) if cols else "*None*")
        with groups[3]:
            st.markdown("**High cardinality**")
            cols = list(profile.high_cardinality_columns.keys()) if isinstance(profile.high_cardinality_columns, dict) else (profile.high_cardinality_columns or [])
            st.markdown(", ".join(f"`{c}`" for c in cols) if cols else "*None*")

    with tabs[2]:
        missing = pd.DataFrame(
            {
                "column": frame.columns,
                "missing": [int(frame[column].isna().sum()) for column in frame.columns],
                "missing_pct": [round(100 * float(frame[column].isna().mean()), 3) for column in frame.columns],
                "dtype": [str(frame[column].dtype) for column in frame.columns],
            }
        ).sort_values("missing_pct", ascending=False)
        st.markdown("#### Missing values by column")
        st.dataframe(missing, width="stretch", hide_index=True)
        if profile.high_missing_columns:
            callout(
                "Columns with at least 40% missing values: "
                + ", ".join(f"{key} ({value}%)" for key, value in profile.high_missing_columns.items()),
                "info",
            )
        st.markdown("#### Invalid or sentinel values")
        if profile.invalid_value_counts:
            st.dataframe(
                pd.DataFrame(
                    [{"column": key, "count": value} for key, value in profile.invalid_value_counts.items()]
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "These are counts of infinite numeric values or common text missing sentinels "
                "(for example 'NA', 'unknown') in otherwise non-numeric columns. The original values are retained."
            )
        else:
            st.caption("No invalid or sentinel values were counted.")
        if profile.numeric_coercions:
            st.info(
                "These text columns look numeric and are parsed only for feature analysis: "
                + ", ".join(profile.numeric_coercions)
            )

    with tabs[3]:
        duplicate_mask = frame.duplicated(keep=False)
        duplicates = frame[duplicate_mask].reset_index(drop=False).rename(columns={"index": "row_index"})
        if duplicates.empty:
            st.success("No exact duplicate rows were found across all columns.")
            return
        st.markdown(f"#### {len(duplicates):,} rows participate in an exact duplicate group")
        st.caption(
            "Exact duplicates are flagged independently of Cleanlab near-duplicates. They are not "
            "automatically poisoning, but they duplicate training weight."
        )
        st.dataframe(duplicates.head(300), width="stretch", hide_index=True)
        if len(duplicates) > 300:
            st.caption(f"Showing the first 300 of {len(duplicates):,} duplicate rows.")


def _render_images(dataset) -> None:
    stats = dataset.profile.image_stats or {}
    records = dataset.images or []
    tabs = st.tabs(["Summary", "Gallery", "Unreadable & low information", "Duplicates", "Mapping metadata"])

    with tabs[0]:
        columns = st.columns(6)
        with columns[0]:
            st.metric("Images", f"{dataset.profile.n_samples:,}")
        with columns[1]:
            st.metric("Readable", f"{stats.get('readable', 0):,}")
        with columns[2]:
            st.metric("Unreadable", f"{stats.get('unreadable', 0):,}")
        with columns[3]:
            st.metric("Total size", human_bytes(stats.get("total_bytes", 0)))
        with columns[4]:
            st.metric("Exact duplicates", f"{stats.get('exact_duplicates', 0):,}")
        with columns[5]:
            st.metric("Missing labels", f"{stats.get('missing_labels', 0):,}")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Formats")
            st.dataframe(
                pd.DataFrame(
                    [{"format": key, "count": value} for key, value in (stats.get("formats") or {}).items()]
                ),
                width="stretch",
                hide_index=True,
            )
        with col2:
            st.markdown("#### Dimensions")
            st.dataframe(
                pd.DataFrame(
                    [{"dimensions": key, "count": value} for key, value in (stats.get("dimensions") or {}).items()]
                ),
                width="stretch",
                hide_index=True,
            )
        if stats.get("mean_width") and stats.get("mean_height"):
            st.caption(
                f"Mean dimensions {fmt(stats.get('mean_width'), 0)} × {fmt(stats.get('mean_height'), 0)} px · "
                f"mean file size {human_bytes(stats.get('mean_bytes') or 0)}"
            )
        if dataset.profile.class_distribution:
            st.markdown("#### Class distribution")
            class_frame = pd.DataFrame(
                {
                    "label": list(dataset.profile.class_distribution.keys()),
                    "count": list(dataset.profile.class_distribution.values()),
                }
            )
            st.dataframe(class_frame, width="stretch", hide_index=True)
        if stats.get("format_mismatches"):
            st.warning(
                f"{stats.get('format_mismatches'):,} image(s) have a file extension that differs from the "
                "decoded container format. They remain usable, but the mismatch is recorded."
            )

    with tabs[1]:
        readable = [record for record in records if record.readable]
        if not readable:
            st.caption("No readable images are available.")
            return
        page_size = 12
        pages = max(1, (len(readable) + page_size - 1) // page_size)
        page = st.number_input("Gallery page", min_value=1, max_value=pages, value=1, step=1)
        start = (int(page) - 1) * page_size
        for index, record in enumerate(readable[start : start + page_size]):
            if index % 4 == 0:
                columns = st.columns(4)
            with columns[index % 4]:
                render_image(record, caption=f"{record.rel_path}\n{record.label or 'unlabelled'}", width=260)
        st.caption(f"Page {page} of {pages}. Showing {len(readable):,} readable images in total.")

    with tabs[2]:
        unreadable = [record for record in records if not record.readable]
        if unreadable:
            st.markdown(f"#### Unreadable images ({len(unreadable):,})")
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "sample_id": record.sample_id,
                            "rel_path": record.rel_path,
                            "error": record.error,
                            "size_bytes": record.size_bytes,
                        }
                        for record in unreadable
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.success("Every image file could be decoded within the pixel safety limit.")
        low_ids = set(stats.get("low_information_sample_ids") or [])
        low = [record for record in records if record.sample_id in low_ids]
        st.markdown(f"#### Low-information images ({len(low):,})")
        if low:
            st.caption(
                "Low brightness variance or entropy can indicate blank, corrupted or uninformative images. "
                "This is a screening signal, not proof of poisoning."
            )
            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "sample_id": record.sample_id,
                            "rel_path": record.rel_path,
                            "brightness_std": record.brightness_std,
                            "entropy": record.entropy,
                        }
                        for record in low
                    ]
                ),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("No low-information images were detected.")

    with tabs[3]:
        by_hash: dict[str, list] = defaultdict(list)
        for record in records:
            if record.sha256:
                by_hash[record.sha256].append(record)
        groups = [group for group in by_hash.values() if len(group) > 1]
        if not groups:
            st.success("No byte-identical image files were found.")
            return
        st.markdown(f"#### {len(groups):,} exact duplicate group(s)")
        rows = []
        for group in groups:
            for position, record in enumerate(group):
                rows.append(
                    {
                        "group_size": len(group),
                        "sample_id": record.sample_id,
                        "rel_path": record.rel_path,
                        "label": record.label,
                        "size_bytes": record.size_bytes,
                    }
                )
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        st.caption(
            "Groups are identified by SHA-256 file hashes. They are reported for redundancy review; "
            "duplicate files are not automatically removed."
        )

    with tabs[4]:
        if dataset.extra_columns:
            st.markdown("#### Mapping Metadata Columns")
            pills = " ".join(
                f'<span class="ps-badge ps-info" style="font-size: 0.82rem; padding: 4px 10px; margin-right: 6px; font-weight: 600;">📋 {html.escape(str(c))}</span>'
                for c in dataset.extra_columns
            )
            st.markdown(
                f'<div class="ps-card" style="margin-bottom: 14px; padding: 14px 18px;">'
                f'<div style="font-size: 0.76rem; font-weight: 700; color: var(--ps-muted); text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px;">Detected Mapping Columns</div>'
                f'<div>{pills}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            rows = []
            for record in records[:2000]:
                row = {"sample_id": record.sample_id, "rel_path": record.rel_path, "label": record.label}
                row.update({f"meta_{key}": value for key, value in record.extra.items()})
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            if len(records) > 2000:
                st.caption("Showing metadata for the first 2,000 images.")
        else:
            st.caption("No extra mapping columns were present.")
