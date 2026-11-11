"""Sanitisation page."""

from __future__ import annotations

import html
from pathlib import Path

import streamlit as st

from backend.config import EXPORT_DIR
from backend.intelligence.sanitizer import (
    build_image_cleaned,
    build_tabular_cleaned,
    format_size,
    write_changes,
    write_sanitised_tabular,
)
from backend.reporting.exporter import dataframe_download_bytes
from backend.utils import timestamp_slug

from .. import state
from ..components import callout, provenance_status, render_no_analysis, render_no_dataset
from ..theme import hero


def render() -> None:
    hero(
        "Sanitisation",
        "Apply human decisions to a new copy. Uploaded originals remain byte-for-byte untouched.",
        kicker="PoisonShield · Mitigation",
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

    summary = store.summary()
    n_total = len(dataset.sample_ids)
    removed = sum(1 for decision in store.decisions.values() if decision.action == "remove")
    corrected = sum(1 for decision in store.decisions.values() if decision.action == "approve_correction")
    uncertain = sum(1 for decision in store.decisions.values() if decision.action == "uncertain")
    included = n_total - removed
    columns = st.columns(5)
    with columns[0]:
        st.metric("Original samples", f"{n_total:,}")
    with columns[1]:
        st.metric("Included in cleaned copy", f"{included:,}")
    with columns[2]:
        st.metric("Removed", f"{removed:,}")
    with columns[3]:
        st.metric("Corrected labels", f"{corrected:,}")
    with columns[4]:
        st.metric("Marked uncertain", f"{uncertain:,}")

    if not store.decisions:
        callout(
            "No review decisions have been recorded. Generating a cleaned copy now produces an "
            "identical copy with provenance metadata; it does not remove anything automatically.",
            "info",
        )
    else:
        callout(
            "Only explicit human decisions are applied. No sample is removed or relabelled because of "
            "a detector score alone. Review decisions are recorded with timestamps and reviewer names.",
            "ok",
        )

    status, message = provenance_status(dataset)
    if status is True:
        callout(html.escape(message), "ok")
    elif status is False:
        callout(html.escape(message), "danger")
    else:
        st.caption(message)

    st.markdown("#### Generate cleaned copy")
    if dataset.modality == "tabular":
        include_review_columns = st.checkbox(
            "Include PoisonShield sample IDs and review actions as extra columns",
            value=False,
            key="ps_san_include_cols",
        )
        output_format = st.selectbox("Output format", ["CSV", "XLSX"], key="ps_san_format")
        if st.button("Generate cleaned tabular dataset", type="primary", key="ps_generate_tabular"):
            try:
                result = build_tabular_cleaned(
                    bundle,
                    store,
                    include_review_columns=include_review_columns,
                )
                suffix = ".xlsx" if output_format == "XLSX" else ".csv"
                output_path = EXPORT_DIR / f"{bundle.session_id}_cleaned_{timestamp_slug()}{suffix}"
                write_sanitised_tabular(result.dataframe, output_path)
                result.output_path = output_path
                st.session_state["ps_sanitisation"] = result
                changes_path = EXPORT_DIR / f"{bundle.session_id}_review_changes_{timestamp_slug()}.csv"
                write_changes(result.changes, changes_path)
                st.session_state["ps_sanitisation_changes_path"] = changes_path
                st.success(f"Cleaned dataset written to {output_path.name}.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not generate the cleaned dataset: {exc}")
    else:
        if st.button("Generate cleaned image archive (ZIP)", type="primary", key="ps_generate_images"):
            try:
                result = build_image_cleaned(bundle, store)
                st.session_state["ps_sanitisation"] = result
                changes_path = EXPORT_DIR / f"{bundle.session_id}_review_changes_{timestamp_slug()}.csv"
                write_changes(result.changes, changes_path)
                st.session_state["ps_sanitisation_changes_path"] = changes_path
                st.success(f"Cleaned image archive written to {Path(result.output_path).name}.")
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not generate the cleaned image archive: {exc}")

    sanitisation = state.current_sanitisation()
    if sanitisation is None:
        st.caption("No cleaned copy has been generated in this session yet.")
    else:
        st.divider()
        st.markdown("#### Cleaned copy")
        columns = st.columns(5)
        with columns[0]:
            st.metric("Included", f"{sanitisation.n_included:,}")
        with columns[1]:
            st.metric("Removed", f"{sanitisation.n_removed:,}")
        with columns[2]:
            st.metric("Corrected", f"{sanitisation.n_corrected:,}")
        with columns[3]:
            st.metric("Uncertain", f"{sanitisation.n_uncertain:,}")
        with columns[4]:
            st.metric("File size", format_size(sanitisation.output_path))
        for note in sanitisation.notes:
            st.caption(note)
        download_cols = st.columns(3)
        with download_cols[0]:
            if sanitisation.modality == "tabular" and sanitisation.dataframe is not None and sanitisation.output_path:
                fmt = "xlsx" if sanitisation.output_path.suffix.lower() == ".xlsx" else "csv"
                payload, mime, extension = dataframe_download_bytes(sanitisation.dataframe, fmt)
                st.download_button(
                    f"Download cleaned dataset (.{extension})",
                    data=payload,
                    file_name=f"poisonshield_cleaned_{bundle.session_id}.{extension}",
                    mime=mime,
                    type="primary",
                )
            elif sanitisation.archive_path and sanitisation.archive_path.exists():
                st.download_button(
                    "Download cleaned image archive (.zip)",
                    data=sanitisation.archive_path.read_bytes(),
                    file_name=f"poisonshield_cleaned_images_{bundle.session_id}.zip",
                    mime="application/zip",
                    type="primary",
                )
        with download_cols[1]:
            changes_path = st.session_state.get("ps_sanitisation_changes_path")
            if changes_path and Path(changes_path).exists():
                st.download_button(
                    "Download review changes (CSV)",
                    data=Path(changes_path).read_bytes(),
                    file_name=Path(changes_path).name,
                    mime="text/csv",
                )
        with download_cols[2]:
            st.caption("The cleaned copy is evidence-safe: labels.csv / review columns record what changed.")

        st.markdown("#### Review changes")
        if sanitisation.changes.empty:
            st.caption("No review decisions were applied.")
        else:
            st.dataframe(sanitisation.changes, width="stretch", hide_index=True)

    st.divider()
    st.markdown("#### Why sanitisation is not automatic")
    st.markdown(
        "- PoisonShield risk scores are review priorities; automatically deleting every flagged sample "
        "would discard legitimate rare examples and weaken the model.\n"
        "- Cleanlab label quality is model-based evidence. A human must confirm a label change.\n"
        "- Duplicate removal is a choice: duplicates can be genuine repeated measurements.\n"
        "- The cleaned copy never overwrites the source. The original file hash is checked above."
    )
