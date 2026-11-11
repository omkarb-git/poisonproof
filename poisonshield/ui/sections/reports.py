"""Reports and exports page."""

from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from backend.reporting.exporter import archive_size_label, build_report_archive, dataframe_download_bytes
from backend.utils import human_bytes, json_safe

from .. import state
from ..components import callout, provenance_status, render_no_analysis, render_no_dataset
from ..theme import hero


def render() -> None:
    hero(
        "Reports & Export",
        "Every metric and evidence trail in one portable archive, with cleaned deliverables kept separate.",
        kicker="PoisonShield · Reporting",
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

    columns = st.columns(4)
    with columns[0]:
        st.metric("Risk rows", f"{len(bundle.risk.scores):,}")
    with columns[1]:
        st.metric("Explanations", f"{len(bundle.risk.explanations):,}")
    with columns[2]:
        st.metric("Review decisions", f"{len(store.decisions):,}")
    with columns[3]:
        st.metric("Detector completion", str(bundle.risk.quality.get("detector_completion", "0/0")))

    status, message = provenance_status(dataset)
    if status is True:
        callout(html.escape(message), "ok")
    elif status is False:
        callout(html.escape(message), "danger")
    else:
        st.caption(message)

    st.markdown("#### Audit report archive")
    st.caption(
        "The ZIP contains a self-contained HTML report, machine-readable JSON, risk and evidence CSVs, "
        "Cleanlab outputs, detector status, relationships, review history and the cleaned copy when generated."
    )
    if st.button("Generate audit report archive", type="primary", key="ps_generate_report"):
        try:
            sanitisation = state.current_sanitisation()
            path = build_report_archive(bundle, store, sanitisation)
            st.session_state["ps_report_path"] = str(path)
            st.success(f"Report archive generated: {path.name} ({archive_size_label(path)}).")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not generate the report: {exc}")

    report_path = st.session_state.get("ps_report_path")
    if report_path and Path(report_path).exists():
        path = Path(report_path)
        st.download_button(
            "Download audit report (ZIP)",
            data=path.read_bytes(),
            file_name=path.name,
            mime="application/zip",
            type="primary",
        )
        st.caption(f"{path.name} · {human_bytes(path.stat().st_size)}")
    else:
        st.caption("No report archive has been generated in this session yet.")

    st.divider()
    st.markdown("#### Individual data exports")
    downloads = [
        ("Risk scores", bundle.risk.scores, "risk_scores.csv"),
        ("Risk evidence", bundle.risk.evidence, "risk_evidence.csv"),
        ("Cleanlab issues", bundle.audit.issues, "cleanlab_issues.csv"),
        ("Cleanlab summary", bundle.audit.summary, "cleanlab_summary.csv"),
        ("Relationship metrics", bundle.relationships.per_sample, "relationships.csv"),
        ("Review changes", store.changes_frame(), "review_changes.csv"),
    ]
    grid = st.columns(3)
    for index, (label, frame, filename) in enumerate(downloads):
        with grid[index % 3]:
            st.download_button(
                f"Download {label.lower()} (CSV)",
                data=frame.to_csv(index=label in {"Risk scores", "Risk evidence", "Cleanlab issues", "Relationship metrics"}).encode("utf-8-sig"),
                file_name=filename,
                mime="text/csv",
                key=f"ps_download_{index}",
            )

    session_payload = {
        "session_id": bundle.session_id,
        "created_at": bundle.created_at,
        "dataset": {
            "source_name": dataset.source_name,
            "modality": dataset.modality,
            "task": dataset.task,
            "label_column": dataset.label_column,
        },
        "profile": json_safe(dataset.profile.to_dict()),
        "config": json_safe(bundle.config),
        "timings": bundle.timings,
        "memory": bundle.memory,
        "warnings": bundle.warnings,
        "detector_status": [status.to_dict() for status in bundle.audit.statuses],
        "risk_weights": bundle.risk.weights,
        "risk_thresholds": bundle.risk.thresholds,
        "quality": json_safe(bundle.risk.quality),
        "review_decisions": store.changes_frame().to_dict(orient="records"),
    }
    st.download_button(
        "Download session metadata (JSON)",
        data=json.dumps(session_payload, indent=2, ensure_ascii=False, default=str).encode("utf-8"),
        file_name=f"poisonshield_{bundle.session_id}_session.json",
        mime="application/json",
    )

    st.divider()
    st.markdown("#### Cleaned dataset")
    sanitisation = state.current_sanitisation()
    if sanitisation is None:
        st.caption(
            "No cleaned copy has been generated. Open Sanitisation to apply recorded review decisions to "
            "a new copy; the original file is never modified."
        )
    elif sanitisation.modality == "tabular" and sanitisation.dataframe is not None:
        fmt = "xlsx" if sanitisation.output_path and sanitisation.output_path.suffix.lower() == ".xlsx" else "csv"
        payload, mime, extension = dataframe_download_bytes(sanitisation.dataframe, fmt)
        st.download_button(
            f"Download cleaned tabular dataset (.{extension})",
            data=payload,
            file_name=f"poisonshield_cleaned_{bundle.session_id}.{extension}",
            mime=mime,
            type="primary",
        )
        st.caption(f"{sanitisation.n_included:,} included · {sanitisation.n_removed:,} removed · {sanitisation.n_corrected:,} corrected")
    elif sanitisation.archive_path and sanitisation.archive_path.exists():
        st.download_button(
            "Download cleaned image archive (ZIP)",
            data=sanitisation.archive_path.read_bytes(),
            file_name=sanitisation.archive_path.name,
            mime="application/zip",
            type="primary",
        )
        st.caption(f"{sanitisation.n_included:,} included · {sanitisation.n_removed:,} removed · {sanitisation.n_corrected:,} corrected")

    st.divider()
    st.markdown("#### Session artifacts on disk")
    workspace = Path(bundle.workspace) if bundle.workspace else None
    if workspace and workspace.exists():
        rows = [
            {"file": path.name, "size": human_bytes(path.stat().st_size)}
            for path in sorted(workspace.iterdir())
            if path.is_file()
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(f"Workspace: {workspace}")
    else:
        st.caption("No workspace is available for this session.")

    st.markdown("#### Responsible export")
    st.markdown(
        "- The report states detector failures and unavailable evidence explicitly.\n"
        "- Risk categories are review priorities, not accusations or proof of poisoning.\n"
        "- The cleaned copy is separate from the original and includes provenance metadata.\n"
        "- Review decisions and timestamps are preserved in the archive."
    )
