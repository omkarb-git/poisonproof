"""Self-contained audit report and download packaging."""

from __future__ import annotations

import html
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from ..config import EXPORT_DIR
from ..types import AnalysisBundle
from ..utils import format_duration, human_bytes, json_safe, timestamp_slug
from ..intelligence.reviewer import ReviewStore
from ..intelligence.sanitizer import SanitisationResult

CSS = """
:root { color-scheme: light; }
body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background:#f4f7fb; color:#16202f; }
.wrap { max-width: 1180px; margin: 0 auto; padding: 32px 24px 64px; }
header { background: linear-gradient(135deg,#07111f 0%,#0d2236 60%,#0b3b44 100%);
         color:#ecfeff; border-radius:18px; padding:28px 32px; margin-bottom:24px; }
header h1 { margin:0 0 6px; font-size:30px; letter-spacing:.06em; }
header p { margin:2px 0; color:#a5f3fc; }
.badge { display:inline-block; background:#0e7490; color:#ecfeff; padding:4px 10px; border-radius:999px;
         font-size:12px; letter-spacing:.04em; margin-top:10px; }
section { background:#fff; border:1px solid #dbe6f0; border-radius:14px; padding:20px 22px; margin:18px 0; }
h2 { margin-top:0; font-size:20px; color:#0e7490; letter-spacing:.02em; }
h3 { font-size:15px; color:#1f3b52; }
table { border-collapse: collapse; width:100%; font-size:13px; margin:10px 0 4px; }
th { background:#e6f6fa; color:#0f3d4a; text-align:left; padding:8px; border:1px solid #cfe6ee; }
td { padding:7px 8px; border:1px solid #e2edf3; vertical-align:top; }
tr:nth-child(even) td { background:#fbfdff; }
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
.metric { background:#f1f9fc; border:1px solid #d6ecf4; border-radius:12px; padding:14px; }
.metric .value { font-size:24px; font-weight:700; color:#0e7490; }
.metric .label { font-size:12px; color:#4b6475; text-transform:uppercase; letter-spacing:.06em; }
.callout { border-left:4px solid #f59e0b; background:#fffbeb; padding:12px 14px; border-radius:8px;
           color:#7c4a03; margin:10px 0; }
.danger { border-left-color:#ef4444; background:#fef2f2; color:#7f1d1d; }
.ok { border-left-color:#10b981; background:#ecfdf5; color:#065f46; }
small, .muted { color:#5b7385; }
pre { white-space:pre-wrap; background:#f7fafc; border:1px solid #e2edf3; padding:10px; border-radius:8px; }
footer { color:#5b7385; font-size:12px; margin-top:28px; }
"""


def _frame_html(frame: pd.DataFrame, max_rows: int = 200) -> str:
    if frame is None or frame.empty:
        return "<p class='muted'>No rows.</p>"
    view = frame.head(max_rows).copy()
    for column in view.columns:
        view[column] = view[column].map(
            lambda value: (
                ""
                if value is None or (isinstance(value, float) and pd.isna(value))
                else (f"{value:.4f}" if isinstance(value, float) else str(value))
            )
        )
    return view.to_html(index=False, classes="table", border=0, escape=True)


def _metric(label: str, value: str) -> str:
    return (
        "<div class='metric'>"
        f"<div class='value'>{html.escape(value)}</div>"
        f"<div class='label'>{html.escape(label)}</div>"
        "</div>"
    )


def _fmt(value: Any, digits: int = 1, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        return f"{float(value):.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return html.escape(str(value))


def build_report_html(
    bundle: AnalysisBundle,
    review_store: ReviewStore | None = None,
    sanitisation: SanitisationResult | None = None,
) -> str:
    quality = bundle.risk.quality
    profile = bundle.dataset.profile
    failed = [status for status in bundle.audit.statuses if status.status == "failed"]
    skipped = [status for status in bundle.audit.statuses if status.status == "skipped"]

    detector_rows = pd.DataFrame(
        [
            {
                "Detector": status.issue,
                "Status": status.status,
                "Flagged / issues": status.n_flagged if status.n_flagged is not None else "n/a",
                "Detail": status.detail,
            }
            for status in bundle.audit.statuses
        ]
    )
    summary_rows = bundle.audit.summary.copy()
    risk_table = (
        bundle.risk.scores.reset_index()[
            [
                "sample_id", "label", "suggested_label", "risk_score", "risk_category",
                "label_issue", "outlier", "duplicate", "relationship_conflict",
                "relationship_consistency", "recommended_action",
            ]
        ]
        .sort_values("risk_score", ascending=False)
        .head(300)
    )
    explanations = list(bundle.risk.explanations.values())
    explanation_rows = []
    for explanation in explanations[:250]:
        findings = "".join(
            f"<li><strong>{html.escape(finding.kind)}</strong>: {html.escape(finding.text)}</li>"
            for finding in explanation.findings
        )
        explanation_rows.append(
            "<div class='finding'>"
            f"<h3>{html.escape(explanation.sample_id)} &mdash; "
            f"{html.escape(explanation.risk_category)} risk "
            f"({_fmt(explanation.risk_score, 3)})</h3>"
            f"<p><strong>Original label:</strong> {html.escape(str(explanation.original_label))} "
            f"&nbsp; <strong>Suggested label:</strong> {html.escape(str(explanation.suggested_label or 'n/a'))}</p>"
            f"<p>{html.escape(explanation.narrative)}</p>"
            f"<ul>{findings}</ul>"
            "</div>"
        )

    class_distribution = pd.DataFrame(
        [{"label": key, "count": value} for key, value in profile.class_distribution.items()]
    )
    warnings_html = "".join(f"<li>{html.escape(str(item))}</li>" for item in bundle.warnings)
    missing_html = "".join(
        f"<li>{html.escape(str(item))}</li>" for item in bundle.risk.missing_evidence
    ) or "<li>None</li>"
    limitations = [
        "Risk scores prioritise manual review and are not proof of poisoning.",
        "Similarity and neighbourhood disagreement are supporting evidence only.",
        "Cleanlab detector availability depends on labels and feature inputs; failed detectors are listed explicitly.",
        "Image feature quality varies by domain and by the selected extractor.",
        "Class-imbalance and non-IID results are dataset-level and are excluded from per-sample risk.",
        "Sanitisation may remove useful signal; model-performance changes are not guaranteed.",
    ]
    limitations_html = "".join(f"<li>{html.escape(item)}</li>" for item in limitations)

    sanitisation_html = ""
    if sanitisation is not None:
        sanitisation_html = f"""
        <section>
          <h2>Sanitisation</h2>
          <div class="metric-grid">
            {_metric("Original samples", f"{sanitisation.n_original:,}")}
            {_metric("Included in cleaned set", f"{sanitisation.n_included:,}")}
            {_metric("Removed", f"{sanitisation.n_removed:,}")}
            {_metric("Corrected labels", f"{sanitisation.n_corrected:,}")}
            {_metric("Marked uncertain", f"{sanitisation.n_uncertain:,}")}
          </div>
          <p class="muted">{html.escape(' '.join(sanitisation.notes))}</p>
          <h3>Review changes</h3>
          {_frame_html(sanitisation.changes, max_rows=500)}
        </section>
        """

    failed_html = ""
    if failed:
        failed_html = (
            "<div class='callout danger'><strong>Detector failures:</strong> "
            + ", ".join(html.escape(status.issue) for status in failed)
            + ". These detectors are excluded from per-sample scoring; the dataset must not be "
            "interpreted as clean for those issue types.</div>"
        )
    skipped_html = ""
    if skipped:
        skipped_html = (
            "<div class='callout'><strong>Detectors not applicable:</strong> "
            + ", ".join(html.escape(status.issue) for status in skipped)
            + ".</div>"
        )

    title = f"PoisonShield Audit Report — {bundle.dataset.source_name}"
    body = f"""
    <div class="wrap">
      <header>
        <h1>POISONSHIELD</h1>
        <p>Secure the Data. Protect the Model.</p>
        <p>{html.escape(bundle.dataset.source_name)} &middot; session {html.escape(bundle.session_id)}</p>
        <span class="badge">{html.escape(bundle.created_at)} &middot; {html.escape(bundle.dataset.modality.upper())}</span>
      </header>

      {failed_html}
      {skipped_html}

      <section>
        <h2>Executive summary</h2>
        <div class="metric-grid">
          {_metric("Total samples", f"{profile.n_samples:,}")}
          {_metric("Analysed samples", f"{quality.get('n_assessed', 0):,}")}
          {_metric("Potential label issues", f"{quality.get('n_label_issues', 0):,}")}
          {_metric("Outliers", f"{quality.get('n_outliers', 0):,}")}
          {_metric("Near duplicates", f"{quality.get('n_near_duplicates', 0):,}")}
          {_metric("Requiring review", f"{quality.get('n_requiring_review', 0):,}")}
          {_metric("Quality index", _fmt(quality.get("quality_index"), 1), )}
          {_metric("Detector completion", str(quality.get("detector_completion", "0/0")))}
        </div>
        <div class="callout">
          The quality index is <strong>100 &times; (1 - mean review risk)</strong> and is a screening
          summary, not a model-accuracy metric. Evidence coverage: {_fmt(quality.get("mean_evidence_coverage"), 2)}.
        </div>
      </section>

      <section>
        <h2>Dataset</h2>
        <p><strong>Modality:</strong> {html.escape(bundle.dataset.modality)} &middot;
           <strong>Task:</strong> {html.escape(bundle.dataset.task)} &middot;
           <strong>Label column:</strong> {html.escape(str(bundle.dataset.label_column or "unlabelled"))}</p>
        <p><strong>Rows / images:</strong> {profile.n_samples:,} &middot;
           <strong>Columns:</strong> {profile.n_columns:,} &middot;
           <strong>Missing cells:</strong> {profile.missing_cells:,} ({_fmt(profile.missing_cell_pct, 2)}%)</p>
        <h3>Class distribution</h3>
        {_frame_html(class_distribution, max_rows=100)}
        {f"<h3>Image statistics</h3>{_frame_html(pd.DataFrame([profile.image_stats]), max_rows=5)}" if profile.modality == 'image' else ''}
      </section>

      <section>
        <h2>Detector status</h2>
        <p class="muted">A detector that did not complete is never treated as evidence that the dataset is clean.</p>
        {_frame_html(detector_rows, max_rows=50)}
        <h3>Cleanlab issue summary</h3>
        {_frame_html(summary_rows, max_rows=50)}
      </section>

      <section>
        <h2>Dataset-level findings</h2>
        <p><strong>Class imbalance quality score:</strong>
           {_fmt(quality.get("class_imbalance", {}).get("score"), 3)}
           &middot; <strong>Non-IID p-value / quality:</strong>
           {_fmt(quality.get("non_iid", {}).get("score"), 3)}</p>
        <p class="muted">These are dataset-level signals and are deliberately excluded from per-sample risk.</p>
      </section>

      <section>
        <h2>Processing</h2>
        <div class="metric-grid">
          {_metric("Total time", format_duration(bundle.timings.get("total")))}
          {_metric("Feature build", format_duration(bundle.timings.get("feature_build")))}
          {_metric("Cleanlab audit", format_duration(bundle.timings.get("cleanlab_audit")))}
          {_metric("Relationships", format_duration(bundle.timings.get("relationships")))}
          {_metric("Peak traced memory", _fmt(bundle.memory.get("tracemalloc_peak_mb"), 1, " MB"))}
        </div>
        <pre>{html.escape(json.dumps(json_safe(bundle.config.get("feature_meta", {})), indent=2)[:6000])}</pre>
      </section>

      <section>
        <h2>Risk ranking</h2>
        {_frame_html(risk_table, max_rows=300)}
      </section>

      <section>
        <h2>Flagged-sample explanations</h2>
        {''.join(explanation_rows) if explanation_rows else '<p class="muted">No samples were flagged.</p>'}
      </section>

      {sanitisation_html}

      <section>
        <h2>Warnings</h2>
        <ul>{warnings_html or '<li>None</li>'}</ul>
        <h3>Unavailable evidence</h3>
        <ul>{missing_html}</ul>
      </section>

      <section>
        <h2>Limitations and responsible use</h2>
        <ul>{limitations_html}</ul>
      </section>

      <footer>
        Generated by PoisonShield. The uploaded source dataset and original image files were not modified.
        Review decisions are recorded separately and sanitisation is applied only to exported copies.
      </footer>
    </div>
    """
    return f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>{body}</body></html>"


def build_report_archive(
    bundle: AnalysisBundle,
    review_store: ReviewStore | None = None,
    sanitisation: SanitisationResult | None = None,
) -> Path:
    """Create a ZIP containing an HTML report, data exports and provenance."""

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = EXPORT_DIR / f"{bundle.session_id}_audit_report_{timestamp_slug()}.zip"
    html_report = build_report_html(bundle, review_store, sanitisation)
    payload = {
        "session": bundle.session_id,
        "created_at": bundle.created_at,
        "dataset": {
            "source_name": bundle.dataset.source_name,
            "modality": bundle.dataset.modality,
            "task": bundle.dataset.task,
            "label_column": bundle.dataset.label_column,
            "n_samples": len(bundle.dataset.sample_ids),
        },
        "profile": bundle.dataset.profile.to_dict(),
        "quality": json_safe(bundle.risk.quality),
        "timings": bundle.timings,
        "memory": bundle.memory,
        "detector_status": [status.to_dict() for status in bundle.audit.statuses],
        "missing_evidence": bundle.risk.missing_evidence,
        "warnings": bundle.warnings,
        "risk_weights": bundle.risk.weights,
        "risk_thresholds": bundle.risk.thresholds,
        "config": json_safe(bundle.config),
        "explanations": {sid: exp.to_dict() for sid, exp in bundle.risk.explanations.items()},
        "sanitisation": sanitisation.summary() if sanitisation else None,
    }
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        archive.writestr("report.html", html_report.encode("utf-8"))
        archive.writestr(
            "report.json", json.dumps(json_safe(payload), indent=2, ensure_ascii=False).encode("utf-8")
        )
        archive.writestr(
            "risk_scores.csv",
            bundle.risk.scores.to_csv(index_label="sample_id").encode("utf-8-sig"),
        )
        archive.writestr(
            "risk_evidence.csv",
            bundle.risk.evidence.to_csv(index_label="sample_id").encode("utf-8-sig"),
        )
        archive.writestr(
            "cleanlab_issues.csv",
            bundle.audit.issues.to_csv(index_label="sample_id").encode("utf-8-sig"),
        )
        archive.writestr(
            "cleanlab_summary.csv",
            bundle.audit.summary.to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            "detector_status.csv",
            pd.DataFrame([s.to_dict() for s in bundle.audit.statuses]).to_csv(index=False).encode("utf-8-sig"),
        )
        archive.writestr(
            "relationships.csv",
            bundle.relationships.per_sample.to_csv(index_label="sample_id").encode("utf-8-sig"),
        )
        archive.writestr(
            "projection.csv",
            bundle.relationships.projection.to_csv(index_label="sample_id").encode("utf-8-sig"),
        )
        if review_store is not None:
            archive.writestr(
                "review_decisions.csv",
                review_store.changes_frame().to_csv(index=False).encode("utf-8-sig"),
            )
            archive.writestr(
                "review_history.jsonl",
                "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in review_store.history).encode("utf-8"),
            )
        if sanitisation is not None:
            if sanitisation.modality == "tabular" and sanitisation.dataframe is not None:
                archive.writestr(
                    "cleaned_dataset.csv",
                    sanitisation.dataframe.to_csv(index=False).encode("utf-8-sig"),
                )
            elif sanitisation.archive_path and sanitisation.archive_path.exists():
                archive.write(sanitisation.archive_path, arcname="cleaned_images.zip")
        readme = (
            "PoisonShield audit report\n"
            "=========================\n"
            f"Session: {bundle.session_id}\n"
            f"Created: {bundle.created_at}\n\n"
            "Open report.html in a browser. report.json contains the machine-readable summary.\n"
            "The original uploaded dataset was not modified.\n"
            "Risk scores prioritise manual review; they do not confirm poisoning.\n"
        )
        archive.writestr("README.txt", readme.encode("utf-8"))
    return archive_path


def dataframe_download_bytes(frame: pd.DataFrame, fmt: str = "csv") -> tuple[bytes, str, str]:
    if fmt == "xlsx":
        import io

        buffer = io.BytesIO()
        frame.to_excel(buffer, index=False, engine="openpyxl")
        return buffer.getvalue(), "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "xlsx"
    return frame.to_csv(index=False).encode("utf-8-sig"), "text/csv", "csv"


def archive_size_label(path: Path) -> str:
    return human_bytes(path.stat().st_size) if path.exists() else "n/a"
