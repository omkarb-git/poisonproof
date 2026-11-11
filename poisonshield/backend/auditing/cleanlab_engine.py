"""Cleanlab Datalab integration for PoisonShield.

Verified against cleanlab 2.9.0 (Datalab API):

    Datalab(data, task=..., label_name=..., verbosity=0)
    find_issues(pred_probs=..., features=..., issue_types={...})
    get_issues() / get_issue_summary() / get_info()

Design notes
------------
* Every Cleanlab issue score is a *quality* score: values closer to 1 are
  better, values closer to 0 are more severe. The risk engine converts these
  into risk evidence as ``1 - quality``.
* The label issue manager internally uses Cleanlab's multiprocessing label
  filter. On Windows (and inside Streamlit) process spawning is unsafe, so we
  explicitly pass ``n_jobs=1`` via ``clean_learning_kwargs``.
* A combined Datalab pass is used for efficiency. Detectors that do not appear
  in the returned summary are reported as failed and are then re-run in
  isolation to capture a real error message. No detector is silently treated
  as "clean".
* Class-imbalance and non-IID are dataset-level findings. They are exposed in
  the report but deliberately excluded from per-sample risk scoring.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

from ..types import AuditResult, DetectorStatus
from ..utils import json_safe

ISSUE_COLUMN_MAP = {
    "label": ("is_label_issue", "label_score"),
    "outlier": ("is_outlier_issue", "outlier_score"),
    "near_duplicate": ("is_near_duplicate_issue", "near_duplicate_score"),
    "class_imbalance": ("is_class_imbalance_issue", "class_imbalance_score"),
    "non_iid": ("is_non_iid_issue", "non_iid_score"),
}

DATASET_LEVEL_ISSUES = {"class_imbalance", "non_iid"}
MIN_LABELLED_FOR_SUPERVISION = 4


class CleanlabAuditError(RuntimeError):
    """Raised when auditing cannot run at all."""


def _label_kwargs() -> dict[str, Any]:
    return {
        "clean_learning_kwargs": {
            "find_label_issues_kwargs": {"n_jobs": 1},
            "verbose": False,
        }
    }


def _issue_specs(issue_names: list[str]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for name in issue_names:
        specs[name] = _label_kwargs() if name == "label" else {}
    return specs


def _filter_info(info: Any, max_items: int = 20) -> dict[str, Any]:
    """Keep dataset-level info compact and JSON-serialisable."""

    if not isinstance(info, dict):
        return {}
    keep: dict[str, Any] = {}
    scalar_types = (str, int, float, bool, np.integer, np.floating, np.bool_)
    for key, value in info.items():
        if isinstance(value, scalar_types):
            keep[str(key)] = value.item() if hasattr(value, "item") else value
        elif isinstance(value, (list, tuple)) and len(value) <= max_items:
            if all(isinstance(v, scalar_types) for v in value):
                keep[str(key)] = [v.item() if hasattr(v, "item") else v for v in value]
        elif isinstance(value, dict) and len(value) <= max_items:
            simple = {
                str(k): (v.item() if hasattr(v, "item") else v)
                for k, v in value.items()
                if isinstance(v, scalar_types)
            }
            if simple:
                keep[str(key)] = simple
    return json_safe(keep)


class _RunOutput:
    def __init__(self) -> None:
        self.issues: pd.DataFrame | None = None
        self.summary: pd.DataFrame | None = None
        self.infos: dict[str, dict[str, Any]] = {}
        self.error: str | None = None
        self.issue_names: list[str] = []


def _run_datalab(
    data: pd.DataFrame,
    task: str,
    label_name: str | None,
    features: np.ndarray | None,
    pred_probs: np.ndarray | None,
    issue_names: list[str],
) -> _RunOutput:
    from cleanlab import Datalab

    output = _RunOutput()
    output.issue_names = issue_names
    try:
        lab = Datalab(
            data=data,
            task=task,
            label_name=label_name,
            verbosity=0,
        )
        lab.find_issues(
            features=features,
            pred_probs=pred_probs,
            issue_types=_issue_specs(issue_names),
        )
        output.issues = lab.get_issues().copy()
        output.summary = lab.get_issue_summary().copy()
        for issue in issue_names:
            try:
                output.infos[issue] = _filter_info(lab.get_info(issue))
            except Exception:  # noqa: BLE001 - info is diagnostic only
                output.infos[issue] = {}
    except Exception as exc:  # noqa: BLE001
        output.error = f"{type(exc).__name__}: {exc}"
    return output


def _diagnose_issue(
    data: pd.DataFrame,
    task: str,
    label_name: str | None,
    features: np.ndarray | None,
    pred_probs: np.ndarray | None,
    issue: str,
) -> str:
    """Run one detector in isolation to obtain its actual error message."""

    from cleanlab import Datalab

    try:
        lab = Datalab(data=data, task=task, label_name=label_name, verbosity=0)
        lab.find_issues(
            features=features,
            pred_probs=pred_probs,
            issue_types=_issue_specs([issue]),
        )
        if lab.get_issue_summary().empty:
            return "Detector ran but returned no summary."
        return "Detector completed in isolation but was absent from the combined pass."
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"


def _normalise_labels(labels: pd.Series, task: str) -> pd.Series:
    if task == "regression":
        return pd.to_numeric(labels, errors="coerce")
    return labels.astype(object).map(lambda v: None if v is None else str(v))


def _class_order(labels: pd.Series) -> list[str]:
    values = labels.dropna().astype(str)
    return sorted(set(values.tolist()))


def _align_pred_probs(
    pred_probs: np.ndarray | None,
    pred_class_order: list[str] | None,
    expected_order: list[str],
) -> np.ndarray | None:
    if pred_probs is None:
        return None
    matrix = np.asarray(pred_probs, dtype=float)
    if matrix.ndim == 1:
        raise ValueError("Classification predicted probabilities must be a 2D array.")
    if matrix.shape[1] != len(expected_order):
        raise ValueError(
            f"pred_probs has {matrix.shape[1]} columns but the labelled data has "
            f"{len(expected_order)} classes."
        )
    if pred_class_order is None:
        return matrix
    if list(pred_class_order) == list(expected_order):
        return matrix
    if set(pred_class_order) != set(expected_order):
        raise ValueError("pred_probs class ordering does not match the dataset classes.")
    order = [pred_class_order.index(name) for name in expected_order]
    return matrix[:, order]


def _extract_issue_frame(
    raw: pd.DataFrame,
    sample_ids: list[str],
    requested: list[str],
) -> pd.DataFrame:
    frame = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))
    for issue in requested:
        is_col, score_col = ISSUE_COLUMN_MAP[issue]
        if is_col in raw.columns:
            values = raw[is_col].to_numpy()
            frame[f"is_{issue}_issue"] = pd.array(values, dtype="boolean")
        if score_col in raw.columns:
            frame[score_col] = pd.to_numeric(raw[score_col], errors="coerce").to_numpy()
    return frame


def _merge_issue_frames(base: pd.DataFrame, addition: pd.DataFrame) -> pd.DataFrame:
    for column in addition.columns:
        if column not in base.columns:
            base[column] = pd.NA
        add = addition[column]
        if str(base[column].dtype) == "boolean":
            base[column] = base[column].astype("boolean")
            base.loc[add.index, column] = add.astype("boolean")
        else:
            base.loc[add.index, column] = add.reindex(base.index)
    return base


def _summarise_dataset_findings(
    summary: pd.DataFrame,
    infos: dict[str, dict[str, Any]],
    requested: list[str],
) -> dict[str, Any]:
    findings: dict[str, Any] = {}
    if summary is None:
        return findings
    for issue in requested:
        if issue not in DATASET_LEVEL_ISSUES and issue not in {"outlier", "near_duplicate"}:
            continue
        row = summary.loc[summary["issue_type"] == issue]
        if row.empty:
            continue
        record = row.iloc[0].to_dict()
        record["info"] = infos.get(issue, {})
        findings[issue] = json_safe(record)
    return findings


def _statuses_for_run(
    output: _RunOutput,
    requested: list[str],
    elapsed: float,
    data_for_diagnosis: pd.DataFrame,
    task: str,
    label_name: str | None,
    features: np.ndarray | None,
    pred_probs: np.ndarray | None,
) -> list[DetectorStatus]:
    statuses: list[DetectorStatus] = []
    summary = output.summary
    for issue in requested:
        present = summary is not None and not summary.empty and bool(
            (summary["issue_type"] == issue).any()
        )
        if present:
            row = summary.loc[summary["issue_type"] == issue].iloc[0]
            statuses.append(
                DetectorStatus(
                    issue=issue,
                    status="completed",
                    detail=(
                        f"Cleanlab Datalab completed this detector "
                        f"(dataset quality score {float(row.get('score', np.nan)):.3f})."
                    ),
                    n_flagged=int(row.get("num_issues", 0)),
                    duration_s=0.0,
                )
            )
        else:
            detail = output.error or _diagnose_issue(
                data_for_diagnosis,
                task,
                label_name,
                features,
                pred_probs,
                issue,
            )
            statuses.append(
                DetectorStatus(
                    issue=issue,
                    status="failed",
                    detail=detail,
                    n_flagged=None,
                    duration_s=0.0,
                )
            )
    if statuses and elapsed:
        share = elapsed / max(len([s for s in statuses if s.status == "completed"]), 1)
        for status in statuses:
            if status.status == "completed":
                status.duration_s = share
    return statuses


def run_cleanlab_audit(
    sample_ids: list[str],
    features: np.ndarray | None,
    labels: pd.Series | None,
    task: str,
    pred_probs: np.ndarray | None = None,
    pred_class_order: list[str] | None = None,
    progress: Any = None,
) -> AuditResult:
    """Run all applicable Cleanlab detectors and normalise their output."""

    from cleanlab import __version__ as cleanlab_version

    started = time.perf_counter()
    warnings: list[str] = []
    missing: list[str] = []
    statuses: list[DetectorStatus] = []
    all_issues = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))
    dataset_findings: dict[str, Any] = {}
    suggested_predicted = pd.Series(index=pd.Index(sample_ids), dtype=object)

    if not sample_ids:
        raise CleanlabAuditError("No samples were supplied for auditing.")
    if features is not None and len(features) != len(sample_ids):
        raise CleanlabAuditError(
            f"Feature rows ({len(features)}) do not match sample ids ({len(sample_ids)})."
        )

    normalised_labels = _normalise_labels(labels, task) if labels is not None else None
    labelled_mask = normalised_labels.notna().to_numpy() if normalised_labels is not None else np.zeros(len(sample_ids), bool)
    n_labelled = int(labelled_mask.sum())
    has_supervision = task in {"classification", "regression"} and n_labelled >= MIN_LABELLED_FOR_SUPERVISION
    feature_only_names = ["outlier", "near_duplicate"]
    output_label: _RunOutput | None = None
    # With a fully labelled feature dataset, the supervised pass can run the
    # feature-quality detectors as well. This avoids embedding/KNN work twice.
    combined = bool(has_supervision and n_labelled == len(sample_ids) and features is not None)

    # ---- feature-only pass (all analyzable samples) ----------------------
    feature_data = pd.DataFrame({"sample_id": sample_ids})
    feature_task = "classification"
    output_full: _RunOutput | None = None
    if features is not None and not combined:
        if progress:
            progress(0.05, "Running Cleanlab feature-quality detectors...")
        output_full = _run_datalab(
            data=feature_data,
            task=feature_task,
            label_name=None,
            features=features,
            pred_probs=None,
            issue_names=feature_only_names,
        )
        if output_full.error:
            warnings.append(f"Cleanlab feature-only pass: {output_full.error}")
        if output_full.issues is not None:
            all_issues = _merge_issue_frames(
                all_issues,
                _extract_issue_frame(output_full.issues, sample_ids, feature_only_names),
            )
        statuses.extend(
            _statuses_for_run(
                output_full,
                feature_only_names,
                time.perf_counter() - started,
                feature_data,
                feature_task,
                None,
                features,
                None,
            )
        )
        dataset_findings.update(
            _summarise_dataset_findings(
                output_full.summary if output_full.summary is not None else pd.DataFrame(),
                output_full.infos,
                feature_only_names,
            )
        )
    elif features is None:
        for issue in feature_only_names:
            statuses.append(
                DetectorStatus(
                    issue=issue,
                    status="skipped",
                    detail="Feature representation unavailable.",
                )
            )
        missing.extend(feature_only_names)

    # ---- supervised pass (labelled subset only) --------------------------
    supervised_names: list[str] = []
    if has_supervision:
        supervised_names = ["label", "non_iid"]
        if combined:
            supervised_names = ["label", "outlier", "near_duplicate", "non_iid"]

        labelled_values = normalised_labels[labelled_mask]
        if task == "classification":
            unique_labels = labelled_values.dropna().astype(str).unique()
            if len(unique_labels) >= 2:
                supervised_names.append("class_imbalance")
            else:
                statuses.append(
                    DetectorStatus(
                        issue="class_imbalance",
                        status="skipped",
                        detail="Only one labelled class is present; imbalance is not applicable.",
                    )
                )
                missing.append("class_imbalance")
        subset_positions = np.flatnonzero(labelled_mask)
        subset_ids = [sample_ids[i] for i in subset_positions]
        subset_labels = normalised_labels.iloc[subset_positions].reset_index(drop=True)
        subset_features = features[subset_positions] if features is not None else None
        data = pd.DataFrame({"sample_id": subset_ids, "label": subset_labels.to_numpy()})
        label_name = "label"
        subset_probs = None
        if task == "classification":
            expected_order = _class_order(subset_labels)
            try:
                subset_probs = _align_pred_probs(
                    pred_probs[subset_positions] if pred_probs is not None else None,
                    pred_class_order,
                    expected_order,
                )
            except ValueError as exc:
                warnings.append(f"Predicted probabilities could not be aligned: {exc}")
                subset_probs = None
            if subset_probs is None:
                warnings.append(
                    "No usable out-of-sample probabilities were available; Cleanlab "
                    "will derive label issues from the feature representation where possible."
                )
        if progress:
            progress(
                0.35 if not combined else 0.12,
                "Running Cleanlab label and dataset-structure detectors..."
                if not combined
                else "Running Cleanlab detectors in one combined pass...",
            )
        output_label = _run_datalab(
            data=data,
            task=task,
            label_name=label_name,
            features=subset_features,
            pred_probs=subset_probs,
            issue_names=supervised_names,
        )
        if output_label.error:
            warnings.append(f"Cleanlab supervised pass: {output_label.error}")
        if output_label.issues is not None:
            all_issues = _merge_issue_frames(
                all_issues,
                _extract_issue_frame(output_label.issues, subset_ids, supervised_names),
            )
            if "label_score" in all_issues.columns and subset_probs is not None:
                predicted = np.asarray(subset_probs).argmax(axis=1)
                order = _class_order(subset_labels)
                labels_pred = [order[i] for i in predicted]
                suggested_predicted.loc[subset_ids] = labels_pred
        statuses.extend(
            _statuses_for_run(
                output_label,
                supervised_names,
                time.perf_counter() - started,
                data,
                task,
                label_name,
                subset_features,
                subset_probs,
            )
        )
        dataset_findings.update(
            _summarise_dataset_findings(
                output_label.summary if output_label.summary is not None else pd.DataFrame(),
                output_label.infos,
                supervised_names,
            )
        )
        if n_labelled < len(sample_ids):
            warnings.append(
                f"Label-error and dataset-structure checks ran on the {n_labelled:,} labelled "
                f"sample(s); {len(sample_ids) - n_labelled:,} unlabelled sample(s) received "
                "feature-only checks."
            )
        # The labelled pass already computed outliers/duplicates; if the
        # feature-only pass failed, fall back to those supervised results.
        if output_full is None or output_full.issues is None:
            if output_label.issues is not None:
                fallback = _extract_issue_frame(output_label.issues, subset_ids, feature_only_names)
                for column in fallback.columns:
                    if column in all_issues.columns:
                        all_issues = all_issues.drop(columns=column)
                all_issues = _merge_issue_frames(all_issues, fallback)
    else:
        for issue in ["label", "class_imbalance", "non_iid"]:
            statuses.append(
                DetectorStatus(
                    issue=issue,
                    status="skipped",
                    detail=(
                        "No usable labels were supplied; label-error and label-structure "
                        "checks are not applicable."
                    ),
                )
            )
        missing.extend(["label", "class_imbalance", "non_iid"])

    # ---- deterministic column order -------------------------------------
    ordered_columns: list[str] = []
    for issue in ["label", "outlier", "near_duplicate", "class_imbalance", "non_iid"]:
        ordered_columns.extend([f"is_{issue}_issue", f"{issue}_score"])
    for column in ordered_columns:
        if column not in all_issues.columns:
            if column.startswith("is_"):
                all_issues[column] = pd.Series(
                    pd.NA, index=all_issues.index, dtype="boolean"
                )
            else:
                all_issues[column] = pd.Series(
                    np.nan, index=all_issues.index, dtype="float64"
                )
    all_issues = all_issues[ordered_columns]
    all_issues["predicted_label"] = suggested_predicted

    # ---- summary ---------------------------------------------------------
    summaries: list[pd.DataFrame] = []
    for output in [output_full, output_label if has_supervision else None]:
        if output is not None and output.summary is not None:
            summaries.append(output.summary)
    summary = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame(
        columns=["issue_type", "score", "num_issues"]
    )
    if not summary.empty:
        summary = summary.drop_duplicates(subset=["issue_type"], keep="first").reset_index(drop=True)

    # A detector that is requested but missing must never be reported clean.
    failed = [s for s in statuses if s.status == "failed"]
    if statuses and all(s.status == "failed" for s in statuses):
        raise CleanlabAuditError(
            "All Cleanlab detectors failed; the dataset was not audited. "
            + " | ".join(s.detail for s in statuses)
        )
    if failed:
        warnings.append(
            f"{len(failed)} Cleanlab detector(s) failed and are excluded from risk scoring: "
            + ", ".join(s.issue for s in failed)
        )

    all_issues["predicted_label"] = all_issues["predicted_label"].astype(object).where(
        all_issues["predicted_label"].notna(), None
    )

    audit = AuditResult(
        issues=all_issues,
        summary=summary,
        statuses=statuses,
        duration_s=time.perf_counter() - started,
        dataset_level_findings=dataset_findings,
        missing_evidence=sorted(set(missing + [s.issue for s in failed])),
        warnings=warnings,
        pred_probs=pred_probs,
        pred_probs_meta={
            "available": pred_probs is not None,
            "class_order": pred_class_order,
        },
        suggested_labels=suggested_predicted if suggested_predicted.notna().any() else None,
    )
    audit.dataset_level_findings["cleanlab_version"] = cleanlab_version
    audit.dataset_level_findings["health_score"] = (
        float(summary["score"].mean()) if not summary.empty else None
    )
    if progress:
        progress(0.55, "Cleanlab audit complete.")
    return audit
