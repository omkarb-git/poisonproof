"""Detection evaluation against user-supplied ground truth.

Metrics are only computed where a real ground-truth column exists. The
evaluation page never invents labels or simulates poisoning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from ..types import AnalysisBundle, LoadedDataset

POSITIVE_VALUES = {
    "1", "true", "yes", "y", "t", "poisoned", "poison", "corrupt", "corrupted",
    "mislabeled", "mislabel", "noisy", "positive", "attack", "bad", "invalid",
}
NEGATIVE_VALUES = {
    "0", "false", "no", "n", "f", "clean", "normal", "valid", "negative", "good",
    "unpoisoned", "uncorrupted",
}

SIGNAL_LABELS = {
    "any_issue": "Any detector flag",
    "label_issue": "Cleanlab label issue",
    "outlier": "Cleanlab outlier",
    "duplicate": "Near duplicate",
    "relationship_conflict": "Neighbourhood conflict",
    "high_risk": "High or Critical risk",
    "risk_score": "Risk score at threshold",
}


@dataclass
class DetectionEvaluation:
    signal: str
    n_evaluated: int
    n_positive: int
    n_flagged: int
    precision: float | None
    recall: float | None
    f1: float | None
    accuracy: float | None
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    roc_auc: float | None = None
    average_precision: float | None = None
    notes: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal": self.signal,
            "n_evaluated": self.n_evaluated,
            "n_positive": self.n_positive,
            "n_flagged": self.n_flagged,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
            "roc_auc": self.roc_auc,
            "average_precision": self.average_precision,
            "notes": self.notes or [],
        }

    def confusion_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [[self.true_positive, self.false_negative], [self.false_positive, self.true_negative]],
            index=["Ground truth positive", "Ground truth negative"],
            columns=["Flagged", "Not flagged"],
        )


def available_ground_truth_columns(dataset: LoadedDataset) -> list[str]:
    if dataset.modality == "tabular" and dataset.dataframe is not None:
        return [c for c in dataset.dataframe.columns if c != dataset.label_column]
    extras: set[str] = set()
    for record in dataset.images or []:
        extras.update(record.extra.keys())
    synthetic = ["__exact_duplicate__", "__unreadable__"]
    return sorted(extras) + synthetic


def ground_truth_series(dataset: LoadedDataset, column: str) -> pd.Series:
    index = dataset.sample_ids
    if column == "__exact_duplicate__":
        records = {record.sample_id: record for record in (dataset.images or [])}
        hash_counts: dict[str, int] = {}
        for record in dataset.images or []:
            if record.sha256:
                hash_counts[record.sha256] = hash_counts.get(record.sha256, 0) + 1
        duplicate_hashes = {value for value, count in hash_counts.items() if count > 1}
        return pd.Series(
            [
                1
                if records.get(sid) and records[sid].sha256 in duplicate_hashes
                else 0
                for sid in index
            ],
            index=index,
            dtype=object,
        )
    if column == "__unreadable__":
        records = {record.sample_id: record for record in (dataset.images or [])}
        return pd.Series(
            [0 if records.get(sid) and records[sid].readable else 1 for sid in index],
            index=index,
            dtype=object,
        )
    if dataset.modality == "tabular" and dataset.dataframe is not None:
        if column not in dataset.dataframe.columns:
            raise ValueError(f"Ground-truth column '{column}' is not present in the table.")
        return pd.Series(
            dataset.dataframe[column].to_numpy(), index=index, dtype=object
        )
    records = {record.sample_id: record for record in (dataset.images or [])}
    values = [records.get(sid).extra.get(column) if records.get(sid) else None for sid in index]
    return pd.Series(values, index=index, dtype=object)


def binarize_truth(series: pd.Series) -> tuple[pd.Series, dict[str, Any]]:
    """Convert a real ground-truth column into nullable 0/1 labels."""

    values: list[int | None] = []
    unmapped: dict[str, int] = {}
    for value in series.to_numpy():
        if value is None or (isinstance(value, float) and np.isnan(value)):
            values.append(None)
            continue
        text = str(value).strip().lower()
        if text in POSITIVE_VALUES:
            values.append(1)
        elif text in NEGATIVE_VALUES:
            values.append(0)
        else:
            try:
                numeric = float(text)
                values.append(1 if numeric > 0.5 else 0)
            except ValueError:
                values.append(None)
                unmapped[text] = unmapped.get(text, 0) + 1
    result = pd.Series(values, index=series.index, dtype="Int64")
    info = {
        "n_mapped": int(result.notna().sum()),
        "n_unmapped": int(result.isna().sum()),
        "unmapped_examples": dict(sorted(unmapped.items(), key=lambda item: -item[1])[:10]),
    }
    if result.notna().sum() == 0:
        raise ValueError(
            "The selected ground-truth column could not be interpreted as binary "
            "(positive/negative). Expected values such as 0/1, true/false or poisoned/clean."
        )
    return result, info


def detection_predictions(
    bundle: AnalysisBundle,
    signal: str,
    threshold: float = 0.5,
) -> pd.Series:
    scores = bundle.risk.scores
    if signal == "any_issue":
        values = (
            scores["label_issue"]
            | scores["outlier"]
            | scores["duplicate"]
            | scores["exact_duplicate"]
            | scores["relationship_conflict"]
        )
    elif signal == "label_issue":
        values = scores["label_issue"]
    elif signal == "outlier":
        values = scores["outlier"]
    elif signal == "duplicate":
        values = scores["duplicate"] | scores["exact_duplicate"]
    elif signal == "relationship_conflict":
        values = scores["relationship_conflict"]
    elif signal == "high_risk":
        values = scores["risk_category"].isin(["High", "Critical"])
    elif signal == "risk_score":
        values = scores["risk_score"] >= float(threshold)
    else:
        raise ValueError(f"Unknown detection signal '{signal}'.")
    return values.fillna(False).astype(bool)


def evaluate_detection(
    bundle: AnalysisBundle,
    truth: pd.Series,
    signal: str,
    threshold: float = 0.5,
) -> DetectionEvaluation:
    predictions = detection_predictions(bundle, signal, threshold)
    aligned_truth = truth.reindex(predictions.index)
    mask = aligned_truth.notna()
    if mask.sum() == 0:
        raise ValueError("No labelled ground-truth samples overlap the analysed samples.")
    y_true = aligned_truth[mask].astype(int)
    y_pred = predictions[mask].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall) > 0
        else None
    )
    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)

    roc_auc = None
    average_precision = None
    notes: list[str] = []
    if signal in {"risk_score", "high_risk"} and y_true.nunique() == 2:
        try:
            from sklearn.metrics import average_precision_score, roc_auc_score

            continuous = bundle.risk.scores["risk_score"].reindex(y_true.index).astype(float)
            roc_auc = float(roc_auc_score(y_true, continuous))
            average_precision = float(average_precision_score(y_true, continuous))
        except Exception as exc:  # noqa: BLE001
            notes.append(f"AUC could not be computed: {exc}")
    if y_true.nunique() < 2:
        notes.append("Ground truth contains only one class in the overlap; recall/AUC are not meaningful.")
    if signal == "duplicate" and "is_near_duplicate_issue" not in bundle.audit.issues.columns:
        notes.append("Near-duplicate detector did not run, so no duplicate flags were available.")

    return DetectionEvaluation(
        signal=signal,
        n_evaluated=int(mask.sum()),
        n_positive=int((y_true == 1).sum()),
        n_flagged=int(y_pred.sum()),
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        roc_auc=roc_auc,
        average_precision=average_precision,
        notes=notes,
    )


def evaluate_all_signals(
    bundle: AnalysisBundle,
    truth: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for signal in SIGNAL_LABELS:
        try:
            result = evaluate_detection(bundle, truth, signal, threshold)
            rows.append(
                {
                    "Signal": SIGNAL_LABELS[signal],
                    "Evaluated": result.n_evaluated,
                    "Ground-truth positives": result.n_positive,
                    "Flagged": result.n_flagged,
                    "Precision": result.precision,
                    "Recall": result.recall,
                    "F1": result.f1,
                    "Accuracy": result.accuracy,
                    "ROC AUC": result.roc_auc,
                    "Notes": "; ".join(result.notes or []),
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "Signal": SIGNAL_LABELS[signal],
                    "Evaluated": 0,
                    "Ground-truth positives": 0,
                    "Flagged": 0,
                    "Precision": None,
                    "Recall": None,
                    "F1": None,
                    "Accuracy": None,
                    "ROC AUC": None,
                    "Notes": str(exc),
                }
            )
    return pd.DataFrame(rows)


def duplicate_accuracy_report(bundle: AnalysisBundle, truth: pd.Series) -> dict[str, Any]:
    """Compare near-duplicate flags with a user-supplied duplicate ground truth."""

    try:
        evaluation = evaluate_detection(bundle, truth, "duplicate")
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "reason": str(exc)}
    return {
        "available": True,
        "precision": evaluation.precision,
        "recall": evaluation.recall,
        "f1": evaluation.f1,
        "accuracy": evaluation.accuracy,
        "n_evaluated": evaluation.n_evaluated,
        "n_positive": evaluation.n_positive,
        "n_flagged": evaluation.n_flagged,
    }
