"""Model-performance comparison on an untouched evaluation dataset.

Sanitisation can remove useful signal as well as noise. These helpers report
the measured delta between a model trained on the original labels and the same
model trained on the reviewed labels; they do not assume improvement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


@dataclass
class ModelComparisonResult:
    modality: str
    task: str
    n_train_original: int
    n_train_cleaned: int
    n_test: int
    original_metrics: dict[str, float | None]
    cleaned_metrics: dict[str, float | None]
    delta: dict[str, float | None]
    classes: list[str]
    notes: list[str] = field(default_factory=list)
    per_class: pd.DataFrame | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "task": self.task,
            "n_train_original": self.n_train_original,
            "n_train_cleaned": self.n_train_cleaned,
            "n_test": self.n_test,
            "original_metrics": self.original_metrics,
            "cleaned_metrics": self.cleaned_metrics,
            "delta": self.delta,
            "classes": self.classes,
            "notes": self.notes,
        }


def _encode_labels(
    train_original: pd.Series,
    train_cleaned: pd.Series,
    test: pd.Series,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    values = pd.concat([train_original, train_cleaned, test], ignore_index=True)
    classes = sorted({str(v) for v in values.dropna().tolist()})
    lookup = {name: index for index, name in enumerate(classes)}

    def encode(series: pd.Series) -> np.ndarray:
        return np.array([lookup[str(v)] if pd.notna(v) and str(v) in lookup else -1 for v in series], dtype=int)

    return encode(train_original), encode(train_cleaned), encode(test), classes


def _classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int]) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"accuracy": None, "macro_f1": None, "weighted_f1": None}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)),
    }


def _regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    if len(y_true) == 0:
        return {"mae": None, "rmse": None, "r2": None}
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else None,
    }


def compare_models(
    train_original: pd.DataFrame,
    train_cleaned: pd.DataFrame,
    test: pd.DataFrame,
    label_column: str,
    feature_transform,
    task: str,
    modality: str = "tabular",
    random_state: int = 42,
) -> ModelComparisonResult:
    """Fit the same classifier/regressor on original and cleaned training data.

    ``feature_transform`` must be a fitted transformer exposing ``transform``.
    The test set is never used for fitting or feature fitting.
    """

    notes: list[str] = []
    original = train_original.copy().reset_index(drop=True)
    cleaned = train_cleaned.copy().reset_index(drop=True)
    holdout = test.copy().reset_index(drop=True)

    for name, frame in (("original training", original), ("cleaned training", cleaned), ("evaluation", holdout)):
        if label_column not in frame.columns:
            raise ValueError(f"The {name} data does not contain the label column '{label_column}'.")

    holdout = holdout[holdout[label_column].notna()].reset_index(drop=True)
    original = original[original[label_column].notna()].reset_index(drop=True)
    cleaned = cleaned[cleaned[label_column].notna()].reset_index(drop=True)
    if original.empty or cleaned.empty or holdout.empty:
        raise ValueError("Original training, cleaned training and evaluation data must all contain labelled rows.")

    train_matrix_original = np.asarray(feature_transform.transform(original))
    train_matrix_cleaned = np.asarray(feature_transform.transform(cleaned))
    test_matrix = np.asarray(feature_transform.transform(holdout))

    if task == "classification":
        y_orig, y_clean, y_test, classes = _encode_labels(
            original[label_column], cleaned[label_column], holdout[label_column]
        )
        valid_orig = y_orig >= 0
        valid_clean = y_clean >= 0
        valid_test = y_test >= 0
        if len(classes) < 2:
            raise ValueError("At least two classes are required for classification comparison.")

        original_model = LogisticRegression(max_iter=3000, random_state=random_state)
        cleaned_model = LogisticRegression(max_iter=3000, random_state=random_state)
        original_model.fit(train_matrix_original[valid_orig], y_orig[valid_orig])
        cleaned_model.fit(train_matrix_cleaned[valid_clean], y_clean[valid_clean])
        pred_original = original_model.predict(test_matrix[valid_test])
        pred_cleaned = cleaned_model.predict(test_matrix[valid_test])
        label_ids = list(range(len(classes)))
        original_metrics = _classification_metrics(y_test[valid_test], pred_original, label_ids)
        cleaned_metrics = _classification_metrics(y_test[valid_test], pred_cleaned, label_ids)

        rows = []
        confusions = []
        for index, class_name in enumerate(classes):
            mask = y_test[valid_test] == index
            support = int(mask.sum())
            rows.append(
                {
                    "class": class_name,
                    "support": support,
                    "original_recall": float((pred_original[mask] == index).mean()) if support else None,
                    "cleaned_recall": float((pred_cleaned[mask] == index).mean()) if support else None,
                }
            )
        confusion = confusion_matrix(y_test[valid_test], pred_cleaned, labels=label_ids)
        per_class = pd.DataFrame(rows)
        notes.append("Macro F1 treats every class equally; inspect per-class support for imbalance.")
    else:
        y_orig = pd.to_numeric(original[label_column], errors="coerce").to_numpy()
        y_clean = pd.to_numeric(cleaned[label_column], errors="coerce").to_numpy()
        y_test = pd.to_numeric(holdout[label_column], errors="coerce").to_numpy()
        valid_orig = np.isfinite(y_orig)
        valid_clean = np.isfinite(y_clean)
        valid_test = np.isfinite(y_test)
        original_model = Ridge(random_state=random_state)
        cleaned_model = Ridge(random_state=random_state)
        original_model.fit(train_matrix_original[valid_orig], y_orig[valid_orig])
        cleaned_model.fit(train_matrix_cleaned[valid_clean], y_clean[valid_clean])
        pred_original = original_model.predict(test_matrix[valid_test])
        pred_cleaned = cleaned_model.predict(test_matrix[valid_test])
        original_metrics = _regression_metrics(y_test[valid_test], pred_original)
        cleaned_metrics = _regression_metrics(y_test[valid_test], pred_cleaned)
        classes = []
        confusion = np.empty((0, 0))
        per_class = None
        notes.append("MAE/RMSE are lower-is-better; R² is higher-is-better.")

    delta: dict[str, float | None] = {}
    for key, value in cleaned_metrics.items():
        base = original_metrics.get(key)
        delta[key] = (
            float(value - base) if value is not None and base is not None else None
        )

    notes.extend(
        [
            "Sanitisation does not guarantee improved test performance; changes may remove genuine signal.",
            "The evaluation dataset is used only for testing and is never used to fit the feature representation or classifier.",
            "This comparison covers one model family and one split; it is not a universal claim.",
        ]
    )
    if len(cleaned) == len(original):
        notes.append(
            "No samples were removed or corrected, so the original and cleaned training sets are identical."
        )
    if task == "classification":
        n_classes_in_clean = len(set(y_clean[valid_clean].tolist()))
        if n_classes_in_clean < len(classes):
            notes.append(
                "The cleaned training set lacks at least one class present in the evaluation data."
            )

    return ModelComparisonResult(
        modality=modality,
        task=task,
        n_train_original=len(original),
        n_train_cleaned=len(cleaned),
        n_test=int(valid_test.sum()),
        original_metrics=original_metrics,
        cleaned_metrics=cleaned_metrics,
        delta=delta,
        classes=classes,
        notes=notes,
        per_class=per_class,
    )


def compare_embedding_models(
    train_ids: list[str],
    train_matrix: np.ndarray,
    original_labels: pd.Series,
    test_ids: list[str],
    test_matrix: np.ndarray,
    test_labels: pd.Series,
    task: str,
    modality: str,
    removed_ids: set[str] | None = None,
    corrected_labels: dict[str, str] | None = None,
    random_state: int = 42,
) -> ModelComparisonResult:
    """Compare a classifier/regressor trained on original vs reviewed embeddings.

    Used for image datasets (and any other modality with a fixed feature
    representation). The evaluation embeddings are never used for fitting.
    """

    removed_ids = set(removed_ids or set())
    corrected_labels = dict(corrected_labels or {})
    train_matrix = np.asarray(train_matrix, dtype=np.float64)
    test_matrix = np.asarray(test_matrix, dtype=np.float64)
    if train_matrix.shape[0] != len(train_ids):
        raise ValueError("Training embedding rows do not match the training sample IDs.")
    if test_matrix.shape[0] != len(test_ids):
        raise ValueError("Evaluation embedding rows do not match the evaluation sample IDs.")
    if train_matrix.shape[1] != test_matrix.shape[1]:
        raise ValueError(
            "Training and evaluation embeddings have different dimensions; "
            "use the same image extractor for both datasets."
        )

    original = original_labels.reindex(train_ids)
    cleaned_values: list[Any] = []
    cleaned_positions: list[int] = []
    for position, sample_id in enumerate(train_ids):
        if sample_id in removed_ids:
            continue
        label = corrected_labels.get(sample_id, original.iloc[position])
        if label is None or (isinstance(label, float) and np.isnan(label)):
            continue
        cleaned_values.append(str(label))
        cleaned_positions.append(position)
    cleaned = pd.Series(cleaned_values, index=[train_ids[i] for i in cleaned_positions])

    holdout = test_labels.reindex(test_ids)
    holdout_mask = holdout.notna().to_numpy()
    holdout = holdout[holdout_mask]
    test_matrix = test_matrix[holdout_mask]
    original_valid = original.notna().to_numpy()

    if task == "classification":
        y_orig, y_clean, y_test, classes = _encode_labels(original, cleaned, holdout)
        valid_orig = y_orig >= 0
        valid_clean = y_clean >= 0
        valid_test = y_test >= 0
        if len(classes) < 2 or valid_clean.sum() == 0:
            raise ValueError("At least two classes and reviewed training rows are required.")
        original_model = LogisticRegression(max_iter=3000, random_state=random_state)
        cleaned_model = LogisticRegression(max_iter=3000, random_state=random_state)
        original_model.fit(train_matrix[valid_orig], y_orig[valid_orig])
        cleaned_model.fit(train_matrix[cleaned_positions][valid_clean], y_clean[valid_clean])
        pred_original = original_model.predict(test_matrix[valid_test])
        pred_cleaned = cleaned_model.predict(test_matrix[valid_test])
        label_ids = list(range(len(classes)))
        original_metrics = _classification_metrics(y_test[valid_test], pred_original, label_ids)
        cleaned_metrics = _classification_metrics(y_test[valid_test], pred_cleaned, label_ids)
        rows = []
        for index, class_name in enumerate(classes):
            mask = y_test[valid_test] == index
            support = int(mask.sum())
            rows.append(
                {
                    "class": class_name,
                    "support": support,
                    "original_recall": float((pred_original[mask] == index).mean()) if support else None,
                    "cleaned_recall": float((pred_cleaned[mask] == index).mean()) if support else None,
                }
            )
        per_class = pd.DataFrame(rows)
        notes = [
            "A logistic classifier is fitted to the supplied embeddings; the embeddings come from the selected image extractor.",
            "Macro F1 treats every class equally; inspect per-class support for imbalance.",
        ]
    else:
        y_orig = pd.to_numeric(original, errors="coerce").to_numpy()
        y_clean = pd.to_numeric(cleaned, errors="coerce").to_numpy()
        y_test = pd.to_numeric(holdout, errors="coerce").to_numpy()
        valid_orig = np.isfinite(y_orig)
        valid_clean = np.isfinite(y_clean)
        valid_test = np.isfinite(y_test)
        original_model = Ridge(random_state=random_state)
        cleaned_model = Ridge(random_state=random_state)
        original_model.fit(train_matrix[valid_orig], y_orig[valid_orig])
        cleaned_model.fit(train_matrix[cleaned_positions][valid_clean], y_clean[valid_clean])
        pred_original = original_model.predict(test_matrix[valid_test])
        pred_cleaned = cleaned_model.predict(test_matrix[valid_test])
        original_metrics = _regression_metrics(y_test[valid_test], pred_original)
        cleaned_metrics = _regression_metrics(y_test[valid_test], pred_cleaned)
        classes = []
        per_class = None
        notes = ["MAE/RMSE are lower-is-better; R2 is higher-is-better."]

    delta: dict[str, float | None] = {}
    for key, value in cleaned_metrics.items():
        base = original_metrics.get(key)
        delta[key] = float(value - base) if value is not None and base is not None else None

    notes.extend(
        [
            "Sanitisation does not guarantee improved test performance; changes may remove genuine signal.",
            "The evaluation dataset is never used to fit the classifier or the image representation.",
            "This comparison covers one model family and one split; it is not a universal claim.",
        ]
    )
    if len(cleaned) == int(original.notna().sum()):
        notes.append(
            "No samples were removed or corrected, so the original and cleaned training sets are identical."
        )
    if task == "classification" and len(set(y_clean[valid_clean].tolist())) < len(classes):
        notes.append("The cleaned training set lacks at least one class present in the evaluation data.")

    return ModelComparisonResult(
        modality=modality,
        task=task,
        n_train_original=int(original_valid.sum()),
        n_train_cleaned=int(len(cleaned)),
        n_test=int(valid_test.sum()),
        original_metrics=original_metrics,
        cleaned_metrics=cleaned_metrics,
        delta=delta,
        classes=classes,
        notes=notes,
        per_class=per_class,
    )
