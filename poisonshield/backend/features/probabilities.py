"""Out-of-sample class probabilities used by Cleanlab's label detector.

Two methods are supported:
* cross-validated logistic regression for tabular representations;
* similarity-weighted KNN voting for image embeddings (and as a fallback).

Both are explicitly out-of-sample for labelled rows: the model never sees the
row it is scoring. This module produces evidence for auditing; it is not a
model-performance benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..utils import timer


@dataclass
class ProbabilityResult:
    pred_probs: np.ndarray | None
    class_order: list[str] | None
    meta: dict[str, Any]


def _knn_probabilities(
    matrix: np.ndarray,
    labels: np.ndarray,
    class_order: list[str],
    k: int,
    metric: str,
) -> np.ndarray:
    n = matrix.shape[0]
    encoded = np.full(n, -1, dtype=int)
    lookup = {name: i for i, name in enumerate(class_order)}
    for i, value in enumerate(labels):
        if value is not None and str(value) in lookup:
            encoded[i] = lookup[str(value)]
    labelled = encoded >= 0
    if labelled.sum() < 2:
        raise ValueError("At least two labelled samples are required for KNN probabilities.")

    query_k = min(n, max(k * 3, k + 5))
    nn = NearestNeighbors(n_neighbors=query_k, metric=metric, n_jobs=-1)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix)
    probs = np.full((n, len(class_order)), 1.0 / len(class_order), dtype=float)
    alpha = 0.5
    for row in range(n):
        candidates: list[int] = []
        weights: list[float] = []
        for distance, index in zip(distances[row], indices[row], strict=False):
            if index == row or not labelled[index]:
                continue
            similarity = 1.0 - float(distance) if metric == "cosine" else 1.0 / (1.0 + float(distance))
            candidates.append(int(encoded[index]))
            weights.append(max(similarity, 1e-6))
            if len(candidates) >= k:
                break
        counts = np.full(len(class_order), alpha, dtype=float)
        for cls, weight in zip(candidates, weights, strict=False):
            counts[cls] += weight
        probs[row] = counts / counts.sum()
    return probs


def compute_predicted_probabilities(
    matrix: np.ndarray,
    labels: pd.Series,
    sample_ids: list[str],
    task: str,
    modality: str,
    k: int = 15,
    random_state: int = 42,
) -> ProbabilityResult:
    """Return probabilities aligned to *sample_ids* and class order for Datalab."""

    if task != "classification":
        return ProbabilityResult(None, None, {"available": False, "reason": "Task is not classification."})
    if matrix is None or matrix.size == 0:
        return ProbabilityResult(None, None, {"available": False, "reason": "No feature matrix."})

    aligned = labels.reindex(sample_ids)
    labelled_mask = aligned.notna().to_numpy()
    labelled_values = aligned[labelled_mask].astype(str)
    class_order = sorted(set(labelled_values.tolist()))
    if len(class_order) < 2 or labelled_mask.sum() < 4:
        return ProbabilityResult(
            None,
            class_order,
            {"available": False, "reason": "Fewer than two labelled classes or too few labelled rows."},
        )

    lookup = {name: i for i, name in enumerate(class_order)}
    encoded = np.full(len(sample_ids), -1, dtype=int)
    for position, value in enumerate(aligned.to_numpy()):
        if value is not None and not (isinstance(value, float) and np.isnan(value)):
            encoded[position] = lookup.get(str(value), -1)

    meta: dict[str, Any] = {
        "available": True,
        "class_order": class_order,
        "n_labelled": int(labelled_mask.sum()),
        "n_classes": len(class_order),
    }
    method = "knn_cosine" if modality == "image" else "logistic_cv"
    if modality != "image" and (len(sample_ids) > 20_000 or len(class_order) > 50):
        method = "knn_euclidean"
    meta["method"] = method

    with timer() as clock:
        if method == "logistic_cv":
            try:
                positions = np.flatnonzero(labelled_mask)
                min_class = int(pd.Series(encoded[positions]).value_counts().min())
                n_splits = int(min(5, min_class))
                if n_splits < 2:
                    raise ValueError("A class has fewer than two samples.")
                cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
                scaler = StandardScaler()
                scaled = scaler.fit_transform(matrix)
                clf = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                )
                oof = cross_val_predict(
                    clf,
                    scaled[positions],
                    encoded[positions],
                    cv=cv,
                    method="predict_proba",
                )
                # cross_val_predict may drop classes absent from a training fold;
                # map its columns back to the global class index space.
                fold_classes = np.unique(encoded[positions])
                if not np.array_equal(fold_classes, np.arange(len(class_order))):
                    raise ValueError("Cross-validation did not cover every class.")
                probs = np.full((len(sample_ids), len(class_order)), 1.0 / len(class_order))
                probs[positions] = oof
                unlabelled = np.flatnonzero(~labelled_mask)
                if len(unlabelled):
                    clf.fit(scaled[positions], encoded[positions])
                    probs[unlabelled] = clf.predict_proba(scaled[unlabelled])
                meta["cv_folds"] = n_splits
            except Exception as exc:  # noqa: BLE001
                meta["logistic_error"] = f"{type(exc).__name__}: {exc}"
                method = "knn_euclidean"
                meta["method"] = method
                probs = _knn_probabilities(matrix, aligned.to_numpy(), class_order, k=k, metric="euclidean")
        if method in {"knn_cosine", "knn_euclidean"}:
            metric = "cosine" if method == "knn_cosine" else "euclidean"
            probs = _knn_probabilities(matrix, aligned.to_numpy(), class_order, k=k, metric=metric)
            meta["k_neighbors"] = k

    meta["duration_s"] = clock["seconds"]
    return ProbabilityResult(probs.astype(np.float32), class_order, meta)
