"""Relationship intelligence: neighbourhoods, similarity and label consistency.

All relationships are computed from the analysis feature representation with a
scalable KNN search; no full pairwise similarity matrix is materialised.
Relationship evidence is supporting evidence for manual review, never proof
that a label is wrong.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..types import RelationshipResult
from ..utils import ProgressCallback, make_progress, pretty_issue_name, shannon_entropy, timer


class RelationshipError(ValueError):
    """Raised when neighbourhoods cannot be constructed."""


def _similarity_from_distance(distance: np.ndarray, metric: str) -> np.ndarray:
    if metric == "cosine":
        return np.clip(1.0 - distance, 0.0, 1.0)
    return np.clip(1.0 / (1.0 + distance), 0.0, 1.0)


def _project_2d(matrix: np.ndarray, random_state: int) -> tuple[np.ndarray, float | None, str]:
    n = matrix.shape[0]
    if n == 0:
        return np.zeros((0, 2)), None, "none"
    if n == 1:
        return np.zeros((1, 2)), None, "single-sample"
    components = min(2, matrix.shape[1], n - 1 if n > 1 else 1)
    if components < 1:
        return np.zeros((n, 2)), None, "none"
    pca = PCA(n_components=components, random_state=random_state)
    coords = pca.fit_transform(matrix)
    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(n)])
    explained = float(pca.explained_variance_ratio_.sum()) if components >= 2 else None
    return coords.astype(np.float32), explained, "PCA"


def _projection_clusters(coords: np.ndarray) -> np.ndarray:
    n = coords.shape[0]
    if n < 6:
        return np.full(n, -1, dtype=int)
    try:
        scaled = StandardScaler().fit_transform(coords)
        eps = 0.45
        clusters = DBSCAN(eps=eps, min_samples=3).fit_predict(scaled)
        return clusters.astype(int)
    except Exception:  # noqa: BLE001 - clustering is a visual aid only
        return np.full(n, -1, dtype=int)


def build_relationships(
    matrix: np.ndarray,
    sample_ids: list[str],
    labels: pd.Series | None,
    modality: str,
    k: int = 15,
    agreement_threshold: float = 0.5,
    random_state: int = 42,
    progress: ProgressCallback | None = None,
) -> RelationshipResult:
    report = make_progress(progress)
    if matrix is None or matrix.size == 0:
        raise RelationshipError("A feature representation is required for relationship analysis.")
    if matrix.shape[0] != len(sample_ids):
        raise RelationshipError(
            f"Feature rows ({matrix.shape[0]}) do not match sample ids ({len(sample_ids)})."
        )
    matrix = np.ascontiguousarray(matrix, dtype=np.float32)
    n = len(sample_ids)
    warnings: list[str] = []
    effective_k = int(min(max(k, 0), max(n - 1, 0)))
    if effective_k < k:
        warnings.append(
            f"Requested k={k} neighbours but only {n - 1} other sample(s) exist; "
            f"using k={effective_k}."
        )
    metric = "cosine" if modality == "image" else "euclidean"

    report(0.58, f"Building {effective_k}-nearest-neighbour relationships...")
    started = timer()
    with started as clock:
        if effective_k > 0:
            nn = NearestNeighbors(
                n_neighbors=min(n, effective_k + 1),
                metric=metric,
                algorithm="auto",
                n_jobs=-1,
            )
            nn.fit(matrix)
            raw_distances, raw_indices = nn.kneighbors(matrix)
        else:
            raw_distances = np.zeros((n, 0), dtype=np.float32)
            raw_indices = np.zeros((n, 0), dtype=np.int64)

        indices = np.full((n, effective_k), -1, dtype=np.int32)
        distances = np.full((n, effective_k), np.nan, dtype=np.float32)
        if effective_k:
            # Fast path: scikit-learn normally returns the query point first.
            if raw_indices.shape[1] > 0 and np.array_equal(raw_indices[:, 0], np.arange(n)):
                indices[:, :] = raw_indices[:, 1:effective_k + 1]
                distances[:, :] = raw_distances[:, 1:effective_k + 1]
            else:
                for row in range(n):
                    mask = raw_indices[row] != row
                    chosen = raw_indices[row][mask][:effective_k]
                    chosen_distances = raw_distances[row][mask][:effective_k]
                    if len(chosen):
                        indices[row, :len(chosen)] = chosen
                        distances[row, :len(chosen)] = chosen_distances
        similarities = _similarity_from_distance(distances, metric)

        label_values: list[str | None] = []
        if labels is not None:
            aligned_labels = labels.reindex(sample_ids)
            for value in aligned_labels.to_numpy():
                label_values.append(
                    None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
                )
        else:
            label_values = [None] * n

        class_names = sorted({value for value in label_values if value is not None})
        class_lookup = {name: i for i, name in enumerate(class_names)}
        codes = np.array([class_lookup.get(value, -1) if value is not None else -1 for value in label_values], dtype=int)
        neighbour_codes = np.full((n, effective_k), -1, dtype=int)
        if effective_k:
            valid = indices >= 0
            flat_indices = indices[valid]
            codes_for_flat = codes[flat_indices]
            placement = np.full(indices.shape, -1, dtype=int)
            placement[valid] = codes_for_flat
            neighbour_codes = placement

        report(0.68, "Analysing neighbourhood label consistency...")
        agreement = np.full(n, np.nan, dtype=float)
        majority_label = np.full(n, None, dtype=object)
        majority_fraction = np.full(n, np.nan, dtype=float)
        entropy_values = np.full(n, np.nan, dtype=float)
        labelled_neighbour_count = np.zeros(n, dtype=int)
        conflict = np.zeros(n, dtype=bool)
        suggested = np.full(n, None, dtype=object)
        confidence = np.full(n, np.nan, dtype=float)
        mean_distance = np.full(n, np.nan, dtype=float)
        mean_similarity = np.full(n, np.nan, dtype=float)
        top_neighbour = np.full(n, None, dtype=object)
        top_similarity = np.full(n, np.nan, dtype=float)

        for row in range(n):
            valid = neighbour_codes[row] >= 0
            if effective_k == 0 or not valid.any():
                continue
            row_codes = neighbour_codes[row][valid]
            row_similarities = np.clip(similarities[row][valid], 1e-6, None)
            counts = np.bincount(row_codes, minlength=len(class_names))
            total = int(counts.sum())
            labelled_neighbour_count[row] = total
            if total == 0:
                continue

            probabilities = counts / total
            entropy_values[row] = shannon_entropy(probabilities)
            top_class = int(np.argmax(counts))
            majority_label[row] = class_names[top_class]
            majority_fraction[row] = float(counts[top_class] / total)

            weights = np.bincount(
                row_codes,
                weights=row_similarities,
                minlength=len(class_names),
            )
            if weights.sum() > 0:
                suggested[row] = class_names[int(np.argmax(weights))]
                confidence[row] = float(weights.max() / weights.sum())

            own_code = codes[row]
            if own_code >= 0:
                own_count = int(counts[own_code])
                agreement[row] = own_count / total
                if (
                    total >= 3
                    and agreement[row] < agreement_threshold
                    and top_class != own_code
                    and counts[top_class] >= 2
                ):
                    conflict[row] = True

            row_distances = distances[row][valid]
            finite = row_distances[np.isfinite(row_distances)]
            if finite.size:
                mean_distance[row] = float(finite.mean())
            row_similarities = similarities[row][valid]
            finite_similarity = row_similarities[np.isfinite(row_similarities)]
            if finite_similarity.size:
                mean_similarity[row] = float(finite_similarity.mean())
            first = int(np.flatnonzero(indices[row] >= 0)[0]) if (indices[row] >= 0).any() else None
            if first is not None:
                top_neighbour[row] = sample_ids[indices[row, first]]
                top_similarity[row] = float(similarities[row, first])

        isolation = pd.Series(mean_distance).rank(pct=True, method="average").to_numpy()
        if n <= 1:
            isolation = np.zeros(n, dtype=float)

        report(0.82, "Projecting the relationship space for visualisation...")
        coords, explained, projection_method = _project_2d(matrix, random_state=random_state)
        clusters = _projection_clusters(coords)

        per_sample = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "label": label_values,
                "mean_distance": mean_distance,
                "mean_similarity": mean_similarity,
                "top_neighbor_id": top_neighbour,
                "top_similarity": top_similarity,
                "neighbor_label_agreement": agreement,
                "majority_neighbor_label": majority_label,
                "majority_fraction": majority_fraction,
                "neighbor_label_entropy": entropy_values,
                "labelled_neighbors": labelled_neighbour_count,
                "conflicting_neighborhood": conflict,
                "isolation_percentile": isolation,
            }
        ).set_index("sample_id")

        projection = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "x": coords[:, 0] if coords.size else np.zeros(n),
                "y": coords[:, 1] if coords.size > 1 else np.zeros(n),
                "cluster": clusters,
            }
        ).set_index("sample_id")

        suggestions = pd.Series(suggested, index=sample_ids, dtype=object)
        suggestions = suggestions.where(suggestions.notna(), None)
        confidences = pd.Series(confidence, index=sample_ids, dtype=float)

    if labels is None or all(value is None for value in label_values):
        warnings.append(
            "No labels are available; relationship metrics cover similarity and isolation only."
        )
    if explained is not None and explained < 0.25:
        warnings.append(
            f"The 2-D {projection_method} projection explains only {explained:.1%} of feature "
            "variance; visual distances are approximate."
        )

    meta = {
        "projection_method": projection_method,
        "projection_explained_variance": explained,
        "metric": metric,
        "k": effective_k,
        "agreement_threshold": agreement_threshold,
        "clustered_samples": int((clusters >= 0).sum()),
        "noise_samples": int((clusters < 0).sum()),
        "note": (
            "Neighbourhood evidence is supporting evidence only. Visual or feature similarity "
            "does not by itself prove that a label is incorrect."
        ),
    }
    report(0.9, "Relationship analysis complete.")
    return RelationshipResult(
        sample_ids=sample_ids,
        k=effective_k,
        metric=metric,
        neighbor_indices=indices,
        neighbor_distances=distances,
        neighbor_similarities=similarities,
        per_sample=per_sample,
        projection=projection,
        suggested_labels=suggestions,
        suggested_confidence=confidences,
        duration_s=clock["seconds"],
        warnings=warnings,
        meta=meta,
    )
