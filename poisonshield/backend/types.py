"""Typed containers shared by the backend layers and the Streamlit UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class DatasetProfile:
    modality: str
    n_samples: int
    n_columns: int = 0
    label_column: str | None = None
    task: str = "unlabeled"
    class_distribution: dict[str, int] = field(default_factory=dict)
    missing_cells: int = 0
    missing_cell_pct: float = 0.0
    duplicate_rows: int = 0
    duplicate_sample_ids: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    all_missing_columns: list[str] = field(default_factory=list)
    high_missing_columns: dict[str, float] = field(default_factory=dict)
    high_cardinality_columns: dict[str, int] = field(default_factory=dict)
    id_like_columns: list[str] = field(default_factory=list)
    numeric_coercions: list[str] = field(default_factory=list)
    invalid_value_counts: dict[str, int] = field(default_factory=dict)
    column_stats: list[dict[str, Any]] = field(default_factory=list)
    image_stats: dict[str, Any] = field(default_factory=dict)
    creator: str = "PoisonShield"
    generated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "n_samples": self.n_samples,
            "n_columns": self.n_columns,
            "label_column": self.label_column,
            "task": self.task,
            "class_distribution": self.class_distribution,
            "missing_cells": self.missing_cells,
            "missing_cell_pct": self.missing_cell_pct,
            "duplicate_rows": self.duplicate_rows,
            "duplicate_sample_ids": self.duplicate_sample_ids,
            "numeric_columns": self.numeric_columns,
            "categorical_columns": self.categorical_columns,
            "datetime_columns": self.datetime_columns,
            "constant_columns": self.constant_columns,
            "all_missing_columns": self.all_missing_columns,
            "high_missing_columns": self.high_missing_columns,
            "high_cardinality_columns": self.high_cardinality_columns,
            "id_like_columns": self.id_like_columns,
            "numeric_coercions": self.numeric_coercions,
            "invalid_value_counts": self.invalid_value_counts,
            "column_stats": self.column_stats,
            "image_stats": self.image_stats,
            "creator": self.creator,
            "generated_at": self.generated_at,
        }


@dataclass
class ImageRecord:
    sample_id: str
    path: str
    rel_path: str
    label: str | None
    width: int | None = None
    height: int | None = None
    format: str | None = None
    size_bytes: int = 0
    readable: bool = True
    error: str | None = None
    sha256: str | None = None
    exact_duplicate_of: str | None = None
    brightness_std: float | None = None
    entropy: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "path": self.path,
            "rel_path": self.rel_path,
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "format": self.format,
            "size_bytes": self.size_bytes,
            "readable": self.readable,
            "error": self.error,
            "sha256": self.sha256,
            "exact_duplicate_of": self.exact_duplicate_of,
            "brightness_std": self.brightness_std,
            "entropy": self.entropy,
            "extra": self.extra,
        }


@dataclass
class LoadedDataset:
    modality: str
    source_name: str
    sample_ids: list[str]
    profile: DatasetProfile
    dataframe: pd.DataFrame | None = None
    images: list[ImageRecord] | None = None
    label_column: str | None = None
    task: str = "unlabeled"
    has_labels: bool = False
    ground_truth_column: str | None = None
    original_copy: Path | None = None
    extra_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def n_samples(self) -> int:
        return len(self.sample_ids)

    def labels(self) -> pd.Series:
        """Return the label column indexed by sample_id (string labels)."""
        if not self.has_labels:
            return pd.Series(dtype=object)
        if self.modality == "tabular" and self.dataframe is not None and self.label_column:
            values = self.dataframe[self.label_column].reset_index(drop=True)
        else:
            values = pd.Series([rec.label for rec in (self.images or [])], dtype=object)
        series = pd.Series(values.to_numpy(), index=self.sample_ids, dtype=object)
        series = series.where(series.notna(), None)
        return series

    def display_frame(self) -> pd.DataFrame:
        """UI table containing sample id, label and image metadata (or full rows)."""
        if self.modality == "tabular" and self.dataframe is not None:
            frame = self.dataframe.copy().reset_index(drop=True)
            frame.insert(0, "sample_id", self.sample_ids)
            return frame
        records = []
        for rec in self.images or []:
            row = rec.to_dict()
            row.pop("path", None)
            row.pop("sha256", None)
            records.append(row)
        return pd.DataFrame(records)


@dataclass
class FeatureResult:
    matrix: np.ndarray
    sample_ids: list[str]
    method: str
    meta: dict[str, Any] = field(default_factory=dict)
    preprocessor: Any | None = None
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)


@dataclass
class DetectorStatus:
    issue: str
    status: str  # completed | skipped | failed
    detail: str = ""
    n_flagged: int | None = None
    duration_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "issue": self.issue,
            "status": self.status,
            "detail": self.detail,
            "n_flagged": self.n_flagged,
            "duration_s": self.duration_s,
        }


@dataclass
class AuditResult:
    issues: pd.DataFrame
    summary: pd.DataFrame
    statuses: list[DetectorStatus]
    duration_s: float = 0.0
    dataset_level_findings: dict[str, Any] = field(default_factory=dict)
    missing_evidence: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    pred_probs: np.ndarray | None = None
    pred_probs_meta: dict[str, Any] = field(default_factory=dict)
    suggested_labels: pd.Series | None = None

    @property
    def completed_issue_names(self) -> list[str]:
        return [s.issue for s in self.statuses if s.status == "completed"]

    @property
    def all_failed(self) -> bool:
        return bool(self.statuses) and all(s.status == "failed" for s in self.statuses)


@dataclass
class RelationshipResult:
    sample_ids: list[str]
    k: int
    metric: str
    neighbor_indices: np.ndarray
    neighbor_distances: np.ndarray
    neighbor_similarities: np.ndarray
    per_sample: pd.DataFrame
    projection: pd.DataFrame
    suggested_labels: pd.Series
    suggested_confidence: pd.Series
    duration_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def neighbours_for(self, sample_id: str, limit: int | None = None) -> pd.DataFrame:
        """Build a readable neighbour table for one sample on demand."""

        positions = {sid: i for i, sid in enumerate(self.sample_ids)}
        if sample_id not in positions:
            return pd.DataFrame()
        row = positions[sample_id]
        indices = self.neighbor_indices[row]
        distances = self.neighbor_distances[row]
        similarities = self.neighbor_similarities[row]
        labels = self.per_sample.reindex(self.sample_ids)["label"].to_numpy()
        own_label = labels[row] if len(labels) > row else None
        records = []
        for rank, (index, distance, similarity) in enumerate(
            zip(indices, distances, similarities, strict=False), start=1
        ):
            if index < 0:
                continue
            neighbour_label = labels[index]
            records.append(
                {
                    "rank": rank,
                    "neighbor_id": self.sample_ids[index],
                    "neighbor_label": neighbour_label,
                    "distance": float(distance),
                    "similarity": float(similarity),
                    "label_match": (
                        None
                        if own_label is None or neighbour_label is None
                        else bool(str(own_label) == str(neighbour_label))
                    ),
                }
            )
            if limit and len(records) >= limit:
                break
        return pd.DataFrame(records)

    def edges(self, ids: list[str] | None = None, max_edges: int = 4000) -> pd.DataFrame:
        positions = {sid: i for i, sid in enumerate(self.sample_ids)}
        allowed = set(ids) if ids is not None else None
        labels = self.per_sample.reindex(self.sample_ids)["label"].to_numpy()
        records = []
        for row, sid in enumerate(self.sample_ids):
            if allowed is not None and sid not in allowed:
                continue
            for index, distance, similarity in zip(
                self.neighbor_indices[row],
                self.neighbor_distances[row],
                self.neighbor_similarities[row],
                strict=False,
            ):
                if index < 0:
                    continue
                target = self.sample_ids[index]
                if allowed is not None and target not in allowed:
                    continue
                if sid >= target:
                    continue
                records.append(
                    {
                        "source": sid,
                        "target": target,
                        "similarity": float(similarity),
                        "distance": float(distance),
                        "label_match": (
                            None
                            if labels[row] is None or labels[index] is None
                            else bool(str(labels[row]) == str(labels[index]))
                        ),
                    }
                )
                if len(records) >= max_edges:
                    return pd.DataFrame(records)
        return pd.DataFrame(records)


@dataclass
class Finding:
    kind: str
    severity: float
    text: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "severity": round(float(self.severity), 4),
            "text": self.text,
            "evidence": self.evidence,
        }


@dataclass
class Explanation:
    sample_id: str
    original_label: str | None
    suggested_label: str | None
    suggested_source: str | None
    findings: list[Finding]
    narrative: str
    risk_score: float
    risk_category: str
    missing_evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "original_label": self.original_label,
            "suggested_label": self.suggested_label,
            "suggested_source": self.suggested_source,
            "findings": [f.to_dict() for f in self.findings],
            "narrative": self.narrative,
            "risk_score": round(float(self.risk_score), 4),
            "risk_category": self.risk_category,
            "missing_evidence": self.missing_evidence,
        }


@dataclass
class RiskResult:
    scores: pd.DataFrame
    evidence: pd.DataFrame
    explanations: dict[str, Explanation]
    weights: dict[str, float]
    thresholds: dict[str, float]
    quality: dict[str, Any]
    missing_evidence: list[str] = field(default_factory=list)
    duration_s: float = 0.0


@dataclass
class AnalysisBundle:
    session_id: str
    created_at: str
    dataset: LoadedDataset
    features: FeatureResult
    audit: AuditResult
    relationships: RelationshipResult
    risk: RiskResult
    timings: dict[str, float] = field(default_factory=dict)
    memory: dict[str, float | None] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    config: dict[str, Any] = field(default_factory=dict)
    workspace: Path | None = None

    @property
    def sample_ids(self) -> list[str]:
        return self.dataset.sample_ids

    def flagged_count(self, column: str) -> int:
        if column not in self.risk.scores:
            return 0
        return int(self.risk.scores[column].fillna(False).astype(bool).sum())
