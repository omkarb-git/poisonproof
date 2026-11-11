"""Persist analysis artifacts under the PoisonShield workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import SESSIONS_DIR
from ..types import AnalysisBundle
from ..utils import json_safe, write_json


def _sample_table(bundle: AnalysisBundle) -> pd.DataFrame:
    dataset = bundle.dataset
    if dataset.modality == "tabular" and dataset.dataframe is not None:
        frame = dataset.dataframe.copy().reset_index(drop=True)
        frame.insert(0, "sample_id", dataset.sample_ids)
        return frame
    rows = []
    for record in dataset.images or []:
        rows.append(
            {
                "sample_id": record.sample_id,
                "rel_path": record.rel_path,
                "source_path": record.path,
                "label": record.label,
                "readable": record.readable,
                "error": record.error,
                "width": record.width,
                "height": record.height,
                "format": record.format,
                "size_bytes": record.size_bytes,
                "sha256": record.sha256,
                "exact_duplicate_of": record.exact_duplicate_of,
                "brightness_std": record.brightness_std,
                "entropy": record.entropy,
                **{f"meta_{key}": value for key, value in record.extra.items()},
            }
        )
    return pd.DataFrame(rows)


def save_session_artifacts(bundle: AnalysisBundle) -> Path:
    workspace = Path(bundle.workspace) if bundle.workspace else SESSIONS_DIR / bundle.session_id
    workspace.mkdir(parents=True, exist_ok=True)

    detector_status = pd.DataFrame([status.to_dict() for status in bundle.audit.statuses])
    explanations = {sid: exp.to_dict() for sid, exp in bundle.risk.explanations.items()}
    meta: dict[str, Any] = {
        "session_id": bundle.session_id,
        "created_at": bundle.created_at,
        "profile": bundle.dataset.profile.to_dict(),
        "dataset": {
            "source_name": bundle.dataset.source_name,
            "modality": bundle.dataset.modality,
            "task": bundle.dataset.task,
            "has_labels": bundle.dataset.has_labels,
            "label_column": bundle.dataset.label_column,
            "sample_ids": bundle.dataset.sample_ids,
            "notes": bundle.dataset.notes,
            "original_copy": str(bundle.dataset.original_copy) if bundle.dataset.original_copy else None,
        },
        "config": json_safe(bundle.config),
        "timings": bundle.timings,
        "memory": bundle.memory,
        "warnings": bundle.warnings,
        "detector_status": [status.to_dict() for status in bundle.audit.statuses],
        "audit_summary": json_safe(bundle.audit.summary.to_dict(orient="records")),
        "dataset_level_findings": bundle.audit.dataset_level_findings,
        "missing_evidence": bundle.risk.missing_evidence,
        "risk_weights": bundle.risk.weights,
        "risk_thresholds": bundle.risk.thresholds,
        "quality": bundle.risk.quality,
        "explanation_count": len(explanations),
        "limitations": [
            "Risk scores prioritise manual review; they do not confirm data poisoning.",
            "Relationship similarity is supporting evidence, not proof of label error.",
            "Cleanlab detector availability depends on labels and feature inputs.",
            "Class-imbalance and non-IID findings are dataset-level and are excluded from per-sample risk.",
            "Image feature quality varies by domain and extractor.",
        ],
    }
    write_json(workspace / "session.json", meta)
    write_json(workspace / "explanations.json", explanations)
    write_json(workspace / "feature_sample_ids.json", bundle.features.sample_ids)
    np.save(workspace / "features.npy", bundle.features.matrix)

    _sample_table(bundle).to_csv(workspace / "samples.csv", index=False, encoding="utf-8-sig")
    bundle.risk.scores.to_csv(workspace / "risk_scores.csv", index_label="sample_id", encoding="utf-8-sig")
    bundle.risk.evidence.to_csv(workspace / "risk_evidence.csv", index_label="sample_id", encoding="utf-8-sig")
    bundle.audit.issues.to_csv(workspace / "cleanlab_issues.csv", index_label="sample_id", encoding="utf-8-sig")
    bundle.audit.summary.to_csv(workspace / "cleanlab_summary.csv", index=False, encoding="utf-8-sig")
    detector_status.to_csv(workspace / "detector_status.csv", index=False, encoding="utf-8-sig")
    bundle.relationships.per_sample.to_csv(
        workspace / "relationships.csv", index_label="sample_id", encoding="utf-8-sig"
    )
    bundle.relationships.projection.to_csv(
        workspace / "projection.csv", index_label="sample_id", encoding="utf-8-sig"
    )
    pd.DataFrame(
        {
            "sample_id": bundle.relationships.sample_ids,
            "suggested_label": bundle.relationships.suggested_labels.reindex(
                bundle.relationships.sample_ids
            ).tolist(),
            "suggested_confidence": bundle.relationships.suggested_confidence.reindex(
                bundle.relationships.sample_ids
            ).tolist(),
        }
    ).to_csv(workspace / "neighbour_suggestions.csv", index=False, encoding="utf-8-sig")
    return workspace
