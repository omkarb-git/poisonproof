"""End-to-end PoisonShield analysis orchestration.

This module wires the layers together in a fixed, reproducible order:

    ingestion profile
      -> feature representation
      -> out-of-sample class probabilities (when labelled classification)
      -> Cleanlab audit
      -> relationship intelligence
      -> unified risk assessment
      -> traceable explanations
      -> session artifacts

Every stage records real timings and warnings. Detector failures are surfaced
through the audit result and never silently converted into a clean dataset.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .auditing.cleanlab_engine import run_cleanlab_audit
from .config import AnalysisConfig, SESSIONS_DIR, ensure_workspace
from .features.image_features import extract_image_features
from .features.probabilities import compute_predicted_probabilities
from .features.tabular_features import build_tabular_features
from .intelligence.relationship_engine import build_relationships
from .intelligence.risk_engine import assess_risk, populate_explanations
from .types import AnalysisBundle, FeatureResult, LoadedDataset
from .utils import (
    PeakMemory,
    ProgressCallback,
    format_duration,
    make_progress,
    new_session_id,
    try_psutil_rss_mb,
    utc_now_iso,
)


class AnalysisError(RuntimeError):
    """Raised when a required analysis stage cannot complete."""


@dataclass
class AnalysisRun:
    bundle: AnalysisBundle
    workspace: Path


def _sub_progress(
    progress: ProgressCallback,
    start: float,
    end: float,
) -> ProgressCallback:
    span = max(end - start, 0.0)

    def report(fraction: float, message: str) -> None:
        progress(start + span * max(0.0, min(1.0, fraction)), message)

    return report


def _build_features(
    dataset: LoadedDataset,
    config: AnalysisConfig,
    progress: ProgressCallback,
) -> FeatureResult:
    if dataset.modality == "tabular":
        if dataset.dataframe is None:
            raise AnalysisError("Tabular dataset has no dataframe.")
        try:
            features = build_tabular_features(
                dataset.dataframe,
                label_column=dataset.label_column,
                id_like_columns=dataset.profile.id_like_columns,
                constant_columns=dataset.profile.constant_columns,
                all_missing_columns=dataset.profile.all_missing_columns,
                max_onehot_categories=config.max_onehot_categories,
                svd_components=config.svd_components,
                random_state=config.random_seed,
                excluded_columns=[config.ground_truth_column] if config.ground_truth_column else None,
            )
        except Exception as exc:  # noqa: BLE001
            raise AnalysisError(f"Tabular feature representation failed: {exc}") from exc
        features.sample_ids = list(dataset.sample_ids)
        return features

    if dataset.modality == "image":
        if not dataset.images:
            raise AnalysisError("Image dataset has no records.")
        try:
            features = extract_image_features(
                dataset.images,
                extractor=config.image_extractor,
                batch_size=config.image_batch_size,
                progress=progress,
            )
        except Exception as exc:  # noqa: BLE001
            raise AnalysisError(f"Image feature extraction failed: {exc}") from exc
        return features

    raise AnalysisError(f"Unsupported modality '{dataset.modality}'.")


def _feature_labels(dataset: LoadedDataset, features: FeatureResult) -> pd.Series:
    """Labels aligned to the *analyzable* feature rows (may contain None)."""

    labels = dataset.labels()
    if labels.empty:
        return pd.Series([None] * len(features.sample_ids), index=features.sample_ids, dtype=object)
    aligned = labels.reindex(features.sample_ids)
    return aligned.astype(object).where(aligned.notna(), None)


def analyse_dataset(
    dataset: LoadedDataset,
    config: AnalysisConfig,
    workspace: Path | None = None,
    progress: ProgressCallback | None = None,
) -> AnalysisBundle:
    """Run the complete PoisonShield audit for an ingested dataset."""

    ensure_workspace()
    config = config.validated()
    report = make_progress(progress)
    session_id = new_session_id()
    workspace = Path(workspace) if workspace else SESSIONS_DIR / session_id
    workspace.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    warnings: list[str] = list(dataset.notes)
    pipeline_started = time.perf_counter()

    report(0.01, "Preparing analysis session...")
    rss_start = try_psutil_rss_mb()

    with PeakMemory() as peak:
        # 1. Features -----------------------------------------------------
        report(0.03, "Building the analysis feature representation...")
        started = time.perf_counter()
        features = _build_features(dataset, config, _sub_progress(report, 0.03, 0.45))
        timings["feature_build"] = time.perf_counter() - started
        if features.method == "image":
            # Image extraction has its own internal timer; use wall-clock here.
            pass
        warnings.extend(features.warnings)

        report(0.46, "Preparing out-of-sample class probabilities...")
        labels = _feature_labels(dataset, features)
        started = time.perf_counter()
        probability_result = compute_predicted_probabilities(
            features.matrix,
            labels,
            features.sample_ids,
            task=dataset.task,
            modality=dataset.modality,
            k=config.k_neighbors,
            random_state=config.random_seed,
        )
        timings["predicted_probabilities"] = time.perf_counter() - started
        if probability_result.meta.get("logistic_error"):
            warnings.append(
                "Logistic out-of-sample probabilities failed and were replaced by "
                f"KNN probabilities: {probability_result.meta['logistic_error']}"
            )

        # 2. Cleanlab audit ----------------------------------------------
        report(0.50, "Running Cleanlab Datalab detectors...")
        started = time.perf_counter()
        try:
            audit = run_cleanlab_audit(
                sample_ids=features.sample_ids,
                features=features.matrix,
                labels=labels,
                task=dataset.task,
                pred_probs=probability_result.pred_probs,
                pred_class_order=probability_result.class_order,
                progress=_sub_progress(report, 0.50, 0.70),
            )
        except Exception as exc:  # noqa: BLE001
            raise AnalysisError(f"Cleanlab auditing failed: {exc}") from exc
        audit.pred_probs = probability_result.pred_probs
        audit.pred_probs_meta = probability_result.meta
        timings["cleanlab_audit"] = time.perf_counter() - started
        warnings.extend(audit.warnings)

        # 3. Relationships -----------------------------------------------
        report(0.72, "Computing neighbourhood relationships...")
        started = time.perf_counter()
        try:
            relationships = build_relationships(
                features.matrix,
                features.sample_ids,
                labels,
                modality=dataset.modality,
                k=config.k_neighbors,
                agreement_threshold=config.neighbor_agreement_threshold,
                random_state=config.random_seed,
                progress=_sub_progress(report, 0.72, 0.90),
            )
        except Exception as exc:  # noqa: BLE001
            raise AnalysisError(f"Relationship analysis failed: {exc}") from exc
        timings["relationships"] = time.perf_counter() - started
        warnings.extend(relationships.warnings)

        # 4. Risk ----------------------------------------------------------
        report(0.91, "Scoring unified sample risk...")
        started = time.perf_counter()
        risk = assess_risk(dataset, features, audit, relationships, config)
        timings["risk_scoring"] = time.perf_counter() - started

    timings["total"] = time.perf_counter() - pipeline_started
    rss_end = try_psutil_rss_mb()

    bundle = AnalysisBundle(
        session_id=session_id,
        created_at=utc_now_iso(),
        dataset=dataset,
        features=features,
        audit=audit,
        relationships=relationships,
        risk=risk,
        timings=timings,
        memory={
            "rss_start_mb": rss_start,
            "rss_end_mb": rss_end,
            "tracemalloc_peak_mb": peak.peak_mb,
        },
        warnings=sorted(set(warnings)),
        config={
            **config.to_dict(),
            "feature_meta": features.meta,
            "probability_meta": probability_result.meta,
            "relationship_meta": relationships.meta,
            "session_id": session_id,
            "created_at": utc_now_iso(),
        },
        workspace=workspace,
    )
    report(0.95, "Generating explanations for flagged samples...")
    populate_explanations(bundle)
    report(0.98, "Saving session artifacts...")
    try:
        from .reporting.artifacts import save_session_artifacts

        save_session_artifacts(bundle)
    except Exception as exc:  # noqa: BLE001 - artifact persistence must not hide results
        bundle.warnings.append(f"Session artifacts could not be fully saved: {exc}")
    report(1.0, f"Analysis complete in {format_duration(timings['total'])}.")
    return bundle
