"""Unified PoisonShield risk scoring.

Signals are normalised to [0, 1] where 1 means more risk. Correlated signals
(Cleanlab label quality and neighbourhood label disagreement) are fused inside
one evidence group instead of being counted twice. Unavailable evidence groups
are excluded from the weighted average and are reported explicitly, so a
missing detector never silently improves a sample's score.

The score is a review-prioritisation score, not a probability of poisoning.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..config import AnalysisConfig, RiskThresholds, RiskWeights
from ..types import AnalysisBundle, AuditResult, DatasetProfile, FeatureResult, LoadedDataset, RelationshipResult, RiskResult
from ..utils import json_safe, timer
from .explainability import build_explanations

SIGNAL_GROUPS = ("label_evidence", "outlier_evidence", "duplicate_evidence", "integrity_evidence")
GROUP_LABELS = {
    "label_evidence": "Label and neighbourhood evidence",
    "outlier_evidence": "Outlier evidence",
    "duplicate_evidence": "Duplicate evidence",
    "integrity_evidence": "Data-integrity evidence",
}


def _safe_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result


def _clipped_risk(quality: Any) -> float:
    quality = _safe_float(quality)
    if np.isnan(quality):
        return np.nan
    return float(np.clip(1.0 - quality, 0.0, 1.0))


def _image_integrity(dataset: LoadedDataset) -> pd.DataFrame:
    records = {record.sample_id: record for record in (dataset.images or [])}
    rows = []
    for sample_id in dataset.sample_ids:
        record = records.get(sample_id)
        if record is None:
            rows.append(
                {
                    "integrity_score": 1.0,
                    "integrity_reasons": "missing image record",
                }
            )
            continue
        score = 0.0
        reasons: list[str] = []
        if not record.readable:
            score = 1.0
            reasons.append(f"unreadable image ({record.error or 'decode error'})")
        else:
            if (
                record.brightness_std is not None
                and record.entropy is not None
                and (record.brightness_std < 6.0 or record.entropy < 1.8)
            ):
                score = max(score, 0.6)
                reasons.append(
                    f"low visual information (brightness std {record.brightness_std:.2f}, "
                    f"entropy {record.entropy:.2f})"
                )
            if "format_mismatch" in record.extra:
                score = max(score, 0.3)
                reasons.append(f"file-format mismatch ({record.extra['format_mismatch']})")
        rows.append(
            {
                "integrity_score": float(np.clip(score, 0.0, 1.0)),
                "integrity_reasons": "; ".join(reasons),
            }
        )
    return pd.DataFrame(rows, index=dataset.sample_ids)


def _tabular_integrity(features: FeatureResult) -> pd.DataFrame:
    integrity = features.meta.get("row_integrity")
    if not isinstance(integrity, pd.DataFrame) or integrity.empty:
        return pd.DataFrame(
            {"integrity_score": 0.0, "integrity_reasons": ""},
            index=features.sample_ids,
        )
    frame = integrity.copy()
    frame.index = features.sample_ids[: len(frame)]
    reasons: list[str] = []
    for _, row in frame.iterrows():
        items: list[str] = []
        missing_ratio = _safe_float(row.get("missing_ratio"))
        if not np.isnan(missing_ratio) and missing_ratio > 0:
            items.append(f"{missing_ratio:.0%} of feature values missing")
        sentinel_ratio = _safe_float(row.get("sentinel_ratio"))
        if not np.isnan(sentinel_ratio) and sentinel_ratio > 0:
            items.append(f"missing-value sentinels in {sentinel_ratio:.0%} of features")
        infinite = _safe_float(row.get("infinite_values"))
        if not np.isnan(infinite) and infinite > 0:
            items.append(f"{int(infinite)} infinite value(s)")
        reasons.append("; ".join(items))
    frame["integrity_reasons"] = reasons
    return frame


def build_risk_evidence(
    dataset: LoadedDataset,
    features: FeatureResult,
    audit: AuditResult,
    relationships: RelationshipResult,
) -> pd.DataFrame:
    """Merge all detector outputs into one sample-aligned evidence table."""

    index = pd.Index(dataset.sample_ids, name="sample_id")
    evidence = pd.DataFrame(index=index)

    issues = audit.issues.reindex(index)
    quality_map = {
        "label_quality": "label_score",
        "outlier_quality": "outlier_score",
        "near_duplicate_quality": "near_duplicate_score",
        "class_imbalance_quality": "class_imbalance_score",
        "non_iid_quality": "non_iid_score",
    }
    for output_column, source_column in quality_map.items():
        evidence[output_column] = pd.to_numeric(issues.get(source_column), errors="coerce")
    for output_column, source_column in {
        "is_label_issue": "is_label_issue",
        "is_outlier_issue": "is_outlier_issue",
        "is_near_duplicate_issue": "is_near_duplicate_issue",
    }.items():
        if source_column in issues.columns:
            evidence[output_column] = issues[source_column].astype("boolean")
        else:
            evidence[output_column] = pd.array([pd.NA] * len(index), dtype="boolean")
    evidence["predicted_label"] = issues.get("predicted_label", pd.Series([None] * len(index), index=index))

    labels = dataset.labels().reindex(index)
    evidence["label"] = labels

    rel = relationships.per_sample.reindex(index)
    for column in [
        "neighbor_label_agreement",
        "majority_neighbor_label",
        "majority_fraction",
        "labelled_neighbors",
        "conflicting_neighborhood",
        "mean_distance",
        "mean_similarity",
        "isolation_percentile",
        "top_neighbor_id",
        "top_similarity",
    ]:
        evidence[column] = rel[column] if column in rel.columns else np.nan
    evidence["relationship_conflict"] = evidence["conflicting_neighborhood"].fillna(False).astype(bool)
    evidence["neighbor_suggestion"] = relationships.suggested_labels.reindex(index)
    evidence["neighbor_confidence"] = relationships.suggested_confidence.reindex(index)

    if dataset.modality == "tabular":
        integrity = _tabular_integrity(features)
    else:
        integrity = _image_integrity(dataset)
    evidence["integrity_score"] = integrity["integrity_score"].reindex(index).fillna(0.0)
    evidence["integrity_reasons"] = integrity["integrity_reasons"].reindex(index).fillna("")

    # Exact duplicates are tracked independently of Cleanlab near-duplicates.
    # Every member of a duplicate group is marked, not only the later copies.
    if dataset.modality == "tabular" and dataset.dataframe is not None:
        duplicate_mask = dataset.dataframe.duplicated(keep=False).to_numpy()
        evidence["exact_duplicate"] = pd.Series(
            duplicate_mask, index=index, dtype=bool
        )
    elif dataset.modality == "image":
        duplicate_ids: set[str] = set()
        for record in dataset.images or []:
            if record.exact_duplicate_of:
                duplicate_ids.add(record.sample_id)
                duplicate_ids.add(record.exact_duplicate_of)
        evidence["exact_duplicate"] = pd.Series(
            index.isin(duplicate_ids), index=index, dtype=bool
        )
    else:
        evidence["exact_duplicate"] = pd.Series(False, index=index, dtype=bool)

    # Model-prediction confidence from the out-of-sample probabilities.
    prediction_confidence = pd.Series(np.nan, index=index, dtype=float)
    if audit.pred_probs is not None and features.sample_ids:
        try:
            probs = np.asarray(audit.pred_probs, dtype=float)
            if probs.ndim == 2 and probs.shape[0] == len(features.sample_ids):
                prediction_confidence = pd.Series(
                    probs.max(axis=1), index=features.sample_ids
                ).reindex(index)
        except Exception:  # noqa: BLE001 - confidence is optional
            pass
    evidence["model_confidence"] = prediction_confidence

    evidence["label_check_available"] = evidence["label_quality"].notna()
    evidence["outlier_check_available"] = evidence["outlier_quality"].notna()
    evidence["duplicate_check_available"] = evidence["near_duplicate_quality"].notna()
    evidence["relationship_check_available"] = evidence["neighbor_label_agreement"].notna()
    return evidence


def _score_sample(signals: dict[str, float], availability: dict[str, bool], weights: RiskWeights) -> tuple[float, float]:
    numerator = 0.0
    denominator = 0.0
    for group in SIGNAL_GROUPS:
        if not availability.get(group):
            continue
        weight = getattr(weights, group)
        numerator += weight * signals.get(group, 0.0)
        denominator += weight
    if denominator <= 0:
        return 0.0, 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0)), float(denominator)


def _category(score: float, thresholds: RiskThresholds) -> str:
    if score <= thresholds.low_max:
        return "Low"
    if score <= thresholds.moderate_max:
        return "Moderate"
    if score <= thresholds.high_max:
        return "High"
    return "Critical"


def _recommendation(
    category: str,
    label_issue: bool,
    outlier: bool,
    duplicate: bool,
    conflict: bool,
    integrity: float,
    suggested_label: str | None,
) -> str:
    if category == "Critical":
        base = "Quarantine from training and verify this sample manually."
    elif category == "High":
        base = "Manual review required before this sample is used for training."
    elif category == "Moderate":
        base = "Spot-check this sample during review."
    else:
        base = "No action required; retain unless other evidence appears."
    extras: list[str] = []
    if suggested_label is not None:
        extras.append(f"Verify the assigned label against the suggested label '{suggested_label}'.")
    if duplicate:
        extras.append("Review duplicate status; consider removing redundant copies after verification.")
    if outlier:
        extras.append("Inspect for corruption or legitimate rarity before removing.")
    if conflict:
        extras.append("Inspect the nearest neighbours listed in the Relationship Explorer.")
    if integrity >= 0.5:
        extras.append("Repair or remove the record if the integrity problem is confirmed.")
    return " ".join([base] + extras)


def score_evidence(
    evidence: pd.DataFrame,
    dataset: LoadedDataset,
    weights: RiskWeights,
    thresholds: RiskThresholds,
    missing_evidence: list[str],
    audit: AuditResult | None = None,
    relationships: RelationshipResult | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    weights = weights.validated()
    thresholds = thresholds.validated()
    index = evidence.index
    scores = pd.DataFrame(index=index)
    scores["label"] = evidence.get("label")
    scores["label_quality"] = evidence.get("label_quality")
    scores["outlier_quality"] = evidence.get("outlier_quality")
    scores["near_duplicate_quality"] = evidence.get("near_duplicate_quality")
    scores["relationship_consistency"] = evidence.get("neighbor_label_agreement")

    label_issue = evidence.get("is_label_issue", pd.Series(False, index=index)).fillna(False).astype(bool)
    outlier_flag = evidence.get("is_outlier_issue", pd.Series(False, index=index)).fillna(False).astype(bool)
    duplicate_flag = evidence.get("is_near_duplicate_issue", pd.Series(False, index=index)).fillna(False).astype(bool)
    conflict_flag = evidence.get("relationship_conflict", pd.Series(False, index=index)).fillna(False).astype(bool)
    scores["label_issue"] = label_issue
    scores["outlier"] = outlier_flag
    scores["duplicate"] = duplicate_flag
    scores["relationship_conflict"] = conflict_flag
    scores["label_check_available"] = evidence.get("label_check_available", False)
    scores["outlier_check_available"] = evidence.get("outlier_check_available", False)
    scores["duplicate_check_available"] = evidence.get("duplicate_check_available", False)
    scores["relationship_check_available"] = evidence.get("relationship_check_available", False)

    # Normalised risk signals. Cleanlab exposes quality scores (higher better).
    label_signal = (1.0 - pd.to_numeric(evidence.get("label_quality"), errors="coerce")).clip(0.0, 1.0)
    label_signal = label_signal.mask(label_issue & label_signal.lt(0.45), 0.45)

    agreement = pd.to_numeric(evidence.get("neighbor_label_agreement"), errors="coerce")
    relationship_signal = (1.0 - agreement).clip(0.0, 1.0)
    relationship_signal = relationship_signal.mask(conflict_flag, np.maximum(relationship_signal.fillna(0), 0.60))

    both = label_signal.notna() & relationship_signal.notna()
    label_group = pd.Series(np.nan, index=index, dtype=float)
    label_group.loc[both] = np.minimum(
        1.0,
        np.maximum(label_signal[both], relationship_signal[both])
        + 0.10 * np.minimum(label_signal[both], relationship_signal[both]),
    )
    label_group.loc[label_signal.notna() & ~relationship_signal.notna()] = label_signal
    label_group.loc[~label_signal.notna() & relationship_signal.notna()] = relationship_signal

    outlier_signal = (1.0 - pd.to_numeric(evidence.get("outlier_quality"), errors="coerce")).clip(0.0, 1.0)
    isolation = pd.to_numeric(evidence.get("isolation_percentile"), errors="coerce")
    outlier_fallback = isolation.notna() & outlier_signal.isna()
    outlier_signal = outlier_signal.fillna((isolation * 0.8).clip(0.0, 1.0))

    duplicate_quality = pd.to_numeric(evidence.get("near_duplicate_quality"), errors="coerce")
    duplicate_signal = (1.0 - duplicate_quality).clip(0.0, 1.0)
    if "exact_duplicate" in evidence.columns:
        exact_mask = evidence["exact_duplicate"].fillna(False).astype(bool)
    elif dataset.modality == "image":
        exact_duplicate_ids = {
            record.sample_id for record in (dataset.images or []) if record.exact_duplicate_of
        }
        exact_mask = pd.Series(index.isin(exact_duplicate_ids), index=index, dtype=bool)
    else:
        exact_mask = pd.Series(False, index=index, dtype=bool)
    duplicate_available = duplicate_quality.notna() | exact_mask
    duplicate_signal = duplicate_signal.fillna(0.0).where(~exact_mask, 1.0)

    integrity_signal = pd.to_numeric(evidence.get("integrity_score"), errors="coerce").fillna(0.0).clip(0.0, 1.0)

    signals = pd.DataFrame(
        {
            "label_signal": label_signal,
            "relationship_signal": relationship_signal,
            "label_group": label_group,
            "outlier_signal": outlier_signal,
            "duplicate_signal": duplicate_signal,
            "integrity_signal": integrity_signal,
        },
        index=index,
    )
    availability = pd.DataFrame(
        {
            "label_evidence": label_group.notna(),
            "outlier_evidence": outlier_signal.notna(),
            "duplicate_evidence": duplicate_available,
            "integrity_evidence": pd.Series(True, index=index),
        },
        index=index,
    )

    risk_scores: list[float] = []
    coverages: list[float] = []
    category_values: list[str] = []
    recommendations: list[str] = []
    suggested_labels: list[str | None] = []
    suggested_sources: list[str | None] = []
    suggested_confidences: list[float | None] = []
    neighbor_suggestion = evidence.get("neighbor_suggestion", pd.Series(None, index=index))
    neighbor_confidence = pd.to_numeric(evidence.get("neighbor_confidence"), errors="coerce")
    model_prediction = evidence.get("predicted_label", pd.Series(None, index=index))
    model_confidence = pd.to_numeric(evidence.get("model_confidence"), errors="coerce")
    original_labels = evidence.get("label", pd.Series(None, index=index))

    for position, sample_id in enumerate(index):
        row_signals = {
            "label_evidence": 0.0 if pd.isna(signals["label_group"].iloc[position]) else float(signals["label_group"].iloc[position]),
            "outlier_evidence": 0.0 if pd.isna(signals["outlier_signal"].iloc[position]) else float(signals["outlier_signal"].iloc[position]),
            "duplicate_evidence": 0.0 if pd.isna(signals["duplicate_signal"].iloc[position]) else float(signals["duplicate_signal"].iloc[position]),
            "integrity_evidence": float(signals["integrity_signal"].iloc[position]),
        }
        row_availability = {group: bool(availability[group].iloc[position]) for group in SIGNAL_GROUPS}
        score, weight_available = _score_sample(row_signals, row_availability, weights)
        coverage = float(weight_available)
        category = _category(score, thresholds)

        original = original_labels.iloc[position]
        original_str = None if original is None or (isinstance(original, float) and np.isnan(original)) else str(original)
        candidate = neighbor_suggestion.iloc[position]
        candidate = None if candidate is None or (isinstance(candidate, float) and np.isnan(candidate)) else str(candidate)
        candidate_confidence = None if pd.isna(neighbor_confidence.iloc[position]) else float(neighbor_confidence.iloc[position])
        source = "neighbour vote"
        if candidate is None or candidate_confidence is None or candidate_confidence < 0.5:
            fallback = model_prediction.iloc[position]
            fallback = None if fallback is None or (isinstance(fallback, float) and np.isnan(fallback)) else str(fallback)
            fallback_confidence = None if pd.isna(model_confidence.iloc[position]) else float(model_confidence.iloc[position])
            if fallback is not None and fallback_confidence is not None and fallback_confidence >= 0.5:
                candidate, candidate_confidence, source = fallback, fallback_confidence, "out-of-sample model"
            else:
                # A low-confidence guess is not actionable evidence. Prefer
                # silence over inventing a label suggestion.
                candidate, candidate_confidence, source = None, None, None

        if candidate is not None and original_str is not None and candidate == original_str:
            candidate = None
            candidate_confidence = None
            source = None

        risk_scores.append(score)
        coverages.append(coverage)
        category_values.append(category)
        suggested_labels.append(candidate)
        suggested_sources.append(source if candidate is not None else None)
        suggested_confidences.append(candidate_confidence)
        recommendations.append(
            _recommendation(
                category,
                bool(label_issue.iloc[position]),
                bool(outlier_flag.iloc[position]),
                bool(duplicate_flag.iloc[position] or exact_mask.iloc[position]),
                bool(conflict_flag.iloc[position]),
                float(integrity_signal.iloc[position]),
                candidate,
            )
        )

    scores["suggested_label"] = suggested_labels
    scores["suggested_source"] = suggested_sources
    scores["suggested_confidence"] = suggested_confidences
    scores["risk_score"] = risk_scores
    scores["risk_category"] = category_values
    scores["evidence_coverage"] = coverages
    scores["recommended_action"] = recommendations
    scores["integrity_score"] = integrity_signal
    scores["exact_duplicate"] = exact_mask
    scores["neighbor_suggestion"] = neighbor_suggestion
    scores["model_prediction"] = model_prediction

    # Persist availability flags and signals for re-scoring / auditability.
    for column in signals.columns:
        evidence[column] = signals[column]
    for column in availability.columns:
        evidence[column] = availability[column]
    evidence["outlier_source"] = np.where(outlier_fallback, "local isolation fallback", "cleanlab")
    evidence["outlier_source"] = evidence["outlier_source"].where(outlier_signal.notna(), "unavailable")

    quality = _quality_summary(scores, evidence, audit, relationships, missing_evidence, weights)
    return scores, quality


def _quality_summary(
    scores: pd.DataFrame,
    evidence: pd.DataFrame,
    audit: AuditResult | None,
    relationships: RelationshipResult | None,
    missing_evidence: list[str],
    weights: RiskWeights,
) -> dict[str, Any]:
    n = len(scores)
    assessed = int((scores["evidence_coverage"] > 0).sum()) if n else 0
    mean_risk = float(scores["risk_score"].mean()) if n else None
    distribution = scores["risk_category"].value_counts().to_dict() if n else {}
    label_issues = int(scores["label_issue"].sum()) if n else 0
    outliers = int(scores["outlier"].sum()) if n else 0
    near_duplicates = int(scores["duplicate"].sum()) if n else 0
    exact_duplicates = int(scores["exact_duplicate"].sum()) if n else 0
    conflicts = int(scores["relationship_conflict"].sum()) if n else 0
    high_risk = int(scores["risk_category"].isin(["High", "Critical"]).sum()) if n else 0
    review_mask = (
        scores["risk_category"].isin(["High", "Critical"])
        | scores["label_issue"]
        | scores["duplicate"]
        | scores["exact_duplicate"]
        | scores["relationship_conflict"]
    )
    requiring_review = int(review_mask.sum()) if n else 0
    integrity_mean = float(scores["integrity_score"].mean()) if n else None
    agreement = pd.to_numeric(scores["relationship_consistency"], errors="coerce")
    consistency = float(agreement.mean()) if agreement.notna().any() else None

    detector_quality: dict[str, float | None] = {}
    detector_completion = 0
    detector_applicable = 0
    if audit is not None:
        if not audit.summary.empty:
            for _, row in audit.summary.iterrows():
                detector_quality[str(row["issue_type"])] = float(row["score"])
        detector_statuses = audit.statuses
        detector_applicable = len(detector_statuses)
        detector_completion = sum(1 for status in detector_statuses if status.status == "completed")
        class_imbalance = audit.dataset_level_findings.get("class_imbalance", {})
        non_iid = audit.dataset_level_findings.get("non_iid", {})
    else:
        class_imbalance, non_iid = {}, {}

    quality_index = None if mean_risk is None else float(np.clip(100.0 * (1.0 - mean_risk), 0.0, 100.0))
    coverage = float(scores["evidence_coverage"].mean()) if n else 0.0
    return {
        "quality_index": quality_index,
        "mean_risk": mean_risk,
        "n_samples": n,
        "n_assessed": assessed,
        "risk_distribution": {key: int(distribution.get(key, 0)) for key in ["Low", "Moderate", "High", "Critical"]},
        "n_label_issues": label_issues,
        "n_outliers": outliers,
        "n_near_duplicates": near_duplicates,
        "n_exact_duplicates": exact_duplicates,
        "n_relationship_conflicts": conflicts,
        "n_high_risk": high_risk,
        "n_requiring_review": requiring_review,
        "integrity_mean": integrity_mean,
        "relationship_consistency": consistency,
        "detector_quality": json_safe(detector_quality),
        "detector_completion": f"{detector_completion}/{detector_applicable}" if detector_applicable else "0/0",
        "detector_completion_ratio": (
            detector_completion / detector_applicable if detector_applicable else 0.0
        ),
        "class_imbalance": json_safe(class_imbalance),
        "non_iid": json_safe(non_iid),
        "missing_evidence": sorted(set(missing_evidence)),
        "mean_evidence_coverage": coverage,
        "weights": weights.to_dict(),
        "notes": [
            "The Quality Index is 100 x (1 - mean review risk) across samples. It is a screening "
            "summary, not an accuracy metric.",
            "Class-imbalance and non-IID results are dataset-level and are excluded from per-sample risk.",
            "Scores prioritise manual review; they do not confirm poisoning.",
        ],
    }


def assess_risk(
    dataset: LoadedDataset,
    features: FeatureResult,
    audit: AuditResult,
    relationships: RelationshipResult,
    config: AnalysisConfig,
) -> RiskResult:
    evidence = build_risk_evidence(dataset, features, audit, relationships)
    missing = list(audit.missing_evidence)
    if not dataset.has_labels:
        missing.extend(["label", "class_imbalance", "non_iid"])
    missing = sorted(set(missing))
    with timer() as clock:
        scores, quality = score_evidence(
            evidence,
            dataset,
            config.risk_weights,
            config.risk_thresholds,
            missing,
            audit,
            relationships,
        )
    return RiskResult(
        scores=scores,
        evidence=evidence,
        explanations={},
        weights=config.risk_weights.to_dict(),
        thresholds={
            "low_max": config.risk_thresholds.low_max,
            "moderate_max": config.risk_thresholds.moderate_max,
            "high_max": config.risk_thresholds.high_max,
        },
        quality=quality,
        missing_evidence=missing,
        duration_s=clock["seconds"],
    )


def flagged_sample_ids(scores: pd.DataFrame) -> list[str]:
    """Samples that need a stored explanation."""

    if scores.empty:
        return []
    mask = (
        scores["risk_category"].isin(["High", "Critical"])
        | scores["label_issue"]
        | scores["outlier"]
        | scores["duplicate"]
        | scores["exact_duplicate"]
        | scores["relationship_conflict"]
        | (scores["integrity_score"].fillna(0) >= 0.25)
    )
    return scores.index[mask].tolist()


def populate_explanations(bundle: AnalysisBundle) -> None:
    ids = flagged_sample_ids(bundle.risk.scores)
    bundle.risk.explanations = build_explanations(bundle, ids)


def rescore_bundle(
    bundle: AnalysisBundle,
    weights: RiskWeights,
    thresholds: RiskThresholds,
    config: AnalysisConfig | None = None,
) -> RiskResult:
    """Recompute risk after a weight/threshold change without re-running detectors."""

    evidence = bundle.risk.evidence.copy()
    # Remove previously derived signal columns so score_evidence regenerates them.
    derived = [
        "label_signal", "relationship_signal", "label_group", "outlier_signal",
        "duplicate_signal", "integrity_signal", "outlier_source",
        "label_evidence", "outlier_evidence", "duplicate_evidence", "integrity_evidence",
    ]
    for column in derived:
        if column in evidence.columns:
            evidence = evidence.drop(columns=column)

    if config is not None:
        config.risk_weights = weights
        config.risk_thresholds = thresholds
    with timer() as clock:
        scores, quality = score_evidence(
            evidence,
            bundle.dataset,
            weights,
            thresholds,
            bundle.risk.missing_evidence,
            bundle.audit,
            bundle.relationships,
        )
    risk = RiskResult(
        scores=scores,
        evidence=evidence,
        explanations={},
        weights=weights.validated().to_dict(),
        thresholds={
            "low_max": thresholds.low_max,
            "moderate_max": thresholds.moderate_max,
            "high_max": thresholds.high_max,
        },
        quality=quality,
        missing_evidence=bundle.risk.missing_evidence,
        duration_s=clock["seconds"],
    )
    bundle.risk = risk
    populate_explanations(bundle)
    bundle.config["risk_weights"] = risk.weights
    bundle.config["risk_thresholds"] = risk.thresholds
    return risk
