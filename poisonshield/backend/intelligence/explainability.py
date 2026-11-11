"""Traceable, template-based explanations for flagged samples.

Every statement is generated from stored detector output or calculated
evidence. When a detector did not run, the explanation says so rather than
claiming the sample is clean. No natural-language model is used.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..types import AnalysisBundle, Explanation, Finding


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(result) else result


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.0f}%"


def _num(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _sample_label(bundle: AnalysisBundle, sample_id: str) -> str | None:
    labels = bundle.dataset.labels()
    if sample_id in labels.index:
        value = labels.loc[sample_id]
        return None if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    return None


def _image_record(bundle: AnalysisBundle, sample_id: str):
    for record in bundle.dataset.images or []:
        if record.sample_id == sample_id:
            return record
    return None


def build_explanation(
    bundle: AnalysisBundle,
    sample_id: str,
    missing_evidence: list[str],
) -> Explanation:
    evidence = bundle.risk.evidence.loc[sample_id] if sample_id in bundle.risk.evidence.index else pd.Series(dtype=object)
    score = bundle.risk.scores.loc[sample_id] if sample_id in bundle.risk.scores.index else pd.Series(dtype=object)
    original_label = _sample_label(bundle, sample_id)
    suggested_label = score.get("suggested_label")
    if suggested_label is None or (isinstance(suggested_label, float) and np.isnan(suggested_label)):
        suggested_label = None
    else:
        suggested_label = str(suggested_label)
    suggested_source = score.get("suggested_source")
    confidence = _as_float(score.get("suggested_confidence"))
    findings: list[Finding] = []

    # ---- label-quality evidence -----------------------------------------
    label_quality = _as_float(evidence.get("label_quality"))
    is_label_issue = bool(evidence.get("is_label_issue")) if pd.notna(evidence.get("is_label_issue")) else None
    if label_quality is not None:
        if is_label_issue:
            findings.append(
                Finding(
                    kind="label_quality",
                    severity=float(np.clip(1 - label_quality, 0, 1)),
                    text=(
                        f"Cleanlab's label-quality detector flagged this sample "
                        f"(label quality {_num(label_quality)} of 1)."
                    ),
                    evidence={
                        "label_quality": label_quality,
                        "is_label_issue": True,
                        "detector": "cleanlab.datalab.label",
                    },
                )
            )
        else:
            findings.append(
                Finding(
                    kind="label_quality_ok",
                    severity=0.0,
                    text=(
                        f"Cleanlab's label-quality detector did not flag this sample "
                        f"(label quality {_num(label_quality)} of 1)."
                    ),
                    evidence={"label_quality": label_quality, "is_label_issue": False},
                )
            )

    # ---- neighbourhood evidence -----------------------------------------
    agreement = _as_float(evidence.get("neighbor_label_agreement"))
    labelled_neighbours = _as_float(evidence.get("labelled_neighbors"))
    majority = evidence.get("majority_neighbor_label")
    majority_fraction = _as_float(evidence.get("majority_fraction"))
    conflict = bool(evidence.get("relationship_conflict")) if pd.notna(evidence.get("relationship_conflict")) else False
    k = bundle.relationships.k
    if agreement is not None and labelled_neighbours:
        if conflict:
            findings.append(
                Finding(
                    kind="neighborhood_conflict",
                    severity=float(np.clip(1 - agreement, 0, 1)),
                    text=(
                        f"Only {_pct(agreement)} of the {int(labelled_neighbours)} labelled nearest "
                        f"neighbours share the assigned label; the majority neighbour label is "
                        f"'{majority}' ({_pct(majority_fraction)} of labelled neighbours)."
                    ),
                    evidence={
                        "neighbor_label_agreement": agreement,
                        "labelled_neighbors": int(labelled_neighbours),
                        "majority_neighbor_label": None if majority is None else str(majority),
                        "majority_fraction": majority_fraction,
                        "k": k,
                        "metric": bundle.relationships.metric,
                    },
                )
            )
        else:
            findings.append(
                Finding(
                    kind="neighborhood_consistent",
                    severity=0.0,
                    text=(
                        f"{_pct(agreement)} of the {int(labelled_neighbours)} labelled nearest "
                        "neighbours share the assigned label."
                    ),
                    evidence={
                        "neighbor_label_agreement": agreement,
                        "labelled_neighbors": int(labelled_neighbours),
                        "k": k,
                    },
                )
            )
    elif bundle.dataset.has_labels:
        findings.append(
            Finding(
                kind="neighborhood_unavailable",
                severity=0.0,
                text=(
                    "Neighbourhood label consistency could not be evaluated because this sample "
                    "is outside the labelled feature subset or has too few labelled neighbours."
                ),
                evidence={},
            )
        )

    top_neighbor = evidence.get("top_neighbor_id")
    top_similarity = _as_float(evidence.get("top_similarity"))
    if top_neighbor is not None and top_similarity is not None:
        findings.append(
            Finding(
                kind="similarity",
                severity=0.0,
                text=(
                    f"The most similar sample is {top_neighbor} "
                    f"(similarity {_num(top_similarity, 3)}; metric {bundle.relationships.metric})."
                ),
                evidence={
                    "top_neighbor_id": str(top_neighbor),
                    "top_similarity": top_similarity,
                    "metric": bundle.relationships.metric,
                },
            )
        )

    # ---- outlier evidence ------------------------------------------------
    outlier_quality = _as_float(evidence.get("outlier_quality"))
    is_outlier = bool(evidence.get("is_outlier_issue")) if pd.notna(evidence.get("is_outlier_issue")) else None
    if outlier_quality is not None:
        if is_outlier:
            findings.append(
                Finding(
                    kind="outlier",
                    severity=float(np.clip(1 - outlier_quality, 0, 1)),
                    text=(
                        f"Cleanlab's outlier detector flagged this sample as atypical "
                        f"(outlier quality {_num(outlier_quality)} of 1)."
                    ),
                    evidence={
                        "outlier_quality": outlier_quality,
                        "is_outlier_issue": True,
                        "detector": "cleanlab.datalab.outlier",
                    },
                )
            )
        else:
            findings.append(
                Finding(
                    kind="outlier_ok",
                    severity=0.0,
                    text="No significant outlier indication was found by Cleanlab.",
                    evidence={"outlier_quality": outlier_quality, "is_outlier_issue": False},
                )
            )
    elif "outlier" in missing_evidence:
        findings.append(
            Finding(
                kind="outlier_missing",
                severity=0.0,
                text="Outlier evidence is unavailable for this sample.",
                evidence={},
            )
        )

    # ---- duplicate evidence ---------------------------------------------
    duplicate_quality = _as_float(evidence.get("near_duplicate_quality"))
    is_duplicate = bool(evidence.get("is_near_duplicate_issue")) if pd.notna(evidence.get("is_near_duplicate_issue")) else None
    record = _image_record(bundle, sample_id) if bundle.dataset.modality == "image" else None
    exact_duplicate_of = getattr(record, "exact_duplicate_of", None) if record is not None else None
    exact_duplicate_row = (
        bool(evidence.get("exact_duplicate"))
        if bundle.dataset.modality == "tabular" and pd.notna(evidence.get("exact_duplicate"))
        else False
    )
    if is_duplicate or exact_duplicate_of or exact_duplicate_row:
        if exact_duplicate_of:
            text = (
                f"This image file is byte-identical to {exact_duplicate_of} (SHA-256). "
                "Exact duplicates are not automatically poisoning, but they should be reviewed for redundancy."
            )
            duplicate_severity = 0.8
        elif exact_duplicate_row:
            text = (
                "This tabular row is an exact duplicate of at least one other row across every "
                "column. Exact duplicates are not automatically poisoning, but they duplicate "
                "training weight and should be reviewed for redundancy."
            )
            duplicate_severity = 0.8
        else:
            text = (
                "Cleanlab's near-duplicate detector flagged this sample as highly similar to "
                f"another sample (near-duplicate quality {_num(duplicate_quality)} of 1)."
            )
            duplicate_severity = (
                1 - duplicate_quality if duplicate_quality is not None else 0.0
            )
        findings.append(
            Finding(
                kind="duplicate",
                severity=max(
                    duplicate_severity,
                    1 - duplicate_quality if duplicate_quality is not None else 0.0,
                ),
                text=text,
                evidence={
                    "near_duplicate_quality": duplicate_quality,
                    "is_near_duplicate_issue": bool(is_duplicate),
                    "exact_duplicate_of": exact_duplicate_of,
                    "exact_duplicate_row": exact_duplicate_row,
                },
            )
        )
    elif duplicate_quality is not None:
        findings.append(
            Finding(
                kind="duplicate_ok",
                severity=0.0,
                text="No near-duplicate relationship was detected for this sample.",
                evidence={"near_duplicate_quality": duplicate_quality, "is_near_duplicate_issue": False},
            )
        )

    # ---- integrity evidence ---------------------------------------------
    integrity = _as_float(evidence.get("integrity_score")) or 0.0
    integrity_reasons = evidence.get("integrity_reasons")
    if integrity >= 0.25:
        findings.append(
            Finding(
                kind="integrity",
                severity=float(np.clip(integrity, 0, 1)),
                text=(
                    "Data-integrity checks found: "
                    + (str(integrity_reasons) if integrity_reasons else "incomplete or invalid values")
                    + "."
                ),
                evidence={"integrity_score": integrity, "reasons": integrity_reasons},
            )
        )

    # ---- suggested correction -------------------------------------------
    if suggested_label is not None and original_label is not None and suggested_label != original_label:
        findings.append(
            Finding(
                kind="suggested_label",
                severity=float(confidence) if confidence is not None else 0.3,
                text=(
                    f"Supporting evidence suggests the label '{suggested_label}' "
                    f"({suggested_source or 'neighbour analysis'}"
                    + (f", confidence {_pct(confidence)}" if confidence is not None else "")
                    + ")."
                ),
                evidence={
                    "suggested_label": suggested_label,
                    "suggested_source": suggested_source,
                    "suggested_confidence": confidence,
                },
            )
        )

    # ---- narrative -------------------------------------------------------
    significant = [f for f in findings if f.severity >= 0.35 and f.kind not in {"duplicate_ok"}]
    kinds = {f.kind for f in significant}
    narrative_parts: list[str] = []
    if "neighborhood_conflict" in kinds:
        narrative_parts.append(
            f"The assigned label '{original_label}' differs from the labels of several "
            "feature-similar samples."
        )
        if suggested_label is not None:
            narrative_parts.append(
                f"The nearest-neighbour evidence points toward '{suggested_label}', "
                f"but this is supporting evidence only."
            )
    elif "label_quality" in kinds:
        narrative_parts.append(
            f"Cleanlab's label-quality analysis indicates that the assigned label "
            f"'{original_label}' may be unreliable."
        )
    if "duplicate" in kinds:
        narrative_parts.append(
            "This sample has a duplicate relationship that may affect training if retained."
        )
    if "outlier" in kinds:
        narrative_parts.append(
            "Its feature representation is atypical compared with the rest of the dataset, "
            "which can indicate corruption or a legitimately rare sample."
        )
    if "integrity" in kinds:
        narrative_parts.append("The record contains data-integrity problems.")
    if not narrative_parts:
        if score.get("risk_category") in {"High", "Critical"}:
            narrative_parts.append(
                "Multiple weak signals combine to place this sample in a high-priority review band."
            )
        else:
            narrative_parts.append(
                "Detector evidence was reviewed and no strong issue signature is present."
            )
    narrative_parts.append(
        "PoisonShield prioritises this sample for manual verification. "
        "Suspicion is not confirmation of poisoning and should be verified independently."
    )

    return Explanation(
        sample_id=sample_id,
        original_label=original_label,
        suggested_label=suggested_label,
        suggested_source=suggested_source,
        findings=findings,
        narrative=" ".join(narrative_parts),
        risk_score=float(score.get("risk_score", 0.0)),
        risk_category=str(score.get("risk_category", "Low")),
        missing_evidence=list(missing_evidence),
    )


def build_explanations(
    bundle: AnalysisBundle,
    flagged_ids: list[str] | None = None,
) -> dict[str, Explanation]:
    explanations: dict[str, Explanation] = {}
    scores = bundle.risk.scores
    if flagged_ids is None:
        flagged_ids = scores.index.tolist()
    for sample_id in flagged_ids:
        explanations[sample_id] = build_explanation(
            bundle,
            sample_id,
            bundle.risk.missing_evidence,
        )
    return explanations
