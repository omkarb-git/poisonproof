"""Dataset sanitisation: apply human decisions to copies, never originals."""

from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import EXPORT_DIR
from ..types import AnalysisBundle
from ..utils import human_bytes, timestamp_slug
from .reviewer import ReviewStore


@dataclass
class SanitisationResult:
    modality: str
    n_original: int
    n_included: int
    n_removed: int
    n_corrected: int
    n_uncertain: int
    changes: pd.DataFrame
    dataframe: pd.DataFrame | None = None
    archive_path: Path | None = None
    labels_frame: pd.DataFrame | None = None
    output_path: Path | None = None
    notes: list[str] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        return {
            "modality": self.modality,
            "n_original": self.n_original,
            "n_included": self.n_included,
            "n_removed": self.n_removed,
            "n_corrected": self.n_corrected,
            "n_uncertain": self.n_uncertain,
            "output_path": str(self.output_path) if self.output_path else None,
            "notes": self.notes,
        }


def _coerce_corrected_label(value: str | None, original_series: pd.Series) -> Any:
    if value is None:
        return None
    if pd.api.types.is_numeric_dtype(original_series):
        try:
            numeric = float(value)
            if numeric.is_integer() and pd.api.types.is_integer_dtype(original_series):
                return int(numeric)
            return numeric
        except (TypeError, ValueError):
            return value
    return value


def build_tabular_cleaned(
    bundle: AnalysisBundle,
    store: ReviewStore,
    include_review_columns: bool = False,
) -> SanitisationResult:
    dataset = bundle.dataset
    if dataset.modality != "tabular" or dataset.dataframe is None:
        raise ValueError("Tabular sanitisation requires a tabular bundle.")
    frame = dataset.dataframe.copy().reset_index(drop=True)
    label_column = dataset.label_column
    changes = store.changes_frame()
    removed = 0
    corrected = 0
    uncertain = 0
    kept_positions: list[int] = []
    label_updates: dict[int, Any] = {}
    original_labels = dataset.labels()

    for position, sample_id in enumerate(dataset.sample_ids):
        decision = store.get(sample_id)
        if decision is not None and decision.action == "remove":
            removed += 1
            continue
        kept_positions.append(position)
        if decision is not None:
            if decision.action == "approve_correction" and label_column:
                new_value = _coerce_corrected_label(
                    decision.corrected_label,
                    frame[label_column] if label_column in frame else pd.Series(dtype=object),
                )
                label_updates[position] = new_value
                corrected += 1
            elif decision.action == "uncertain":
                uncertain += 1

    cleaned = frame.iloc[kept_positions].copy().reset_index(drop=True)
    if label_column and label_column in cleaned.columns:
        for position, new_value in label_updates.items():
            if position in kept_positions:
                cleaned_position = kept_positions.index(position)
                cleaned.at[cleaned_position, label_column] = new_value

    if include_review_columns:
        cleaned.insert(0, "poisonshield_sample_id", [dataset.sample_ids[p] for p in kept_positions])
        cleaned.insert(1, "poisonshield_review_action", [
            (store.get(dataset.sample_ids[p]).action if store.get(dataset.sample_ids[p]) else "")
            for p in kept_positions
        ])

    if corrected and label_column:
        original_dtype = frame[label_column].dtype
        try:
            cleaned[label_column] = cleaned[label_column].astype(original_dtype)
        except (TypeError, ValueError):
            cleaned[label_column] = cleaned[label_column].astype(object)
            bundle.warnings.append(
                f"Corrected label values could not be cast back to '{original_dtype}'; "
                "the cleaned label column is stored as text."
            )

    notes = [
        "The original uploaded dataset was not modified.",
        "Only samples with explicit review decisions were changed or removed.",
    ]
    if not store.decisions:
        notes.append("No review decisions were recorded; the cleaned copy is identical to the original.")

    return SanitisationResult(
        modality="tabular",
        n_original=len(frame),
        n_included=len(cleaned),
        n_removed=removed,
        n_corrected=corrected,
        n_uncertain=uncertain,
        changes=changes,
        dataframe=cleaned,
        output_path=None,
        notes=notes,
    )


def _safe_folder(label: str | None) -> str:
    if label is None:
        return "unlabeled"
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", str(label)).strip(" .")
    return cleaned[:80] or "unlabeled"


def _unique_name(name: str, used: set[str]) -> str:
    candidate = name
    stem = Path(name).stem
    suffix = Path(name).suffix
    counter = 1
    while candidate.lower() in used:
        counter += 1
        candidate = f"{stem}_{counter}{suffix}"
    used.add(candidate.lower())
    return candidate


def build_image_cleaned(
    bundle: AnalysisBundle,
    store: ReviewStore,
    output_path: Path | None = None,
) -> SanitisationResult:
    dataset = bundle.dataset
    if dataset.modality != "image" or not dataset.images:
        raise ValueError("Image sanitisation requires an image bundle.")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = Path(output_path) if output_path else (
        EXPORT_DIR / f"{bundle.session_id}_cleaned_images_{timestamp_slug()}.zip"
    )
    changes = store.changes_frame()
    rows: list[dict[str, Any]] = []
    removed = 0
    corrected = 0
    uncertain = 0
    used_names: set[str] = set()
    readme_lines = [
        "PoisonShield sanitised image dataset",
        f"Session: {bundle.session_id}",
        f"Created: {timestamp_slug()}",
        "The uploaded source files were not modified.",
        "Reviewed labels are recorded in labels.csv; folders reflect effective labels.",
        "Detection results are review priorities, not confirmation of poisoning.",
        "",
    ]

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True) as archive:
        for record in dataset.images:
            decision = store.get(record.sample_id)
            if decision is not None and decision.action == "remove":
                removed += 1
                continue
            effective_label = store.effective_label(record.sample_id, record.label)
            if (
                decision is not None
                and decision.action == "approve_correction"
                and effective_label is not None
                and str(effective_label) != str(record.label)
            ):
                corrected += 1
            if decision is not None and decision.action == "uncertain":
                uncertain += 1
            folder = _safe_folder(effective_label)
            file_name = _unique_name(Path(record.path).name, used_names)
            arcname = f"cleaned/{folder}/{file_name}"
            try:
                archive.write(record.path, arcname=arcname)
            except OSError as exc:
                bundle.warnings.append(f"Could not add {record.rel_path} to the cleaned archive: {exc}")
                continue
            rows.append(
                {
                    "sample_id": record.sample_id,
                    "cleaned_path": arcname,
                    "original_path": record.rel_path,
                    "label": "" if effective_label is None else str(effective_label),
                    "action": decision.action if decision else "",
                    "original_label": "" if record.label is None else str(record.label),
                    "suggested_label": (
                        str(decision.suggested_label) if decision and decision.suggested_label is not None else ""
                    ),
                    "notes": decision.notes if decision else "",
                    "reviewer": decision.reviewer if decision else "",
                    "timestamp": decision.timestamp if decision else "",
                    "readable": bool(record.readable),
                    "width": record.width,
                    "height": record.height,
                    "format": record.format,
                }
            )

        labels_frame = pd.DataFrame(rows)
        readme_lines.append(f"Included images: {len(labels_frame):,}")
        readme_lines.append(f"Removed images: {removed:,}")
        readme_lines.append(f"Corrected labels: {corrected:,}")
        archive.writestr("cleaned/labels.csv", labels_frame.to_csv(index=False).encode("utf-8-sig"))
        archive.writestr("cleaned/README.txt", "\n".join(readme_lines).encode("utf-8"))

    notes = [
        "The original uploaded images were not modified.",
        "Every cleaned image is copied into a new ZIP archive; labels.csv records provenance and decisions.",
        "Unreadable files are included when they were not explicitly removed, but marked readable=false.",
    ]
    if not store.decisions:
        notes.append("No review decisions were recorded; the cleaned archive preserves all original labels.")
    return SanitisationResult(
        modality="image",
        n_original=len(dataset.images),
        n_included=len(rows),
        n_removed=removed,
        n_corrected=corrected,
        n_uncertain=uncertain,
        changes=changes,
        archive_path=output_path,
        labels_frame=labels_frame,
        output_path=output_path,
        notes=notes,
    )


def write_sanitised_tabular(df: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        df.to_excel(path, index=False, engine="openpyxl")
    else:
        df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def write_changes(changes: pd.DataFrame, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    changes.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def format_size(path: Path | None) -> str:
    if path is None or not Path(path).exists():
        return "n/a"
    return human_bytes(Path(path).stat().st_size)
