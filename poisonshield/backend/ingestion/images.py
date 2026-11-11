"""Image ingestion: safe loading, validation, label mapping and profiling."""

from __future__ import annotations

import io
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

from ..config import (
    IMAGE_PIXEL_LIMIT,
    MAX_IMAGE_COUNT,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from ..types import DatasetProfile, ImageRecord, LoadedDataset
from ..utils import (
    safe_extract_zip,
    sha256_bytes,
    sha256_file,
    shannon_entropy,
    stable_sample_ids,
    utc_now_iso,
)

PATH_COLUMN_CANDIDATES = (
    "path", "file", "filename", "file_name", "image", "image_path", "img",
    "img_path", "rel_path", "relative_path", "name",
)
LABEL_COLUMN_CANDIDATES = (
    "label", "labels", "class", "class_name", "target", "category", "cat", "y",
)
GROUND_TRUTH_CANDIDATES = (
    "is_poisoned", "poisoned", "is_corrupted", "corrupted", "is_corrupt",
    "ground_truth", "is_mislabeled", "mislabeled", "is_noisy", "noisy",
    "is_poison", "attack", "poison_label",
)
LOW_INFORMATION_STD = 6.0
LOW_INFORMATION_ENTROPY = 1.8


class ImageLoadError(ValueError):
    """Raised when an image dataset cannot be used safely."""


def find_image_files(root: Path) -> list[Path]:
    root = Path(root)
    if not root.exists():
        raise ImageLoadError(f"Image folder not found: {root}")
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        and not any(part.startswith(".") or part == "__MACOSX" for part in path.relative_to(root).parts)
    ]
    return sorted(files, key=lambda p: str(p).lower())


def _normalise_key(value: str) -> str:
    text = str(value).replace("\\", "/").strip()
    while text.startswith("./"):
        text = text[2:]
    return text.lower()


def resolve_mapping_columns(
    mapping: pd.DataFrame,
    path_column: str | None = None,
    label_column: str | None = None,
) -> tuple[str, str]:
    if mapping.empty:
        raise ImageLoadError("The label mapping file is empty.")

    def pick(candidates: Iterable[str], explicit: str | None, kind: str) -> str:
        if explicit:
            if explicit not in mapping.columns:
                raise ImageLoadError(f"Column '{explicit}' is not present in the mapping file.")
            return explicit
        lookup = {str(c).strip().lower(): c for c in mapping.columns}
        for candidate in candidates:
            if candidate in lookup:
                return lookup[candidate]
        raise ImageLoadError(
            f"Could not identify the {kind} column in the mapping file. "
            "Use a column named path/file/filename for the image reference "
            "and label/class/target for the label."
        )

    return (
        pick(PATH_COLUMN_CANDIDATES, path_column, "image path"),
        pick(LABEL_COLUMN_CANDIDATES, label_column, "label"),
    )


def suggest_image_label_column(mapping: pd.DataFrame) -> str | None:
    try:
        _, label_column = resolve_mapping_columns(mapping)
        return str(label_column)
    except ImageLoadError:
        candidates = [c for c in mapping.columns if str(c).strip().lower() in LABEL_COLUMN_CANDIDATES]
        return str(candidates[0]) if candidates else None


def suggest_ground_truth_column(columns: Iterable[str]) -> str | None:
    lookup = {str(c).strip().lower(): str(c) for c in columns}
    for candidate in GROUND_TRUTH_CANDIDATES:
        if candidate in lookup:
            return lookup[candidate]
    for key, value in lookup.items():
        if "poison" in key or "corrupt" in key or "mislabel" in key:
            return value
    return None


def _mapping_index(
    mapping: pd.DataFrame,
    path_column: str,
    label_column: str,
    extra_columns: list[str],
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for row in mapping.to_dict(orient="records"):
        raw_key = row.get(path_column)
        if raw_key is None or (isinstance(raw_key, float) and np.isnan(raw_key)):
            continue
        key = _normalise_key(raw_key)
        label_value = row.get(label_column)
        label = None if label_value is None or (isinstance(label_value, float) and np.isnan(label_value)) else str(label_value)
        extras: dict[str, Any] = {}
        for column in extra_columns:
            value = row.get(column)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                extras[str(column)] = None
            else:
                extras[str(column)] = value.item() if hasattr(value, "item") else value
        payload = {"label": label, "extras": extras, "key": key}
        index.setdefault(key, []).append(payload)
        basename = _normalise_key(Path(key).name)
        if basename != key:
            index.setdefault(basename, []).append(payload)
    return index


def _match_mapping(
    rel_path: str,
    index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any] | None:
    key = _normalise_key(rel_path)
    for candidate in (key, _normalise_key(Path(key).name)):
        matches = index.get(candidate, [])
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            # Prefer an exact relative-path match; otherwise ambiguous basenames
            # are ignored rather than silently mislabelled.
            exact = [m for m in matches if m["key"] == key]
            return exact[0] if len(exact) == 1 else None
    return None


def _inspect_image(path: Path) -> dict[str, Any]:
    info: dict[str, Any] = {
        "readable": True,
        "error": None,
        "width": None,
        "height": None,
        "format": None,
        "brightness_std": None,
        "entropy": None,
    }
    try:
        with Image.open(path) as img:
            width, height = img.size
            info["width"] = int(width)
            info["height"] = int(height)
            info["format"] = img.format
            if width * height > IMAGE_PIXEL_LIMIT:
                info.update(readable=False, error=f"Image exceeds the {IMAGE_PIXEL_LIMIT:,}-pixel safety limit.")
                return info
            img.load()
            thumb = img.convert("RGB")
            thumb.thumbnail((64, 64))
            gray = np.asarray(thumb.convert("L"), dtype=np.float32)
            if gray.size:
                info["brightness_std"] = float(gray.std())
                histogram = np.bincount(gray.astype(np.uint8).ravel(), minlength=256).astype(float)
                info["entropy"] = float(shannon_entropy(histogram))
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        info.update(readable=False, error=f"{type(exc).__name__}: {exc}")
    except Exception as exc:  # noqa: BLE001 - never let one file break ingestion
        info.update(readable=False, error=f"{type(exc).__name__}: {exc}")
    return info


def _enumerate(root: Path, max_images: int) -> list[Path]:
    files = find_image_files(root)
    if not files:
        raise ImageLoadError(
            "No supported image files were found. Supported formats: "
            + ", ".join(sorted(SUPPORTED_IMAGE_EXTENSIONS))
        )
    if len(files) > max_images:
        raise ImageLoadError(
            f"Found {len(files):,} images, above the configured limit of {max_images:,}. "
            "Increase the limit in Analysis settings or upload a smaller subset."
        )
    return files


def _records_from_files(
    files: list[Path],
    root: Path,
    mapping: pd.DataFrame | None,
    path_column: str | None,
    label_column: str | None,
    infer_folder_labels: bool,
) -> tuple[list[ImageRecord], list[str], list[str]]:
    warnings: list[str] = []
    mapping_index: dict[str, list[dict[str, Any]]] = {}
    extra_columns: list[str] = []
    if mapping is not None and not mapping.empty:
        resolved_path, resolved_label = resolve_mapping_columns(mapping, path_column, label_column)
        extra_columns = [str(c) for c in mapping.columns if c not in (resolved_path, resolved_label)]
        mapping_index = _mapping_index(mapping, resolved_path, resolved_label, extra_columns)
        if not mapping_index:
            warnings.append("The label mapping file did not contain usable image-path rows.")

    records: list[ImageRecord] = []
    hash_to_sample: dict[str, str] = {}
    unmatched = 0
    for position, path in enumerate(files, start=1):
        rel_path = path.relative_to(root).as_posix()
        sample_id = f"IMG{position:06d}"
        info = _inspect_image(path)
        digest = sha256_file(path)
        label: str | None = None
        extras: dict[str, Any] = {}
        match = _match_mapping(rel_path, mapping_index) if mapping_index else None
        if match:
            label = match["label"]
            extras = dict(match["extras"])
        elif mapping_index:
            unmatched += 1
        if label is None and infer_folder_labels:
            parents = Path(rel_path).parts[:-1]
            if parents:
                label = parents[-1]
        duplicate_of = hash_to_sample.get(digest)
        if duplicate_of is None:
            hash_to_sample[digest] = sample_id
        record = ImageRecord(
            sample_id=sample_id,
            path=str(path),
            rel_path=rel_path,
            label=label,
            width=info["width"],
            height=info["height"],
            format=info["format"],
            size_bytes=path.stat().st_size,
            readable=info["readable"],
            error=info["error"],
            sha256=digest,
            exact_duplicate_of=duplicate_of,
            brightness_std=info["brightness_std"],
            entropy=info["entropy"],
            extra=extras,
        )
        if not record.readable:
            warnings.append(f"{rel_path}: unreadable ({record.error})")
        extension = path.suffix.lower().lstrip(".")
        container = str(info["format"]).lower() if info["format"] else ""
        equivalent = {
            "jpg": {"jpg", "jpeg"},
            "jpeg": {"jpg", "jpeg"},
            "tif": {"tif", "tiff"},
            "tiff": {"tif", "tiff"},
        }.get(extension, {extension})
        if container and container not in equivalent:
            record.extra.setdefault(
                "format_mismatch", f"extension={path.suffix.lower()} container={info['format']}"
            )
        records.append(record)

    if unmatched:
        warnings.append(
            f"{unmatched:,} image(s) had no matching row in the label mapping file "
            "and were treated as unlabelled (folder names used when available)."
        )
    if records and all(r.label is None for r in records):
        warnings.append("No image labels were available; dataset will be analysed as unlabelled.")
    return records, warnings, extra_columns


def _check_zip_image_count(archive: Path, max_images: int) -> None:
    import zipfile

    with zipfile.ZipFile(archive) as bundle:
        count = sum(
            1
            for member in bundle.infolist()
            if not member.is_dir()
            and Path(member.filename).suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        )
    if count > max_images:
        raise ImageLoadError(
            f"Archive contains {count:,} images, above the configured limit of {max_images:,}."
        )


def load_image_zip(
    archive: Path,
    destination: Path,
    max_images: int = MAX_IMAGE_COUNT,
    mapping: pd.DataFrame | None = None,
    path_column: str | None = None,
    label_column: str | None = None,
    infer_folder_labels: bool = True,
) -> tuple[list[ImageRecord], list[str], list[str], Path]:
    archive = Path(archive)
    _check_zip_image_count(archive, max_images)
    root = safe_extract_zip(archive, destination)
    records, warnings, extra_columns = _records_from_files(
        _enumerate(root, max_images), root, mapping, path_column, label_column, infer_folder_labels
    )
    return records, warnings, extra_columns, root


def load_image_folder(
    root: Path,
    max_images: int = MAX_IMAGE_COUNT,
    mapping: pd.DataFrame | None = None,
    path_column: str | None = None,
    label_column: str | None = None,
    infer_folder_labels: bool = True,
) -> tuple[list[ImageRecord], list[str], list[str]]:
    root = Path(root).resolve()
    records, warnings, extra_columns = _records_from_files(
        _enumerate(root, max_images), root, mapping, path_column, label_column, infer_folder_labels
    )
    return records, warnings, extra_columns


def load_uploaded_images(
    uploads: list[tuple[str, bytes]],
    destination: Path,
    max_images: int = MAX_IMAGE_COUNT,
    mapping: pd.DataFrame | None = None,
    path_column: str | None = None,
    label_column: str | None = None,
    infer_folder_labels: bool = False,
) -> tuple[list[ImageRecord], list[str], list[str]]:
    if len(uploads) > max_images:
        raise ImageLoadError(
            f"{len(uploads):,} images uploaded, above the configured limit of {max_images:,}."
        )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    used: set[str] = set()
    for name, payload in uploads:
        candidate = Path(name.replace("\\", "/")).name
        if Path(candidate).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        stem, suffix = Path(candidate).stem, Path(candidate).suffix
        safe_name = f"{stem}{suffix}"
        counter = 1
        while safe_name.lower() in used:
            counter += 1
            safe_name = f"{stem}_{counter}{suffix}"
        used.add(safe_name.lower())
        target = (destination / safe_name).resolve()
        try:
            target.relative_to(destination.resolve())
        except ValueError as exc:
            raise ImageLoadError(f"Unsafe file name: {name}") from exc
        target.write_bytes(payload)
        saved.append(target)
    if not saved:
        raise ImageLoadError("No supported image files were presented for upload.")
    return load_image_folder(
        destination,
        max_images=max_images,
        mapping=mapping,
        path_column=path_column,
        label_column=label_column,
        infer_folder_labels=infer_folder_labels,
    )


def load_mapping_from_bytes(name: str, payload: bytes) -> pd.DataFrame:
    suffix = Path(name).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise ImageLoadError("The label mapping must be a CSV, XLSX or XLS file.")
    buffer = io.BytesIO(payload)
    if suffix == ".csv":
        try:
            return pd.read_csv(buffer, encoding="utf-8-sig")
        except UnicodeDecodeError:
            buffer.seek(0)
            return pd.read_csv(buffer, encoding="latin-1")
    return pd.read_excel(buffer)


def build_image_dataset(
    records: list[ImageRecord],
    source_name: str,
    extra_columns: list[str] | None = None,
    original_copy: Path | None = None,
    notes: list[str] | None = None,
) -> LoadedDataset:
    if not records:
        raise ImageLoadError("No images were loaded.")

    labelled = [r for r in records if r.label is not None]
    has_labels = len(labelled) >= 2 and len({r.label for r in labelled}) >= 2
    task = "classification" if has_labels else "unlabeled"
    label_column = "label" if has_labels else None

    class_distribution: dict[str, int] = {}
    for record in records:
        key = record.label if record.label is not None else "<missing>"
        class_distribution[str(key)] = class_distribution.get(str(key), 0) + 1

    readable = [r for r in records if r.readable]
    formats = Counter(str(r.format) for r in readable if r.format)
    dimensions = Counter(f"{r.width}x{r.height}" for r in readable if r.width and r.height)
    sizes = [r.size_bytes for r in records if r.size_bytes]
    duplicate_ids = [r.sample_id for r in records if r.exact_duplicate_of]
    unreadable = [r for r in records if not r.readable]
    low_information = [
        r for r in readable
        if (r.brightness_std is not None and r.brightness_std < LOW_INFORMATION_STD)
        or (r.entropy is not None and r.entropy < LOW_INFORMATION_ENTROPY)
    ]
    mismatch = [r for r in records if "format_mismatch" in r.extra]

    image_stats: dict[str, Any] = {
        "readable": len(readable),
        "unreadable": len(unreadable),
        "unreadable_examples": [
            {"sample_id": r.sample_id, "rel_path": r.rel_path, "error": r.error} for r in unreadable[:20]
        ],
        "formats": dict(formats.most_common()),
        "dimensions": dict(dimensions.most_common(15)),
        "total_bytes": int(sum(sizes)),
        "mean_bytes": float(np.mean(sizes)) if sizes else None,
        "mean_width": float(np.mean([r.width for r in readable if r.width])) if readable else None,
        "mean_height": float(np.mean([r.height for r in readable if r.height])) if readable else None,
        "exact_duplicates": len(duplicate_ids),
        "exact_duplicate_sample_ids": duplicate_ids[:5_000],
        "low_information": len(low_information),
        "low_information_sample_ids": [r.sample_id for r in low_information[:5_000]],
        "format_mismatches": len(mismatch),
        "missing_labels": len(records) - len(labelled),
    }

    profile = DatasetProfile(
        modality="image",
        n_samples=len(records),
        n_columns=len(records[0].extra) + 3 if records else 0,
        label_column=label_column,
        task=task,
        class_distribution=class_distribution,
        duplicate_rows=len(duplicate_ids),
        duplicate_sample_ids=duplicate_ids[:5_000],
        image_stats=image_stats,
        generated_at=utc_now_iso(),
    )

    notes = list(notes or [])
    if unreadable:
        notes.append(
            f"{len(unreadable):,} image(s) could not be decoded. They are retained as "
            "integrity findings and excluded from embedding-based detection."
        )
    if duplicate_ids:
        notes.append(f"{len(duplicate_ids):,} exact duplicate image file(s) detected via SHA-256.")
    if image_stats["missing_labels"]:
        notes.append(
            f"{image_stats['missing_labels']:,} image(s) have no label. Label-error detection "
            "runs on the labelled subset only."
        )
    if not has_labels:
        notes.append("Dataset is unlabelled; feature-only quality checks will be run.")

    return LoadedDataset(
        modality="image",
        source_name=source_name,
        sample_ids=[r.sample_id for r in records],
        profile=profile,
        images=records,
        label_column=label_column,
        task=task,
        has_labels=has_labels,
        original_copy=original_copy,
        extra_columns=[c for c in (extra_columns or [])],
        notes=notes,
    )
