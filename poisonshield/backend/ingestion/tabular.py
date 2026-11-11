"""Tabular ingestion (CSV / XLSX / XLS) and profiling."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import (
    LABEL_HINTS,
    MAX_TABULAR_COLUMNS,
    MAX_TABULAR_ROWS,
    MISSING_SENTINELS,
    SUPPORTED_TABULAR_EXTENSIONS,
)
from ..types import DatasetProfile, LoadedDataset
from ..utils import sha256_file, stable_sample_ids, utc_now_iso


class TabularLoadError(ValueError):
    """Raised when an uploaded table cannot be used safely."""


def read_tabular(path: Path, max_rows: int = MAX_TABULAR_ROWS) -> pd.DataFrame:
    """Read a tabular file with encoding fallbacks and defensive limits.

    The returned dataframe is a fresh in-memory copy. The file on disk is only
    read, never written.
    """

    path = Path(path)
    if not path.exists():
        raise TabularLoadError(f"File not found: {path}")
    if path.suffix.lower() not in SUPPORTED_TABULAR_EXTENSIONS:
        raise TabularLoadError(
            "Unsupported tabular format "
            f"'{path.suffix}'. Use CSV, XLSX or XLS."
        )

    frame: pd.DataFrame | None = None
    errors: list[str] = []

    if path.suffix.lower() == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
            try:
                frame = pd.read_csv(
                    path,
                    encoding=encoding,
                    sep=None,
                    engine="python",
                    nrows=max_rows + 1,
                )
                break
            except Exception as exc:  # noqa: BLE001 - try the next encoding
                errors.append(f"{encoding}: {exc}")
        if frame is None:
            # Last resort: comma-separated.
            try:
                frame = pd.read_csv(path, encoding="utf-8", nrows=max_rows + 1)
            except Exception as exc:  # noqa: BLE001
                raise TabularLoadError(
                    "Could not read the CSV file. " + " | ".join(errors + [str(exc)])
                ) from exc
    else:
        try:
            frame = pd.read_excel(path, nrows=max_rows + 1)
        except ImportError as exc:
            raise TabularLoadError(
                "Reading Excel files requires openpyxl (XLSX) or xlrd (XLS). "
                "Install them with: pip install openpyxl xlrd"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise TabularLoadError(f"Could not read the Excel workbook: {exc}") from exc

    if frame is None or frame.empty:
        raise TabularLoadError("The table contains no rows or could not be parsed.")
    if len(frame) > max_rows:
        raise TabularLoadError(
            f"Dataset has more than the configured limit of {max_rows:,} rows. "
            "Increase the row limit in Analysis settings if your machine can handle it."
        )
    if frame.shape[1] > MAX_TABULAR_COLUMNS:
        raise TabularLoadError(
            f"Dataset has {frame.shape[1]:,} columns; the supported limit is "
            f"{MAX_TABULAR_COLUMNS:,}."
        )
    if frame.shape[1] == 0:
        raise TabularLoadError("No columns were detected in the table.")

    frame.columns = [str(c).strip() if str(c).strip() else f"column_{i + 1}" for i, c in enumerate(frame.columns)]
    # Keep duplicate column names distinguishable.
    seen: dict[str, int] = {}
    columns: list[str] = []
    for column in frame.columns:
        if column in seen:
            seen[column] += 1
            columns.append(f"{column}__{seen[column]}")
        else:
            seen[column] = 0
            columns.append(column)
    frame.columns = columns
    frame = frame.reset_index(drop=True)
    return frame


def _unique_ratio(series: pd.Series) -> float:
    non_null = int(series.notna().sum())
    if non_null == 0:
        return 0.0
    return float(series.nunique(dropna=True) / non_null)


def detect_id_like_columns(frame: pd.DataFrame) -> list[str]:
    """Columns that identify rows rather than describe them."""

    detected: list[str] = []
    n = len(frame)
    for column in frame.columns:
        series = frame[column]
        if n < 10:
            continue
        ratio = _unique_ratio(series)
        dtype = series.dtype
        if ratio < 0.98:
            continue
        if pd.api.types.is_numeric_dtype(dtype):
            # A monotonic sequence of (mostly) unique numbers is an identifier.
            values = pd.to_numeric(series, errors="coerce").dropna()
            if len(values) == 0:
                continue
            monotonic = values.is_monotonic_increasing or values.is_monotonic_decreasing
            integerish = bool(np.all(np.equal(np.mod(values, 1), 0)))
            if monotonic and integerish:
                detected.append(column)
        else:
            detected.append(column)
    return detected


def _coercible_to_numeric(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series):
        return False
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return False
    cleaned = non_null.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    parsed = pd.to_numeric(cleaned, errors="coerce")
    return float(parsed.notna().mean()) >= 0.95


def _numeric_like_columns(frame: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in exclude and _coercible_to_numeric(frame[column])
    ]


def suggest_label_column(frame: pd.DataFrame) -> str | None:
    """Rank plausible target columns without silently choosing one."""

    if frame.shape[1] == 0:
        return None
    n = len(frame)
    id_like = set(detect_id_like_columns(frame))
    candidates: list[tuple[float, str]] = []
    for column in frame.columns:
        if column in id_like:
            continue
        series = frame[column]
        non_null = int(series.notna().sum())
        if non_null < 2:
            continue
        unique = int(series.nunique(dropna=True))
        if unique < 2:
            continue
        ratio = unique / max(non_null, 1)
        # Targets are usually low-cardinality for classification but may be
        # continuous for regression. Avoid near-unique text or float IDs.
        if ratio > 0.85:
            continue
        if unique > max(200, int(0.5 * non_null)) and not pd.api.types.is_numeric_dtype(series):
            continue
        score = 0.0
        lowered = column.strip().lower()
        if lowered in LABEL_HINTS:
            score += 5.0
        elif any(hint in lowered for hint in LABEL_HINTS):
            score += 2.5
        if pd.api.types.is_numeric_dtype(series):
            score += 0.4
        if 2 <= unique <= 30:
            score += 1.5
        elif 31 <= unique <= 200:
            score += 0.6
        score -= float(series.isna().mean()) * 2.0
        if score > 0:
            candidates.append((score, column))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def infer_task(frame: pd.DataFrame, label_column: str | None) -> str:
    if not label_column or label_column not in frame.columns:
        return "unlabeled"
    series = frame[label_column]
    non_null = series.dropna()
    if len(non_null) == 0:
        return "unlabeled"
    unique = int(non_null.nunique())
    if unique < 2:
        return "unlabeled"
    if pd.api.types.is_float_dtype(series) or (
        pd.api.types.is_numeric_dtype(series) and unique > max(20, int(0.05 * len(non_null)))
    ):
        return "regression"
    return "classification"


def _safe_value(value: Any) -> Any:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if np.isnan(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return str(value)


def _column_stats(frame: pd.DataFrame, roles: dict[str, str]) -> list[dict[str, Any]]:
    stats: list[dict[str, Any]] = []
    n = max(len(frame), 1)
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        unique = int(series.nunique(dropna=True))
        entry: dict[str, Any] = {
            "column": column,
            "role": roles.get(column, "feature"),
            "dtype": str(series.dtype),
            "missing": missing,
            "missing_pct": round(100.0 * missing / n, 2),
            "unique": unique,
            "top_value": None,
        }
        if unique > 0 and not pd.api.types.is_numeric_dtype(series):
            counts = series.astype(str).value_counts(dropna=True)
            if len(counts):
                entry["top_value"] = _safe_value(counts.index[0])
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            finite = numeric[np.isfinite(numeric)]
            if len(finite):
                entry.update(
                    min=float(finite.min()),
                    max=float(finite.max()),
                    mean=float(finite.mean()),
                    std=float(finite.std(ddof=0)),
                )
            entry["infinite_values"] = int((~np.isfinite(numeric) & numeric.notna()).sum())
        stats.append(entry)
    return stats


def profile_tabular(
    frame: pd.DataFrame,
    sample_ids: list[str],
    label_column: str | None,
    task: str,
) -> DatasetProfile:
    n = len(frame)
    numeric_cols = [
        c for c in frame.columns
        if pd.api.types.is_numeric_dtype(frame[c]) and not pd.api.types.is_bool_dtype(frame[c])
    ]
    bool_cols = [c for c in frame.columns if pd.api.types.is_bool_dtype(frame[c])]
    datetime_cols = [c for c in frame.columns if pd.api.types.is_datetime64_any_dtype(frame[c])]
    categorical_cols = [
        c for c in frame.columns
        if c not in numeric_cols + bool_cols + datetime_cols
    ]
    id_like = detect_id_like_columns(frame)
    numeric_coercions = _numeric_like_columns(frame, exclude=set(id_like + ([label_column] if label_column else [])))

    roles: dict[str, str] = {}
    for column in frame.columns:
        if column == label_column:
            roles[column] = "label"
        elif column in id_like:
            roles[column] = "identifier"
        elif column in numeric_cols:
            roles[column] = "numeric feature"
        elif column in categorical_cols:
            roles[column] = "categorical feature"
        else:
            roles[column] = "feature"

    missing_cells = int(frame.isna().sum().sum())
    duplicate_mask = frame.duplicated(keep="first")
    duplicate_rows = int(duplicate_mask.sum())
    duplicate_ids = [sample_ids[i] for i in np.flatnonzero(duplicate_mask.to_numpy())]

    constant_columns: list[str] = []
    all_missing_columns: list[str] = []
    high_missing: dict[str, float] = {}
    high_cardinality: dict[str, int] = {}
    invalid_counts: dict[str, int] = {}
    for column in frame.columns:
        series = frame[column]
        missing = int(series.isna().sum())
        if missing == n:
            all_missing_columns.append(column)
        if series.nunique(dropna=True) <= 1 and missing < n:
            constant_columns.append(column)
        pct = 100.0 * missing / max(n, 1)
        if pct >= 40.0:
            high_missing[column] = round(pct, 2)
        unique = int(series.nunique(dropna=True))
        if column in categorical_cols and unique > min(200, max(50, int(0.5 * n))):
            high_cardinality[column] = unique
        if pd.api.types.is_numeric_dtype(series):
            numeric = pd.to_numeric(series, errors="coerce")
            infinite = int((~np.isfinite(numeric) & numeric.notna()).sum())
            if infinite:
                invalid_counts[column] = infinite
        elif column not in datetime_cols:
            sentinel_count = int(
                series.dropna().astype(str).str.strip().str.lower().isin(MISSING_SENTINELS).sum()
            )
            if sentinel_count:
                invalid_counts[column] = sentinel_count

    class_distribution: dict[str, int] = {}
    if label_column and label_column in frame.columns:
        counts = frame[label_column].value_counts(dropna=False)
        for key, value in counts.items():
            label = "<missing>" if pd.isna(key) else str(key)
            class_distribution[label] = int(value)

    profile = DatasetProfile(
        modality="tabular",
        n_samples=n,
        n_columns=int(frame.shape[1]),
        label_column=label_column,
        task=task,
        class_distribution=class_distribution,
        missing_cells=missing_cells,
        missing_cell_pct=round(100.0 * missing_cells / max(n * max(frame.shape[1], 1), 1), 3),
        duplicate_rows=duplicate_rows,
        duplicate_sample_ids=duplicate_ids[:5_000],
        numeric_columns=numeric_cols + bool_cols,
        categorical_columns=categorical_cols,
        datetime_columns=datetime_cols,
        constant_columns=constant_columns,
        all_missing_columns=all_missing_columns,
        high_missing_columns=high_missing,
        high_cardinality_columns=high_cardinality,
        id_like_columns=id_like,
        numeric_coercions=numeric_coercions,
        invalid_value_counts=invalid_counts,
        column_stats=_column_stats(frame, roles),
        generated_at=utc_now_iso(),
    )
    return profile


def build_tabular_dataset(
    frame: pd.DataFrame,
    source_name: str,
    label_column: str | None,
    task: str | None = None,
    original_copy: Path | None = None,
) -> LoadedDataset:
    frame = frame.copy().reset_index(drop=True)
    if frame.empty:
        raise TabularLoadError("The table contains no rows.")

    if label_column == "":
        label_column = None
    if label_column and label_column not in frame.columns:
        raise TabularLoadError(f"Label column '{label_column}' is not present in the dataset.")

    resolved_task = infer_task(frame, label_column) if task in (None, "auto") else task
    if resolved_task == "unlabeled":
        label_column = None
    if resolved_task == "classification" and label_column is None:
        resolved_task = "unlabeled"

    sample_ids = stable_sample_ids(len(frame), prefix="S")
    profile = profile_tabular(frame, sample_ids, label_column, resolved_task)
    notes: list[str] = []
    if profile.duplicate_rows:
        notes.append(f"{profile.duplicate_rows:,} exact duplicate row(s) detected.")
    if profile.numeric_coercions:
        notes.append(
            "Numeric-looking text columns will be parsed for feature analysis: "
            + ", ".join(profile.numeric_coercions)
        )
    if profile.constant_columns:
        notes.append("Constant columns are excluded from feature analysis: " + ", ".join(profile.constant_columns))

    return LoadedDataset(
        modality="tabular",
        source_name=source_name,
        sample_ids=sample_ids,
        profile=profile,
        dataframe=frame,
        label_column=label_column,
        task=resolved_task,
        has_labels=bool(label_column),
        original_copy=original_copy,
        extra_columns=[c for c in frame.columns if c != label_column],
        notes=notes,
    )
