"""Small shared helpers (filesystem, hashing, JSON, metrics, progress)."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
import zipfile
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import numpy as np
import pandas as pd

from .config import ensure_workspace


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def new_session_id() -> str:
    return f"ps_{timestamp_slug()}_{os.urandom(3).hex()}"


def safe_slug(value: str, fallback: str = "dataset") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return cleaned[:80] or fallback


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def human_bytes(num: float | int | None) -> str:
    if num is None or (isinstance(num, float) and math.isnan(num)):
        return "n/a"
    value = float(num)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(value) < 1024.0 or unit == "TB":
            return f"{value:,.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024.0
    return f"{value:.1f} TB"


def shannon_entropy(probabilities: Iterable[float]) -> float:
    values = np.asarray(list(probabilities), dtype=float)
    values = values[values > 0]
    if values.size == 0:
        return 0.0
    values = values / values.sum()
    return float(-(values * np.log2(values)).sum())


def json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp, datetime)):
        return obj.isoformat()
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return sorted(obj)
    return str(obj)


def json_safe(value: Any) -> Any:
    """Round-trip through the JSON encoder to guarantee serialisability."""
    return json.loads(json.dumps(value, default=json_default))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, ensure_ascii=False)


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default


def frame_records(frame: pd.DataFrame, limit: int | None = None) -> list[dict[str, Any]]:
    """Convert a dataframe to JSON-safe records (NaN -> None)."""
    if frame is None:
        return []
    view = frame.head(limit) if limit is not None else frame
    view = view.replace({np.nan: None})
    records: list[dict[str, Any]] = []
    for record in view.to_dict(orient="records"):
        records.append({str(k): json_default(v) if not isinstance(v, (str, int, float, bool, type(None))) else v for k, v in record.items()})
    return records


def safe_extract_zip(archive: Path, destination: Path, max_files: int = 20_000) -> Path:
    """Extract *archive* into *destination* with Zip-Slip and size guards.

    Returns the directory that logically contains the dataset. If the archive
    wraps everything in a single top-level folder, that folder is returned.
    """

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    extracted = 0
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            if member.is_dir():
                continue
            extracted += 1
            if extracted > max_files:
                raise ValueError(f"Archive contains more than {max_files:,} files.")
            name = member.filename.replace("\\", "/")
            if name.startswith("__MACOSX/") or "/._" in name or name.startswith("."):
                continue
            target = (destination / name).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise ValueError(f"Unsafe archive path: {member.filename}") from exc
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, open(target, "wb") as sink:
                while True:
                    chunk = source.read(1 << 20)
                    if not chunk:
                        break
                    sink.write(chunk)

    entries = [p for p in destination.iterdir() if not p.name.startswith(".")]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return destination


@contextmanager
def timer() -> Iterator[dict[str, float]]:
    record: dict[str, float] = {}
    started = time.perf_counter()
    try:
        yield record
    finally:
        record["seconds"] = time.perf_counter() - started


def try_psutil_rss_mb() -> float | None:
    try:
        import psutil  # type: ignore

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except Exception:
        return None


class PeakMemory:
    """Context manager recording peak traced Python memory and (if available) RSS."""

    def __init__(self) -> None:
        self.peak_mb: float | None = None
        self.rss_mb: float | None = None
        self._tracemalloc = None

    def __enter__(self) -> "PeakMemory":
        import tracemalloc

        self._tracemalloc = tracemalloc
        tracemalloc.start()
        self.rss_mb = try_psutil_rss_mb()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        assert self._tracemalloc is not None
        _, peak = self._tracemalloc.get_traced_memory()
        self.peak_mb = peak / (1024 * 1024)
        self._tracemalloc.stop()


ProgressCallback = Callable[[float, str], None]


def noop_progress(_fraction: float, _message: str) -> None:
    return None


def make_progress(progress: ProgressCallback | None) -> ProgressCallback:
    return progress or noop_progress


def format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    if seconds < 1:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60:
        return f"{seconds:.1f} s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m {rest:.0f}s"


def stable_sample_ids(n: int, prefix: str = "S") -> list[str]:
    width = max(6, len(str(n)))
    return [f"{prefix}{i + 1:0{width}d}" for i in range(n)]


def temp_file_path(suffix: str = "") -> Path:
    ensure_workspace()
    handle, name = tempfile.mkstemp(suffix=suffix)
    os.close(handle)
    return Path(name)


def pretty_issue_name(issue: str) -> str:
    return {
        "label": "Label issue",
        "outlier": "Outlier",
        "near_duplicate": "Near duplicate",
        "class_imbalance": "Class imbalance",
        "non_iid": "Non-IID structure",
        "low_information": "Low-information image",
        "integrity": "Data integrity",
        "relationship": "Neighbourhood conflict",
    }.get(issue, issue.replace("_", " ").title())


def path_fingerprint(path: Path) -> str | None:
    """Stable SHA-256 fingerprint for a file or a directory tree.

    Directories are fingerprinted by relative name and per-file SHA-256 so that
    uploads made of individual images can still be checked for accidental
    modification.
    """

    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix()
        digest.update(relative.encode("utf-8", errors="replace"))
        digest.update(b"\x00")
        digest.update(sha256_file(child).encode("ascii"))
        digest.update(b"\x00")
    return digest.hexdigest()

