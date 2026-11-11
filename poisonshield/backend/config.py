"""Central configuration for PoisonShield.

All tunable defaults live here so that behaviour is explicit, documented and
reproducible. The UI may override a subset (risk weights, thresholds, K,
extractor, limits) per analysis session; every override is recorded in the
session configuration that is embedded in exports.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

APP_NAME = "PoisonShield"
APP_TITLE = "An Intelligent Framework for Detecting and Mitigating Data Poisoning Attacks in Machine Learning"
TAGLINE = "Secure the Data. Protect the Model."
VERSION = "1.0.0"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = Path(
    os.environ.get("POISONSHIELD_WORKSPACE", str(PROJECT_ROOT / ".poisonshield"))
).resolve()

# Runtime sub-directories. Everything PoisonShield writes lives here; uploaded
# source files are copied in unchanged and are never modified afterwards.
SESSIONS_DIR = WORKSPACE_ROOT / "sessions"
CACHE_DIR = WORKSPACE_ROOT / "cache"
EXPORT_DIR = WORKSPACE_ROOT / "exports"
UPLOAD_DIR = WORKSPACE_ROOT / "uploads"

RANDOM_SEED = 42

# --- Processing limits -----------------------------------------------------
MAX_TABULAR_ROWS = 200_000
MAX_TABULAR_COLUMNS = 1_000
MAX_IMAGE_COUNT = 3_000
MAX_UPLOAD_MB = 1_024
MAX_EMBEDDING_BATCH = 32
IMAGE_PIXEL_LIMIT = 80_000_000  # decompression-bomb guard (~80 MP)

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif",
}
SUPPORTED_TABULAR_EXTENSIONS = {".csv", ".xlsx", ".xls"}

LABEL_HINTS = (
    "label", "labels", "class", "target", "y", "category", "cat",
    "diagnosis", "outcome", "is_poisoned", "poisoned",
)

# Common textual missing-value sentinels found in CSV exports. These are
# reported and treated as missing for *feature* preparation only; the original
# dataframe is kept untouched for display and export.
MISSING_SENTINELS = {"", "na", "n/a", "null", "none", "nan", "?", "unknown"}


@dataclass
class RiskWeights:
    """Weights of the independent evidence groups in the unified risk score.

    Signals inside the label group (Cleanlab label quality + neighbourhood
    disagreement) are correlated, so they are fused non-additively instead of
    being counted twice. Outlier, duplicate and integrity groups are separate,
    non-overlapping evidence families.

    All weights must be non-negative; they are normalised to sum to 1 at score
    time and the normalised values are recorded with the results.
    """

    label_evidence: float = 0.50
    outlier_evidence: float = 0.20
    duplicate_evidence: float = 0.10
    integrity_evidence: float = 0.20

    def validated(self) -> "RiskWeights":
        values = asdict(self)
        for key, value in values.items():
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"Risk weight '{key}' must be a non-negative number.")
        total = sum(values.values())
        if total <= 0:
            raise ValueError("At least one risk weight must be greater than zero.")
        return RiskWeights(**{k: v / total for k, v in values.items()})

    def to_dict(self) -> dict[str, float]:
        return self.validated().__dict__.copy()


@dataclass
class RiskThresholds:
    """Upper bounds for the Low / Moderate / High categories.

    A score above ``high_max`` is Critical. Bounds must increase and lie in
    (0, 1); they are validated before every scoring run.
    """

    low_max: float = 0.30
    moderate_max: float = 0.55
    high_max: float = 0.75

    def validated(self) -> "RiskThresholds":
        values = [self.low_max, self.moderate_max, self.high_max]
        if any(not (0.0 < v < 1.0) for v in values):
            raise ValueError("Risk thresholds must lie strictly between 0 and 1.")
        if not (values[0] < values[1] < values[2]):
            raise ValueError("Risk thresholds must satisfy low < moderate < high.")
        return self


@dataclass
class AnalysisConfig:
    """Reproducible settings for one analysis run."""

    k_neighbors: int = 15
    random_seed: int = RANDOM_SEED
    task: str = "auto"  # auto | classification | regression | unlabeled
    image_extractor: str = "auto"  # auto | dinov2_vits14 | dinov2_vitb14 | resnet50 | handcrafted
    image_batch_size: int = MAX_EMBEDDING_BATCH
    max_images: int = MAX_IMAGE_COUNT
    max_rows: int = MAX_TABULAR_ROWS
    label_column: str | None = None
    ground_truth_column: str | None = None
    svd_components: int = 64
    max_onehot_categories: int = 20
    neighbor_agreement_threshold: float = 0.50
    risk_weights: RiskWeights = field(default_factory=RiskWeights)
    risk_thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    run_low_information_check: bool = True

    def validated(self) -> "AnalysisConfig":
        if self.k_neighbors < 1:
            raise ValueError("k_neighbors must be at least 1.")
        if self.task not in {"auto", "classification", "regression", "unlabeled"}:
            raise ValueError(f"Unsupported task '{self.task}'.")
        if not (0.0 <= self.neighbor_agreement_threshold <= 1.0):
            raise ValueError("neighbor_agreement_threshold must be between 0 and 1.")
        if self.image_batch_size < 1:
            raise ValueError("image_batch_size must be positive.")
        self.risk_weights = self.risk_weights.validated()
        self.risk_thresholds = self.risk_thresholds.validated()
        return self

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class AppSettings:
    """Process-wide defaults surfaced in the UI."""

    max_upload_mb: int = MAX_UPLOAD_MB
    page_title: str = APP_NAME


def ensure_workspace() -> None:
    for path in (WORKSPACE_ROOT, SESSIONS_DIR, CACHE_DIR, EXPORT_DIR, UPLOAD_DIR):
        path.mkdir(parents=True, exist_ok=True)


def risk_config_from_dict(weights: dict[str, float], thresholds: dict[str, float]) -> tuple[RiskWeights, RiskThresholds]:
    """Build validated risk settings from plain JSON/UI dictionaries."""

    return (
        RiskWeights(**weights).validated(),
        RiskThresholds(**thresholds).validated(),
    )
