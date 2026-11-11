"""Analysis-ready tabular representations built on the original data."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..config import MISSING_SENTINELS
from ..types import FeatureResult
from ..utils import timer


class FeatureBuildError(ValueError):
    """Raised when no meaningful feature representation can be built."""


def _is_numeric_like(series: pd.Series) -> bool:
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True
    non_null = series.dropna().astype(str).str.strip()
    if len(non_null) == 0:
        return False
    cleaned = non_null.str.replace(",", "", regex=False).str.replace("%", "", regex=False)
    return float(pd.to_numeric(cleaned, errors="coerce").notna().mean()) >= 0.95


def _coerce_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce").astype("float64")
        numeric = numeric.mask(~np.isfinite(numeric), np.nan)
        return numeric
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    if pd.api.types.is_datetime64_any_dtype(series):
        numeric = series.astype("int64").astype("float64")
        numeric[series.isna()] = np.nan
        return numeric
    cleaned = (
        series.astype(str)
        .str.strip()
        .replace(list(MISSING_SENTINELS), np.nan)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    numeric = numeric.mask(~np.isfinite(numeric), np.nan)
    return numeric


def normalise_categorical(series: pd.Series) -> pd.Series:
    """Normalise sentinel strings to missing values (feature view only)."""

    values = series.astype("object").copy()
    as_text = values.astype(str).str.strip()
    sentinel_mask = values.notna() & as_text.str.lower().isin(MISSING_SENTINELS)
    values = values.mask(sentinel_mask, np.nan)
    return values


@dataclass
class TabularFeatureBuilder:
    """Fit/transform a numeric representation while preserving row order.

    Numeric columns are median-imputed (with missing indicators), standardised
    and clipped. Categorical columns are most-frequent-imputed and one-hot
    encoded with rare-category bucketing. Datetime columns become integer
    timestamps. Numeric-looking text is parsed conservatively. Very wide
    representations are reduced with TruncatedSVD for scalable KNN search.
    """

    label_column: str | None = None
    id_like_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    all_missing_columns: list[str] = field(default_factory=list)
    excluded_columns: list[str] = field(default_factory=list)
    max_onehot_categories: int = 20
    svd_components: int = 64
    random_state: int = 42
    clip_value: float = 10.0

    numeric_columns_: list[str] = field(default_factory=list, init=False)
    categorical_columns_: list[str] = field(default_factory=list, init=False)
    dropped_columns_: dict[str, str] = field(default_factory=dict, init=False)
    transformed_: ColumnTransformer | None = field(default=None, init=False)
    svd_: TruncatedSVD | None = field(default=None, init=False)
    post_scaler_: StandardScaler | None = field(default=None, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    encoded_dim_: int = 0
    explanation_: dict[str, Any] = field(default_factory=dict, init=False)

    def _select_feature_columns(self, frame: pd.DataFrame) -> list[str]:
        excluded: dict[str, str] = {}
        for column in frame.columns:
            if column == self.label_column:
                excluded[column] = "target label"
            elif column in self.excluded_columns:
                excluded[column] = "excluded (evaluation ground truth / user excluded)"
            elif column in self.id_like_columns:
                excluded[column] = "identifier-like (near-unique)"
            elif column in self.constant_columns:
                excluded[column] = "constant"
            elif column in self.all_missing_columns:
                excluded[column] = "entirely missing"
            else:
                series = frame[column]
                if int(series.nunique(dropna=True)) <= 1 and series.notna().any():
                    excluded[column] = "constant"
                elif series.notna().sum() == 0:
                    excluded[column] = "entirely missing"
        return [c for c in frame.columns if c not in excluded]

    def _prepare(self, frame: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        if not fit and self.transformed_ is not None:
            # Reuse the columns chosen at fit time so that held-out data goes
            # through exactly the same representation.
            prepared = pd.DataFrame(index=frame.index)
            for column in self.feature_columns_:
                if column not in frame.columns:
                    prepared[column] = np.nan if column in self.numeric_columns_ else "missing"
                    continue
                if column in self.numeric_columns_:
                    prepared[column] = _coerce_series(frame[column])
                else:
                    prepared[column] = normalise_categorical(frame[column])
            return prepared

        prepared = pd.DataFrame(index=frame.index)
        numeric: list[str] = []
        categorical: list[str] = []
        coercions: list[str] = []
        for column in self.feature_columns_:
            series = frame[column]
            if pd.api.types.is_datetime64_any_dtype(series):
                prepared[column] = _coerce_series(series)
                numeric.append(column)
                coercions.append(column)
            elif _is_numeric_like(series):
                prepared[column] = _coerce_series(series)
                numeric.append(column)
                if not pd.api.types.is_numeric_dtype(series):
                    coercions.append(column)
            else:
                prepared[column] = normalise_categorical(series)
                categorical.append(column)
        self.numeric_columns_ = numeric
        self.categorical_columns_ = categorical
        self.explanation_["numeric_coercions"] = sorted(set(coercions))
        return prepared

    def fit_transform(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
        if frame.empty:
            raise FeatureBuildError("Cannot build features for an empty dataset.")
        self.feature_columns_ = self._select_feature_columns(frame)
        if not self.feature_columns_:
            raise FeatureBuildError(
                "No usable feature columns remain after excluding the label, identifiers, "
                "constant and entirely-missing columns."
            )
        prepared = self._prepare(frame, fit=True)

        transformers: list[tuple[str, Any, list[str]]] = []
        if self.numeric_columns_:
            transformers.append(
                (
                    "numeric",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
                            ("scale", StandardScaler()),
                        ]
                    ),
                    self.numeric_columns_,
                )
            )
        if self.categorical_columns_:
            transformers.append(
                (
                    "categorical",
                    Pipeline(
                        [
                            ("impute", SimpleImputer(strategy="most_frequent")),
                            (
                                "encode",
                                OneHotEncoder(
                                    handle_unknown="ignore",
                                    max_categories=self.max_onehot_categories,
                                    sparse_output=True,
                                    dtype=np.float32,
                                ),
                            ),
                        ]
                    ),
                    self.categorical_columns_,
                )
            )
        if not transformers:
            raise FeatureBuildError("No numeric or categorical feature columns were available.")

        self.transformed_ = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            sparse_threshold=0.3,
            verbose_feature_names_out=False,
        )
        matrix = self.transformed_.fit_transform(prepared)
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
        self.encoded_dim_ = int(matrix.shape[1])

        svd_applied = False
        explained = None
        if (
            self.encoded_dim_ > self.svd_components
            and matrix.shape[0] > 2
            and self.svd_components > 1
        ):
            n_components = min(self.svd_components, matrix.shape[1] - 1, matrix.shape[0] - 1)
            self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.random_state)
            matrix = self.svd_.fit_transform(matrix).astype(np.float32)
            self.post_scaler_ = StandardScaler()
            matrix = self.post_scaler_.fit_transform(matrix).astype(np.float32)
            explained = float(self.svd_.explained_variance_ratio_.sum())
            svd_applied = True

        matrix = np.nan_to_num(matrix, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)
        matrix = np.clip(matrix, -self.clip_value, self.clip_value).astype(np.float32)

        meta = {
            "method": "tabular",
            "feature_columns": self.feature_columns_,
            "numeric_columns": self.numeric_columns_,
            "categorical_columns": self.categorical_columns_,
            "dropped_columns": self.dropped_columns_,
            "encoded_dim": self.encoded_dim_,
            "output_dim": int(matrix.shape[1]),
            "svd_applied": svd_applied,
            "svd_explained_variance": explained,
            "numeric_coercions": self.explanation_.get("numeric_coercions", []),
            "clip_value": self.clip_value,
        }
        return matrix, meta

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Transform held-out rows with the already-fitted representation."""

        if self.transformed_ is None:
            raise FeatureBuildError("The feature builder has not been fitted.")
        prepared = self._prepare(frame)
        matrix = self.transformed_.transform(prepared)
        if hasattr(matrix, "toarray"):
            matrix = matrix.toarray()
        matrix = np.asarray(matrix, dtype=np.float32)
        if self.svd_ is not None:
            matrix = self.svd_.transform(matrix).astype(np.float32)
            if self.post_scaler_ is not None:
                matrix = self.post_scaler_.transform(matrix).astype(np.float32)
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=self.clip_value, neginf=-self.clip_value)
        return np.clip(matrix, -self.clip_value, self.clip_value).astype(np.float32)


def compute_tabular_integrity(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str | None = None,
) -> pd.DataFrame:
    """Per-row data-integrity evidence derived from the original table."""

    if not feature_columns:
        return pd.DataFrame()
    feature_frame = frame[feature_columns]
    missing_ratio = feature_frame.isna().mean(axis=1).to_numpy(dtype=float)
    sentinel_counts = np.zeros(len(frame), dtype=float)
    for column in feature_columns:
        series = frame[column]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(series):
            continue
        sentinels = series.dropna().astype(str).str.strip().str.lower().isin(MISSING_SENTINELS)
        sentinel_counts += sentinels.reindex(frame.index, fill_value=False).to_numpy(dtype=float)
    sentinel_ratio = sentinel_counts / max(len(feature_columns), 1)

    infinite_counts = np.zeros(len(frame), dtype=float)
    for column in feature_columns:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        infinite_counts += (~np.isfinite(numeric) & numeric.notna()).to_numpy(dtype=float)

    duplicate_mask = frame.duplicated(keep="first").to_numpy()
    duplicate_feature_mask = frame[feature_columns].duplicated(keep="first").to_numpy()

    integrity = (
        0.55 * np.clip(missing_ratio, 0.0, 1.0)
        + 0.25 * np.clip(sentinel_ratio, 0.0, 1.0)
        + 0.20 * (infinite_counts > 0).astype(float)
    )
    # A fully complete row carries no integrity penalty.
    integrity = np.clip(integrity, 0.0, 1.0)
    return pd.DataFrame(
        {
            "missing_ratio": missing_ratio,
            "sentinel_ratio": sentinel_ratio,
            "infinite_values": infinite_counts,
            "exact_duplicate": duplicate_mask,
            "duplicate_features": duplicate_feature_mask,
            "integrity_score": integrity,
        },
        index=frame.index,
    )


def build_tabular_features(
    frame: pd.DataFrame,
    label_column: str | None,
    id_like_columns: list[str],
    constant_columns: list[str],
    all_missing_columns: list[str],
    max_onehot_categories: int = 20,
    svd_components: int = 64,
    random_state: int = 42,
    excluded_columns: list[str] | None = None,
) -> FeatureResult:
    builder = TabularFeatureBuilder(
        label_column=label_column,
        id_like_columns=id_like_columns,
        constant_columns=constant_columns,
        all_missing_columns=all_missing_columns,
        excluded_columns=list(excluded_columns or []),
        max_onehot_categories=max_onehot_categories,
        svd_components=svd_components,
        random_state=random_state,
    )
    with timer() as clock:
        matrix, meta = builder.fit_transform(frame)
    integrity = compute_tabular_integrity(frame, meta["feature_columns"], label_column)
    meta["row_integrity"] = integrity
    return FeatureResult(
        matrix=matrix,
        sample_ids=[],
        method="tabular",
        meta=meta,
        preprocessor=builder,
        duration_s=clock["seconds"],
        warnings=[],
    )
