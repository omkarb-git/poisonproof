"""Session-state helpers for the Streamlit UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from backend.config import AnalysisConfig, SESSIONS_DIR
from backend.evaluation.model_comparison import ModelComparisonResult
from backend.intelligence.reviewer import ReviewStore
from backend.intelligence.sanitizer import SanitisationResult
from backend.types import AnalysisBundle, LoadedDataset
from backend.utils import path_fingerprint

DEFAULTS: dict[str, Any] = {
    "ps_dataset": None,
    "ps_bundle": None,
    "ps_store": None,
    "ps_sanitisation": None,
    "ps_comparison": None,
    "ps_report_path": None,
    "ps_load_warnings": [],
    "ps_original_hash": None,
    "ps_reviewer": "",
    "ps_k": 15,
    "ps_seed": 42,
    "ps_batch": 32,
    "ps_extractor": "auto",
    "ps_weight_label": 0.50,
    "ps_weight_outlier": 0.20,
    "ps_weight_duplicate": 0.10,
    "ps_weight_integrity": 0.20,
    "ps_thr_low": 0.30,
    "ps_thr_moderate": 0.55,
    "ps_thr_high": 0.75,
    "ps_nav": "Overview",
    "ps_holdout_signature": None,
    "ps_last_run_error": None,
    "ps_flash": None,
}


def init_state() -> None:
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value


def current_dataset() -> LoadedDataset | None:
    return st.session_state.get("ps_dataset")


def current_bundle() -> AnalysisBundle | None:
    return st.session_state.get("ps_bundle")


def current_store() -> ReviewStore | None:
    return st.session_state.get("ps_store")


def current_sanitisation() -> SanitisationResult | None:
    return st.session_state.get("ps_sanitisation")


def current_comparison() -> ModelComparisonResult | None:
    return st.session_state.get("ps_comparison")


def set_dataset(dataset: LoadedDataset, original_hash: str | None = None) -> None:
    st.session_state["ps_dataset"] = dataset
    st.session_state["ps_original_hash"] = (
        original_hash
        if original_hash is not None
        else (path_fingerprint(dataset.original_copy) if dataset.original_copy else None)
    )
    clear_analysis(keep_dataset=True)
    st.session_state["ps_load_warnings"] = list(dataset.notes)


def set_bundle(bundle: AnalysisBundle) -> None:
    st.session_state["ps_bundle"] = bundle
    workspace = Path(bundle.workspace) if bundle.workspace else SESSIONS_DIR / bundle.session_id
    reviewer = st.session_state.get("ps_reviewer", "")
    store = ReviewStore(bundle.session_id, workspace, reviewer=reviewer)
    st.session_state["ps_store"] = store
    st.session_state["ps_sanitisation"] = None
    st.session_state["ps_comparison"] = None
    st.session_state["ps_report_path"] = None
    st.session_state["ps_last_run_error"] = None


def clear_analysis(keep_dataset: bool = True) -> None:
    st.session_state["ps_bundle"] = None
    st.session_state["ps_store"] = None
    st.session_state["ps_sanitisation"] = None
    st.session_state["ps_comparison"] = None
    st.session_state["ps_report_path"] = None
    if not keep_dataset:
        st.session_state["ps_dataset"] = None
        st.session_state["ps_load_warnings"] = []
        st.session_state["ps_original_hash"] = None


def reset_all() -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("ps_"):
            del st.session_state[key]
    init_state()


def set_decision(
    sample_id: str,
    action: str,
    original_label: str | None,
    suggested_label: str | None = None,
    corrected_label: str | None = None,
    notes: str = "",
) -> None:
    store = current_store()
    if store is None:
        raise RuntimeError("No review store is available for this analysis session.")
    store.set_decision(
        sample_id,
        action,
        corrected_label=corrected_label,
        original_label=original_label,
        suggested_label=suggested_label,
        notes=notes,
        reviewer=st.session_state.get("ps_reviewer", ""),
    )


def build_analysis_config(dataset: LoadedDataset) -> AnalysisConfig:
    from backend.config import RiskThresholds, RiskWeights

    weights = RiskWeights(
        label_evidence=float(st.session_state.get("ps_weight_label", 0.5)),
        outlier_evidence=float(st.session_state.get("ps_weight_outlier", 0.2)),
        duplicate_evidence=float(st.session_state.get("ps_weight_duplicate", 0.1)),
        integrity_evidence=float(st.session_state.get("ps_weight_integrity", 0.2)),
    )
    thresholds = RiskThresholds(
        low_max=float(st.session_state.get("ps_thr_low", 0.3)),
        moderate_max=float(st.session_state.get("ps_thr_moderate", 0.55)),
        high_max=float(st.session_state.get("ps_thr_high", 0.75)),
    )
    return AnalysisConfig(
        k_neighbors=int(st.session_state.get("ps_k", 15)),
        random_seed=int(st.session_state.get("ps_seed", 42)),
        task=dataset.task,
        image_extractor=str(st.session_state.get("ps_extractor", "auto")),
        image_batch_size=int(st.session_state.get("ps_batch", 32)),
        ground_truth_column=dataset.ground_truth_column,
        risk_weights=weights,
        risk_thresholds=thresholds,
    )


def go_to(section: str) -> None:
    st.session_state["ps_nav"] = section
    st.rerun()
