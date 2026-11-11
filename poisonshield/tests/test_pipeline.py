"""End-to-end pipeline tests."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from backend.config import AnalysisConfig
from backend.ingestion.images import build_image_dataset, load_image_folder
from backend.ingestion.tabular import build_tabular_dataset
from backend.pipeline import analyse_dataset


def _tabular_frame(n: int = 90) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    labels = rng.choice(["cat", "dog", "bird"], size=n)
    frame = pd.DataFrame(
        {
            "height": rng.normal(170, 8, n),
            "weight": rng.normal(70, 10, n),
            "city": rng.choice(["Mumbai", "Delhi", "Pune"], n),
            "label": labels,
        }
    )
    frame.loc[3, "height"] = float("nan")
    frame = pd.concat([frame, frame.iloc[[5, 6]]], ignore_index=True)
    return frame


def test_tabular_labelled_pipeline() -> None:
    frame = _tabular_frame()
    dataset = build_tabular_dataset(frame, source_name="t.csv", label_column="label", task="classification")
    bundle = analyse_dataset(dataset, AnalysisConfig(k_neighbors=8))
    assert len(bundle.sample_ids) == len(frame)
    assert len(bundle.audit.statuses) == 5
    assert all(status.status == "completed" for status in bundle.audit.statuses)
    assert bundle.risk.scores["exact_duplicate"].sum() >= 2
    assert len(bundle.risk.explanations) > 0
    assert bundle.workspace is not None


def test_tabular_unlabelled_pipeline() -> None:
    frame = _tabular_frame().drop(columns=["label"])
    dataset = build_tabular_dataset(frame, source_name="u.csv", label_column=None, task="unlabeled")
    bundle = analyse_dataset(dataset, AnalysisConfig(k_neighbors=8))
    assert len(bundle.sample_ids) == len(frame)
    assert bundle.dataset.has_labels is False
    label_status = next(status for status in bundle.audit.statuses if status.issue == "label")
    assert label_status.status in {"skipped", "failed"}
    assert "label_check_available" in bundle.risk.scores.columns
    assert bool(bundle.risk.scores["label_check_available"].any()) is False


def test_image_sanitisation() -> None:
    root = Path("C:/Users/Omkar/AppData/Local/Temp/opencode/ps_test_image_sanitise")
    if root.exists():
        shutil.rmtree(root)
    (root / "cat").mkdir(parents=True)
    rng = np.random.default_rng(11)
    rows = []
    for i in range(12):
        label = "cat" if i % 2 == 0 else "dog"
        arr = rng.normal(180 if label == "cat" else 90, 30, (32, 32, 3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        (root / label / f"{label}_{i:02d}.png").parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(root / label / f"{label}_{i:02d}.png")
        rows.append({"path": f"{label}/{label}_{i:02d}.png", "label": label})
    mapping = pd.DataFrame(rows)
    records, warnings, extra = load_image_folder(root, mapping=mapping, infer_folder_labels=True)
    dataset = build_image_dataset(records, source_name="images", extra_columns=extra, notes=warnings)
    bundle = analyse_dataset(dataset, AnalysisConfig(k_neighbors=5, image_extractor="handcrafted", image_batch_size=4))
    from backend.intelligence.reviewer import ReviewStore
    from backend.intelligence.sanitizer import build_image_cleaned

    store = ReviewStore(bundle.session_id, bundle.workspace, reviewer="tester")
    store.set_decision(bundle.sample_ids[0], "remove", original_label=str(bundle.risk.scores.loc[bundle.sample_ids[0], "label"]), notes="remove")
    result = build_image_cleaned(bundle, store)
    assert result.n_removed == 1
    assert result.archive_path is not None and result.archive_path.exists()
    assert result.n_included == len(bundle.sample_ids) - 1


def test_model_comparison_tabular() -> None:
    frame = _tabular_frame()
    dataset = build_tabular_dataset(frame, source_name="t.csv", label_column="label", task="classification")
    bundle = analyse_dataset(dataset, AnalysisConfig(k_neighbors=8))
    holdout = frame.sample(30, random_state=3).reset_index(drop=True)
    from backend.evaluation.model_comparison import compare_models
    from backend.intelligence.sanitizer import build_tabular_cleaned
    from backend.intelligence.reviewer import ReviewStore

    store = ReviewStore(bundle.session_id, bundle.workspace, reviewer="tester")
    cleaned = build_tabular_cleaned(bundle, store)
    comparison = compare_models(
        dataset.dataframe, cleaned.dataframe, holdout, "label", bundle.features.preprocessor, dataset.task, "tabular"
    )
    assert comparison.n_test == len(holdout)
    assert set(comparison.original_metrics) == {"accuracy", "macro_f1", "weighted_f1"}
    assert comparison.delta["accuracy"] is not None or comparison.delta["accuracy"] is None

    root = Path("C:/Users/Omkar/AppData/Local/Temp/opencode/ps_test_image_pipeline")
    if root.exists():
        shutil.rmtree(root)
    (root / "cat").mkdir(parents=True)
    (root / "dog").mkdir(parents=True)
    rng = np.random.default_rng(9)
    rows = []
    for i in range(18):
        label = "cat" if i % 2 == 0 else "dog"
        arr = rng.normal(180 if label == "cat" else 90, 30, (32, 32, 3))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        (root / label / f"{label}_{i:02d}.png").parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(root / label / f"{label}_{i:02d}.png")
        rows.append({"path": f"{label}/{label}_{i:02d}.png", "label": label, "is_poisoned": int(i == 2)})
    mapping = pd.DataFrame(rows)
    records, warnings, extra = load_image_folder(root, mapping=mapping, infer_folder_labels=True)
    dataset = build_image_dataset(records, source_name="images", extra_columns=extra, notes=warnings)
    dataset.ground_truth_column = "is_poisoned"
    bundle = analyse_dataset(dataset, AnalysisConfig(k_neighbors=5, image_extractor="handcrafted", image_batch_size=4))
    assert len(bundle.sample_ids) == len(records)
    assert bundle.features.meta["extractor"] == "handcrafted"
    assert all(status.status == "completed" for status in bundle.audit.statuses)
    assert len(bundle.risk.explanations) > 0
