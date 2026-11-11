"""Ingestion tests for both modalities."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from backend.ingestion.images import (
    build_image_dataset,
    load_image_folder,
    load_uploaded_images,
)
from backend.ingestion.tabular import build_tabular_dataset, read_tabular


def test_tabular_profile_and_duplicates(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "id": ["a", "a", "c", "d"],
            "x": [1, 1, 2, 4],
            "label": ["cat", "cat", "dog", "bird"],
        }
    )
    path = tmp_path / "table.csv"
    frame.to_csv(path, index=False)
    read = read_tabular(path)
    dataset = build_tabular_dataset(
        read, source_name="table.csv", label_column="label", task="classification"
    )
    assert dataset.modality == "tabular"
    assert dataset.profile.n_samples == 4
    assert dataset.profile.duplicate_rows == 1
    assert dataset.has_labels
    assert dataset.sample_ids == ["S000001", "S000002", "S000003", "S000004"]


def test_tabular_unlabelled() -> None:
    frame = pd.DataFrame({"x": [1, 2, 3], "y": ["a", "b", "c"]})
    dataset = build_tabular_dataset(frame, source_name="x.csv", label_column=None, task="unlabeled")
    assert not dataset.has_labels
    assert dataset.task == "unlabeled"
    assert dataset.label_column is None


def test_image_mapping_duplicate_basenames(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    Image.fromarray(np.full((16, 16, 3), 10, dtype=np.uint8)).save(tmp_path / "a" / "same.png")
    Image.fromarray(np.full((16, 16, 3), 200, dtype=np.uint8)).save(tmp_path / "b" / "same.png")
    mapping = pd.DataFrame(
        [
            {"path": "a/same.png", "label": "cat", "is_poisoned": 0},
            {"path": "b/same.png", "label": "dog", "is_poisoned": 1},
        ]
    )
    records, warnings, extra = load_image_folder(tmp_path, mapping=mapping, infer_folder_labels=False)
    labels = {record.rel_path: record.label for record in records}
    assert labels["a/same.png"] == "cat"
    assert labels["b/same.png"] == "dog"
    assert "is_poisoned" in extra
    assert not warnings


def test_image_root_level_mapping_not_duplicated(tmp_path: Path) -> None:
    Image.fromarray(np.full((16, 16, 3), 5, dtype=np.uint8)).save(tmp_path / "x.png")
    mapping = pd.DataFrame([{"path": "x.png", "label": "rootcat"}])
    records, warnings, extra = load_image_folder(tmp_path, mapping=mapping, infer_folder_labels=False)
    assert len(records) == 1
    assert records[0].label == "rootcat"
    assert not warnings


def test_image_folder_labels_and_unreadable(tmp_path: Path) -> None:
    (tmp_path / "cat").mkdir()
    (tmp_path / "dog").mkdir()
    Image.fromarray(np.full((16, 16, 3), 7, dtype=np.uint8)).save(tmp_path / "cat" / "one.png")
    (tmp_path / "dog" / "broken.png").write_bytes(b"not an image")
    records, warnings, extra = load_image_folder(tmp_path, infer_folder_labels=True)
    labels = {record.rel_path: record.label for record in records}
    assert labels["cat/one.png"] == "cat"
    assert any(not record.readable for record in records)
    assert extra == []


def test_uploaded_images(tmp_path: Path) -> None:
    payload = []
    for i in range(2):
        image = Image.fromarray(np.full((12, 12, 3), 50 + i * 10, dtype=np.uint8))
        buffer = tmp_path / f"u{i}.png"
        image.save(buffer)
        payload.append((buffer.name, buffer.read_bytes()))
    destination = tmp_path / "dest"
    records, warnings, extra = load_uploaded_images(payload, destination, infer_folder_labels=False)
    assert len(records) == 2
    assert all(record.readable for record in records)
    assert any("No image labels" in warning for warning in warnings)
