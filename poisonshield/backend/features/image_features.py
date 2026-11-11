"""Replaceable pretrained image feature extraction with per-file caching.

The default is DINOv2 (ViT-S/14) through timm. If pretrained weights are not
available (offline machine, unsupported torch build) PoisonShield degrades to a
torchvision/timm ResNet-50 and finally to a deterministic handcrafted descriptor
so that auditing can still proceed. The extractor actually used is always
recorded in the analysis metadata and displayed in the UI/report.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from ..config import CACHE_DIR, SUPPORTED_IMAGE_EXTENSIONS
from ..types import FeatureResult, ImageRecord
from ..utils import ProgressCallback, make_progress, safe_slug, timer

EXTRACTOR_SPECS: dict[str, dict[str, Any]] = {
    "dinov2_vits14": {
        "label": "DINOv2 ViT-S/14",
        "timm": "vit_small_patch14_dinov2.lvd142m",
        "img_size": 518,
        "note": "General-purpose self-supervised vision foundation model (recommended).",
        "foundation": True,
    },
    "dinov2_vitb14": {
        "label": "DINOv2 ViT-B/14",
        "timm": "vit_base_patch14_dinov2.lvd142m",
        "img_size": 518,
        "note": "Larger DINOv2 backbone; better features, slower on CPU.",
        "foundation": True,
    },
    "resnet50": {
        "label": "ResNet-50 (ImageNet)",
        "timm": "resnet50",
        "img_size": 224,
        "note": "Supervised ImageNet backbone; useful fallback.",
        "foundation": False,
    },
    "handcrafted": {
        "label": "PoisonShield classic descriptor",
        "timm": None,
        "img_size": 96,
        "note": "Offline colour/gradient/texture descriptor. No semantic pretraining; "
                "use only when pretrained backbones are unavailable.",
        "foundation": False,
    },
}


class ExtractorUnavailable(RuntimeError):
    """Raised when a requested pretrained backbone cannot be initialised."""


def available_extractors() -> list[dict[str, Any]]:
    options = []
    for key, spec in EXTRACTOR_SPECS.items():
        options.append(
            {
                "key": key,
                "label": spec["label"],
                "note": spec["note"],
                "foundation": spec["foundation"],
            }
        )
    return options


def _torch_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class _TimmEmbedder:
    def __init__(self, key: str) -> None:
        self.key = key
        self.spec = EXTRACTOR_SPECS[key]
        self.device = _torch_device()
        self._torch = None
        self._model = None
        self._transform = None
        self.dim: int | None = None
        self.description = f"{self.spec['label']} on {self.device.upper()}"

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import timm
            import torch
        except Exception as exc:  # noqa: BLE001
            raise ExtractorUnavailable(f"timm/torch unavailable: {exc}") from exc
        self._torch = torch
        try:
            model = timm.create_model(
                self.spec["timm"],
                pretrained=True,
                num_classes=0,
                img_size=int(self.spec["img_size"]),
            )
            model.eval()
            try:
                config = timm.data.resolve_data_config({}, model=model)
                transform = timm.data.create_transform(**config, is_training=False)
            except Exception:  # noqa: BLE001 - fall back to standard preprocessing
                config = {"input_size": (3, int(self.spec["img_size"]), int(self.spec["img_size"]))}
                transform = timm.data.create_transform(**config, is_training=False)
            self._model = model.to(self.device)
            self._transform = transform
            self.dim = int(getattr(model, "num_features", 0)) or None
        except Exception as exc:  # noqa: BLE001
            raise ExtractorUnavailable(
                f"Could not load pretrained weights for {self.spec['label']} "
                f"({type(exc).__name__}: {exc})."
            ) from exc

    def embed_paths(self, paths: list[Path], progress: Callable[[int, int], None]) -> np.ndarray:
        self._load()
        assert self._model is not None and self._transform is not None and self._torch is not None
        vectors: list[np.ndarray] = []
        # The caller batches for progress reporting; embed the given list now.
        batch_images = []
        for path in paths:
            with Image.open(path) as img:
                batch_images.append(self._transform(img.convert("RGB")))
        if not batch_images:
            return np.empty((0, self.dim or 1), dtype=np.float32)
        batch = self._torch.stack(batch_images).to(self.device)
        with self._torch.inference_mode():
            output = self._model(batch)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if isinstance(output, dict):
            output = next(iter(output.values()))
        matrix = output.detach().float().cpu().numpy()
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix / np.clip(norms, 1e-12, None)
        return matrix.astype(np.float32)


def _handcrafted_vector(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        small = np.asarray(rgb.resize((96, 96)), dtype=np.float32) / 255.0
        gray16 = np.asarray(rgb.convert("L").resize((16, 16)), dtype=np.float32) / 255.0
        gray32 = np.asarray(rgb.convert("L").resize((32, 32)), dtype=np.float32) / 255.0

    features: list[np.ndarray] = []
    for channel in range(3):
        hist, _ = np.histogram(small[:, :, channel], bins=16, range=(0.0, 1.0))
        features.append((hist / max(hist.sum(), 1)).astype(np.float32))
    centred = gray16.ravel() - float(gray16.mean())
    features.append((centred / (float(gray16.std()) + 1e-6) * 0.25).astype(np.float32))

    gx = np.diff(gray32, axis=1)
    gy = np.diff(gray32, axis=0)
    # Both gradients describe the 31x31 cell grid between pixels, so crop the
    # same trailing row/column before combining them.
    gx = gx[:-1, :]
    gy = gy[:, :-1]
    magnitude = np.hypot(gx, gy)
    orientation = (np.arctan2(gy, gx) + np.pi) / (2 * np.pi)
    hist, _ = np.histogram(orientation, bins=12, range=(0.0, 1.0), weights=magnitude)
    hist = hist / max(hist.sum(), 1)
    features.append(hist.astype(np.float32))
    features.append(np.array([magnitude.mean(), magnitude.std()], dtype=np.float32))
    features.append(
        np.concatenate(
            [
                small.reshape(-1, 3).mean(axis=0),
                small.reshape(-1, 3).std(axis=0),
            ]
        ).astype(np.float32)
    )
    vector = np.concatenate(features).astype(np.float32)
    vector = np.nan_to_num(vector, nan=0.0)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-12)


class _HandcraftedEmbedder:
    key = "handcrafted"
    description = "PoisonShield classic descriptor (offline, no pretrained weights)"
    dim = 324
    device = "cpu"

    def embed_paths(self, paths: list[Path], progress: Callable[[int, int], None]) -> np.ndarray:
        vectors = []
        for position, path in enumerate(paths, start=1):
            vectors.append(_handcrafted_vector(path))
            progress(position, len(paths))
        return np.vstack(vectors).astype(np.float32) if vectors else np.empty((0, self.dim), np.float32)


def _cache_path(extractor_key: str, file_hash: str) -> Path:
    directory = CACHE_DIR / "image_embeddings" / safe_slug(extractor_key)
    return directory / file_hash[:2] / f"{file_hash}.npy"


def extract_image_features(
    records: list[ImageRecord],
    extractor: str = "auto",
    batch_size: int = 32,
    progress: ProgressCallback | None = None,
    use_cache: bool = True,
) -> FeatureResult:
    """Embed all readable images and return a row-aligned feature matrix."""

    report = make_progress(progress)
    readable = [record for record in records if record.readable]
    if not readable:
        raise ValueError("No readable images are available for embedding.")

    requested = (extractor or "auto").strip()
    if requested == "auto":
        candidates = ["dinov2_vits14", "resnet50", "handcrafted"]
    elif requested in EXTRACTOR_SPECS:
        candidates = [requested] if requested == "handcrafted" else [requested, "handcrafted"]
    else:
        candidates = ["handcrafted"]

    warnings: list[str] = []
    chosen: Any = None
    chosen_key = ""
    for candidate in candidates:
        try:
            embedder = _HandcraftedEmbedder() if candidate == "handcrafted" else _TimmEmbedder(candidate)
            if hasattr(embedder, "_load"):
                report(0.01, f"Loading {EXTRACTOR_SPECS[candidate]['label']} weights...")
                embedder._load()  # type: ignore[attr-defined]
            chosen = embedder
            chosen_key = candidate
            break
        except ExtractorUnavailable as exc:
            warnings.append(str(exc))
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{candidate} could not be initialised: {type(exc).__name__}: {exc}")
    if chosen is None:
        raise RuntimeError(
            "No image feature extractor could be initialised. " + " | ".join(warnings)
        )

    cache_key = f"{chosen_key}_{chosen.dim or 'x'}"
    matrix = np.zeros((len(readable), chosen.dim or 1), dtype=np.float32)
    missing: list[tuple[int, ImageRecord]] = []
    cache_hits = 0
    for position, record in enumerate(readable):
        cached = None
        if use_cache and record.sha256:
            path = _cache_path(cache_key, record.sha256)
            if path.exists():
                try:
                    cached = np.load(path)
                except Exception:  # noqa: BLE001
                    cached = None
        if cached is not None and cached.ndim == 1 and cached.size == matrix.shape[1]:
            matrix[position] = cached
            cache_hits += 1
        else:
            missing.append((position, record))

    report(0.05, f"Extracting embeddings for {len(missing):,} image(s) with {chosen.description}...")
    batch_size = max(1, int(batch_size))
    processed = 0
    for start in range(0, len(missing), batch_size):
        chunk = missing[start:start + batch_size]
        paths = [Path(record.path) for _, record in chunk]
        vectors = chosen.embed_paths(paths, lambda _done, _total: None)
        if vectors.shape[0] != len(chunk):
            raise RuntimeError(
                f"Extractor returned {vectors.shape[0]} embeddings for {len(chunk)} images."
            )
        for (position, record), vector in zip(chunk, vectors, strict=True):
            matrix[position] = vector
            if use_cache and record.sha256:
                target = _cache_path(cache_key, record.sha256)
                target.parent.mkdir(parents=True, exist_ok=True)
                try:
                    np.save(target, vector.astype(np.float32))
                except OSError:
                    pass
        processed += len(chunk)
        fraction = 0.05 + 0.9 * (processed / max(len(missing), 1))
        report(fraction, f"Embedded {processed:,}/{len(missing):,} image(s)...")

    meta = {
        "method": "image",
        "extractor": chosen_key,
        "extractor_label": EXTRACTOR_SPECS.get(chosen_key, {}).get("label", chosen_key),
        "extractor_note": EXTRACTOR_SPECS.get(chosen_key, {}).get("note", ""),
        "device": getattr(chosen, "device", "cpu"),
        "embedding_dim": int(matrix.shape[1]),
        "requested_extractor": requested,
        "fallback_used": chosen_key != requested and requested != "auto",
        "cache_hits": cache_hits,
        "cache_misses": len(missing),
        "cache_enabled": use_cache,
        "n_readable": len(readable),
        "n_unreadable": len(records) - len(readable),
        "unreadable_sample_ids": [r.sample_id for r in records if not r.readable],
        "domain_note": (
            "Visual similarity quality depends on the feature extractor and domain. "
            "General-purpose models can under-represent specialised imagery such as "
            "medical, satellite or industrial inspection data."
        ),
    }
    report(1.0, "Image embeddings ready.")
    return FeatureResult(
        matrix=matrix,
        sample_ids=[record.sample_id for record in readable],
        method="image",
        meta=meta,
        preprocessor=None,
        duration_s=0.0,
        warnings=warnings,
    )
