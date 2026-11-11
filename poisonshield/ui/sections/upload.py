"""Upload and dataset-configuration page."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from backend.config import (
    MAX_IMAGE_COUNT,
    MAX_TABULAR_ROWS,
    SUPPORTED_IMAGE_EXTENSIONS,
    UPLOAD_DIR,
    ensure_workspace,
)
from backend.evaluation.metrics import available_ground_truth_columns
from backend.ingestion.images import (
    ImageLoadError,
    build_image_dataset,
    load_image_folder,
    load_mapping_from_bytes,
    load_uploaded_images,
    resolve_mapping_columns,
    suggest_ground_truth_column,
)
from backend.ingestion.tabular import (
    TabularLoadError,
    build_tabular_dataset,
    read_tabular,
    suggest_label_column,
)
from backend.types import LoadedDataset
from backend.utils import safe_extract_zip, sha256_file, timestamp_slug

from .. import state
from ..components import callout, fmt, render_dataset_line, render_run_controls
from ..theme import hero


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in Path(name).name)
    return cleaned.strip(" .")[:120] or "upload"


def _save_bytes(name: str, payload: bytes, folder: Path) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / _safe_name(name)
    counter = 1
    while target.exists():
        counter += 1
        target = folder / f"{Path(_safe_name(name)).stem}_{counter}{Path(name).suffix}"
    target.write_bytes(payload)
    return target


def _read_mapping_upload(upload: st.runtime.uploaded_file_manager.UploadedFile | None) -> pd.DataFrame | None:
    if upload is None:
        return None
    return load_mapping_from_bytes(upload.name, upload.getvalue())


def _find_mapping_in_zip(root: Path) -> tuple[pd.DataFrame | None, str | None]:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
    ]
    if not candidates:
        return None, None
    if len(candidates) > 1:
        names = ", ".join(path.name for path in candidates[:5])
        return None, f"Several tabular files were found in the archive ({names}); upload the mapping explicitly."
    candidate = candidates[0]
    try:
        frame = load_mapping_from_bytes(candidate.name, candidate.read_bytes())
        resolve_mapping_columns(frame)
        return frame, f"Auto-detected mapping file: {candidate.name}"
    except Exception:  # noqa: BLE001 - an unrelated CSV in the archive is not a mapping
        return None, None


def _render_replace_button() -> None:
    if st.button("Remove dataset and start over", key="ps_replace_dataset"):
        state.reset_all()
        st.rerun()


def render_upload() -> None:
    hero(
        "Upload & Configure",
        "Add a tabular table or an image archive. Originals are copied unchanged and never modified.",
    )
    dataset = state.current_dataset()
    if dataset is not None:
        _render_loaded_dataset(dataset)
        return

    mode = st.radio(
        "Dataset type",
        ["Tabular (CSV / Excel)", "Images (ZIP or files)"],
        horizontal=True,
        key="ps_upload_mode",
    )
    if mode.startswith("Tabular"):
        _render_tabular_upload()
    else:
        _render_image_upload()


def _render_tabular_upload() -> None:
    st.markdown("#### Tabular dataset")
    st.caption(
        "Supported: CSV, XLSX, XLS. A label column is optional; datasets without labels are "
        "audited with feature-only detectors."
    )
    uploaded = st.file_uploader(
        "Choose a table",
        type=["csv", "xlsx", "xls"],
        key="ps_tabular_upload",
    )
    if uploaded is None:
        callout(
            "The original file is copied into the PoisonShield workspace before parsing. "
            "Parsing uses defensive row/column limits and never writes back to the uploaded file.",
            "info",
        )
        return

    st.caption(f"{uploaded.name} · {len(uploaded.getvalue()) / 1024:.1f} KB")
    if not st.button("Load table into PoisonShield", type="primary", key="ps_load_tabular"):
        return

    ensure_workspace()
    folder = UPLOAD_DIR / f"tab_{timestamp_slug()}"
    path = _save_bytes(uploaded.name, uploaded.getvalue(), folder)
    try:
        frame = read_tabular(path, max_rows=MAX_TABULAR_ROWS)
    except TabularLoadError as exc:
        st.error(f"Could not read the table: {exc}")
        return
    label = suggest_label_column(frame)
    dataset = build_tabular_dataset(
        frame,
        source_name=uploaded.name,
        label_column=label,
        task="auto",
        original_copy=path,
    )
    dataset.ground_truth_column = suggest_ground_truth_column(frame.columns)
    state.set_dataset(dataset, original_hash=sha256_file(path))
    st.rerun()


def _render_image_upload() -> None:
    st.markdown("#### Image dataset")
    st.caption(
        "Upload one ZIP archive (recommended, preserves folders) or a batch of image files. "
        "A CSV/Excel mapping with image path and label columns is optional; folder names can be used as labels."
    )
    archive = st.file_uploader("Image ZIP", type=["zip"], key="ps_image_zip")
    files = st.file_uploader(
        "…or individual images",
        type=[ext.lstrip(".") for ext in sorted(SUPPORTED_IMAGE_EXTENSIONS)],
        accept_multiple_files=True,
        key="ps_image_files",
    )
    mapping_upload = st.file_uploader(
        "Optional label mapping (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="ps_image_mapping",
    )
    infer_folder_labels = st.checkbox(
        "Use the deepest folder name as a label when no mapping row matches",
        value=True,
        key="ps_infer_folder_labels",
    )
    if archive is None and not files:
        callout(
            "Limit: "
            f"{MAX_IMAGE_COUNT:,} images per dataset. Images are embedded with a replaceable "
            "pretrained extractor; unreadable files are retained as integrity findings.",
            "info",
        )
        return

    if not st.button("Load images into PoisonShield", type="primary", key="ps_load_images"):
        return
    ensure_workspace()
    folder = UPLOAD_DIR / f"img_{timestamp_slug()}"
    folder.mkdir(parents=True, exist_ok=True)
    try:
        mapping = _read_mapping_upload(mapping_upload)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not read the label mapping: {exc}")
        return

    try:
        if archive is not None:
            archive_path = _save_bytes(archive.name, archive.getvalue(), folder / "source")
            source_name = archive.name
            extract_dir = folder / "extracted"
            root = safe_extract_zip(archive_path, extract_dir)
            auto_note = None
            if mapping is None:
                mapping, auto_note = _find_mapping_in_zip(root)
            records, warnings, extra_columns = load_image_folder(
                root,
                max_images=MAX_IMAGE_COUNT,
                mapping=mapping,
                infer_folder_labels=infer_folder_labels,
            )
            original_copy = archive_path
        else:
            source_name = f"{len(files)} uploaded image file(s)"
            payloads = [(item.name, item.getvalue()) for item in files or []]
            destination = folder / "source"
            records, warnings, extra_columns = load_uploaded_images(
                payloads,
                destination,
                max_images=MAX_IMAGE_COUNT,
                mapping=mapping,
                infer_folder_labels=False,
            )
            original_copy = destination
            auto_note = None
    except (ImageLoadError, ValueError) as exc:
        st.error(f"Could not load the images: {exc}")
        return
    except Exception as exc:  # noqa: BLE001
        st.error(f"Unexpected image-loading error: {exc}")
        return

    dataset = build_image_dataset(
        records,
        source_name=source_name,
        extra_columns=extra_columns,
        original_copy=original_copy,
        notes=list(warnings),
    )
    dataset.ground_truth_column = suggest_ground_truth_column(extra_columns)
    if auto_note:
        dataset.notes.insert(0, auto_note)
    state.set_dataset(dataset)
    st.rerun()


def _render_loaded_dataset(dataset: LoadedDataset) -> None:
    bundle = state.current_bundle()
    render_dataset_line(dataset)
    if bundle is not None:
        callout(
            "An analysis session is loaded. Changing the dataset configuration will clear the "
            "current analysis and its review decisions.",
            "info",
        )
    else:
        callout("Dataset loaded. Review the configuration, then run the analysis.", "ok")

    if dataset.modality == "tabular" and dataset.dataframe is not None:
        _render_tabular_configuration(dataset)
    elif dataset.modality == "image":
        _render_image_configuration(dataset)

    st.divider()
    render_run_controls(dataset)
    st.divider()
    _render_replace_button()


def _render_tabular_configuration(dataset: LoadedDataset) -> None:
    frame = dataset.dataframe
    assert frame is not None
    st.markdown("#### Dataset configuration")
    columns = [str(column) for column in frame.columns]
    label_options = ["<no label column>"] + columns
    current_label = dataset.label_column or "<no label column>"
    label_index = label_options.index(current_label) if current_label in label_options else 0
    ground_options = ["<none>"] + columns
    current_ground = dataset.ground_truth_column or "<none>"
    ground_index = ground_options.index(current_ground) if current_ground in ground_options else 0
    task_options = ["auto", "classification", "regression", "unlabeled"]
    task_index = task_options.index(dataset.task) if dataset.task in task_options else 0

    col1, col2, col3 = st.columns(3)
    with col1:
        label_column = st.selectbox("Label column", label_options, index=label_index, key="ps_cfg_label")
    with col2:
        task = st.selectbox("Task", task_options, index=task_index, key="ps_cfg_task")
    with col3:
        ground_truth = st.selectbox(
            "Evaluation ground-truth column (optional)",
            ground_options,
            index=ground_index,
            key="ps_cfg_ground",
        )
    if ground_truth != "<none>":
        st.caption(
            "The selected ground-truth column is excluded from the feature representation to avoid "
            "leakage. It is only used by the Evaluation page."
        )
    if label_column != "<no label column>" and label_column == ground_truth:
        st.error("The label column and the evaluation ground-truth column must be different.")
        return
    if st.button("Apply dataset configuration", key="ps_apply_cfg"):
        resolved_label = None if label_column == "<no label column>" else label_column
        resolved_ground = None if ground_truth == "<none>" else ground_truth
        try:
            rebuilt = build_tabular_dataset(
                frame.copy(deep=True),
                source_name=dataset.source_name,
                label_column=resolved_label,
                task=task,
                original_copy=dataset.original_copy,
            )
            rebuilt.ground_truth_column = resolved_ground
        except TabularLoadError as exc:
            st.error(f"Could not apply the configuration: {exc}")
            return
        state.set_dataset(rebuilt)
        st.rerun()

    st.markdown("#### Preview")
    preview = frame.head(30).copy()
    st.dataframe(preview, width="stretch", hide_index=True)
    st.caption(f"Showing 30 of {len(frame):,} rows.")

    profile = dataset.profile
    metrics = st.columns(5)
    with metrics[0]:
        st.metric("Rows", f"{profile.n_samples:,}")
    with metrics[1]:
        st.metric("Columns", f"{profile.n_columns:,}")
    with metrics[2]:
        st.metric("Missing cells", f"{profile.missing_cells:,}")
    with metrics[3]:
        st.metric("Exact duplicates", f"{profile.duplicate_rows:,}")
    with metrics[4]:
        st.metric("Classes", f"{len(profile.class_distribution):,}")
    if dataset.notes:
        st.markdown("##### Ingestion notes")
        for note in dataset.notes:
            st.markdown(f"- {html.escape(str(note))}")


def _render_image_configuration(dataset: LoadedDataset) -> None:
    st.markdown("#### Dataset configuration")
    ground_options = ["<none>"] + available_ground_truth_columns(dataset)
    current_ground = dataset.ground_truth_column or "<none>"
    ground_index = ground_options.index(current_ground) if current_ground in ground_options else 0
    ground_truth = st.selectbox(
        "Evaluation ground-truth column (optional)",
        ground_options,
        index=ground_index,
        key="ps_cfg_img_ground",
    )
    if ground_truth != "<none>":
        st.caption(
            "The ground-truth values come from the uploaded label mapping. Image-pixel features do "
            "not include mapping columns, so this selection does not leak into the embeddings."
        )
    if st.button("Apply ground-truth configuration", key="ps_apply_img_cfg"):
        dataset.ground_truth_column = None if ground_truth == "<none>" else ground_truth
        state.set_dataset(dataset)
        st.rerun()

    stats = dataset.profile.image_stats or {}
    metrics = st.columns(5)
    with metrics[0]:
        st.metric("Images", f"{dataset.profile.n_samples:,}")
    with metrics[1]:
        st.metric("Readable", f"{stats.get('readable', 0):,}")
    with metrics[2]:
        st.metric("Unreadable", f"{stats.get('unreadable', 0):,}")
    with metrics[3]:
        st.metric("Exact duplicates", f"{stats.get('exact_duplicates', 0):,}")
    with metrics[4]:
        st.metric("Missing labels", f"{stats.get('missing_labels', 0):,}")

    if dataset.extra_columns:
        st.caption("Mapping columns available: " + ", ".join(dataset.extra_columns))
    if dataset.notes:
        st.markdown("##### Ingestion notes")
        for note in dataset.notes:
            st.markdown(f"- {html.escape(str(note))}")

    preview = [record for record in (dataset.images or []) if record.readable][:12]
    if preview:
        st.markdown("#### Preview")
        columns = st.columns(4)
        for index, record in enumerate(preview):
            with columns[index % 4]:
                from ..components import render_image

                render_image(record, caption=f"{record.rel_path} · {record.label or 'unlabelled'}", width=250)
    if st.session_state.get("ps_load_warnings"):
        st.caption("Warnings are recorded in the inspection page and the exported report.")
