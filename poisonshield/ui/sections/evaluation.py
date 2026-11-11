"""Detection evaluation and model-performance comparison."""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from backend.config import MAX_IMAGE_COUNT, UPLOAD_DIR, ensure_workspace
from backend.evaluation.metrics import (
    SIGNAL_LABELS,
    available_ground_truth_columns,
    binarize_truth,
    duplicate_accuracy_report,
    evaluate_all_signals,
    evaluate_detection,
    ground_truth_series,
)
from backend.evaluation.model_comparison import compare_embedding_models, compare_models
from backend.features.image_features import extract_image_features
from backend.ingestion.images import (
    ImageLoadError,
    build_image_dataset,
    load_image_folder,
    load_mapping_from_bytes,
    load_uploaded_images,
    resolve_mapping_columns,
)
from backend.ingestion.tabular import TabularLoadError, read_tabular
from backend.intelligence.sanitizer import build_tabular_cleaned
from backend.utils import format_duration, human_bytes, safe_extract_zip, timestamp_slug

from .. import state
from ..components import (
    callout,
    fmt,
    render_detector_status,
    render_no_analysis,
    render_no_dataset,
)
from ..theme import hero


def render() -> None:
    hero(
        "Model / Quality Evaluation",
        "Measure detectors against real ground truth and compare models on a genuinely separate holdout.",
        kicker="PoisonShield · Evaluation",
    )
    dataset = state.current_dataset()
    if dataset is None:
        render_no_dataset()
        return
    bundle = state.current_bundle()
    if bundle is None:
        render_no_analysis(dataset)
        return

    tabs = st.tabs(["Detection metrics", "Model comparison", "Quality & limitations"])
    with tabs[0]:
        _render_detection_evaluation(dataset, bundle)
    with tabs[1]:
        _render_model_comparison(dataset, bundle)
    with tabs[2]:
        _render_quality(dataset, bundle)


def _render_detection_evaluation(dataset, bundle) -> None:
    callout(
        "Detection metrics require a real ground-truth column supplied with the data. PoisonShield "
        "does not inject artificial poisoning or invent labels for this page.",
        "info",
    )
    options = available_ground_truth_columns(dataset)
    if not options:
        st.warning(
            "No suitable ground-truth column is available. Add a binary column (for example "
            "is_poisoned with values 0/1 or clean/poisoned) and reload the dataset."
        )
        return
    default = dataset.ground_truth_column if dataset.ground_truth_column in options else options[0]
    selected = st.selectbox(
        "Ground-truth column",
        options,
        index=options.index(default),
        key="ps_eval_truth",
    )
    try:
        raw_truth = ground_truth_series(dataset, selected)
        truth, info = binarize_truth(raw_truth)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Could not interpret the ground-truth column: {exc}")
        return
    st.caption(
        f"Mapped {info['n_mapped']:,} labelled ground-truth values; "
        f"{info['n_unmapped']:,} unmapped/missing values are excluded from metrics."
    )
    if info.get("unmapped_examples"):
        st.caption("Unmapped examples: " + ", ".join(f"{key} ({count})" for key, count in info["unmapped_examples"].items()))
    if dataset.modality == "tabular" and dataset.ground_truth_column != selected:
        callout(
            "This column was not configured as the evaluation ground truth before analysis. If it is "
            "still present in the feature representation, metrics may be contaminated by leakage. "
            "Set it as ground truth in Upload and re-run analysis for a clean evaluation.",
            "danger",
        )
        if st.button("Exclude this column and re-run analysis", key="ps_eval_fix_ground"):
            dataset.ground_truth_column = selected
            state.clear_analysis(keep_dataset=True)
            state.flash(
                "Ground-truth selection updated. Open Upload or Overview to run the analysis again.",
                "info",
            )
            st.rerun()

    threshold = 0.5
    if st.checkbox("Use a custom risk-score threshold", key="ps_eval_custom_thr"):
        threshold = st.slider("Risk threshold", 0.0, 1.0, 0.5, 0.01, key="ps_eval_thr")
    table = evaluate_all_signals(bundle, truth, threshold=threshold)
    st.markdown("#### Detection performance by signal")
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        column_config={
            "Precision": st.column_config.NumberColumn(format="%.3f"),
            "Recall": st.column_config.NumberColumn(format="%.3f"),
            "F1": st.column_config.NumberColumn(format="%.3f"),
            "Accuracy": st.column_config.NumberColumn(format="%.3f"),
            "ROC AUC": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    st.caption(
        "Precision = flagged ground-truth positives / all flagged; recall = found positives / all "
        "positives. AUC is computed only for continuous risk-score signals when both classes are present."
    )

    signal_key = st.selectbox(
        "Inspect one signal",
        list(SIGNAL_LABELS.keys()),
        format_func=lambda key: SIGNAL_LABELS[key],
        key="ps_eval_signal",
    )
    try:
        evaluation = evaluate_detection(bundle, truth, signal_key, threshold=threshold)
    except Exception as exc:  # noqa: BLE001
        st.error(str(exc))
        return
    metrics = st.columns(5)
    with metrics[0]:
        st.metric("Precision", fmt(evaluation.precision, 3))
    with metrics[1]:
        st.metric("Recall", fmt(evaluation.recall, 3))
    with metrics[2]:
        st.metric("F1", fmt(evaluation.f1, 3))
    with metrics[3]:
        st.metric("Accuracy", fmt(evaluation.accuracy, 3))
    with metrics[4]:
        st.metric("ROC AUC", fmt(evaluation.roc_auc, 3))
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Confusion matrix")
        st.dataframe(evaluation.confusion_frame(), width="stretch")
    with col2:
        st.markdown("#### Evaluation notes")
        if evaluation.notes:
            for note in evaluation.notes:
                st.markdown(f"- {note}")
        st.caption(
            f"Evaluated {evaluation.n_evaluated:,} samples with "
            f"{evaluation.n_positive:,} ground-truth positives and {evaluation.n_flagged:,} flags."
        )
        st.markdown("#### Duplicate-detection quality")
        duplicates = duplicate_accuracy_report(bundle, truth)
        if duplicates.get("available"):
            st.write(
                {
                    "precision": fmt(duplicates.get("precision"), 3),
                    "recall": fmt(duplicates.get("recall"), 3),
                    "f1": fmt(duplicates.get("f1"), 3),
                    "accuracy": fmt(duplicates.get("accuracy"), 3),
                }
            )
        else:
            st.caption(duplicates.get("reason", "No duplicate evaluation was possible."))

    st.markdown("#### Metric interpretation")
    st.markdown(
        "- Precision is often more important when false alarms are costly; recall matters when missing "
        "poisoned samples is costly. No single threshold is universally correct.\n"
        "- Metrics depend on the quality of the ground-truth column and on class balance.\n"
        "- A detector that did not run is not a correct negative; check the detector status below."
    )
    with st.expander("Detector status", expanded=False):
        render_detector_status(bundle)


def _render_model_comparison(dataset, bundle) -> None:
    callout(
        "Use a separate, untouched evaluation dataset. Sanitisation may remove genuine signal, so the "
        "measured delta is reported as-is and never assumed to be an improvement.",
        "info",
    )
    if not dataset.has_labels or not dataset.label_column:
        st.warning("Model comparison requires a labelled dataset.")
        return
    if dataset.task not in {"classification", "regression"}:
        st.warning("Model comparison is available for classification or regression datasets.")
        return
    store = state.current_store()
    if store is None:
        st.warning("No review store is attached to this session.")
        return

    if dataset.modality == "tabular":
        _tabular_comparison(dataset, bundle, store)
    else:
        _image_comparison(dataset, bundle, store)

    comparison = state.current_comparison()
    if comparison is None:
        return
    st.divider()
    st.markdown("#### Comparison result")
    metric_names = list(comparison.original_metrics.keys())
    rows = []
    for metric in metric_names:
        original = comparison.original_metrics.get(metric)
        cleaned = comparison.cleaned_metrics.get(metric)
        delta = comparison.delta.get(metric)
        rows.append(
            {
                "Metric": metric,
                "Original labels": original,
                "Reviewed labels": cleaned,
                "Delta (reviewed − original)": delta,
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        column_config={
            "Original labels": st.column_config.NumberColumn(format="%.4f"),
            "Reviewed labels": st.column_config.NumberColumn(format="%.4f"),
            "Delta (reviewed − original)": st.column_config.NumberColumn(format="%.4f"),
        },
    )
    st.caption(
        f"Training rows: {comparison.n_train_original:,} original → {comparison.n_train_cleaned:,} reviewed · "
        f"evaluation rows: {comparison.n_test:,} · task: {comparison.task}"
    )
    if comparison.per_class is not None and not comparison.per_class.empty:
        st.markdown("#### Per-class recall")
        st.dataframe(comparison.per_class, width="stretch", hide_index=True)
    for note in comparison.notes:
        st.caption(note)
    st.markdown("#### Interpretation guardrails")
    st.markdown(
        "- This is one model family (logistic regression / ridge) on one feature representation and one split.\n"
        "- The evaluation dataset is never used to fit the classifier or the feature representation.\n"
        "- A negative delta is a real possible outcome and must be reported, not hidden.\n"
        "- Detection metrics and model-performance metrics answer different questions."
    )


def _tabular_comparison(dataset, bundle, store) -> None:
    upload = st.file_uploader(
        "Untouched evaluation table (CSV / Excel)",
        type=["csv", "xlsx", "xls"],
        key="ps_eval_tab_file",
    )
    if upload is not None and st.button("Read evaluation table", key="ps_eval_read_tab"):
        ensure_workspace()
        folder = UPLOAD_DIR / f"eval_tab_{timestamp_slug()}"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / upload.name
        path.write_bytes(upload.getvalue())
        try:
            frame = read_tabular(path)
        except TabularLoadError as exc:
            st.error(f"Could not read the evaluation table: {exc}")
            return
        st.session_state["ps_eval_holdout_df"] = frame
        st.session_state["ps_eval_holdout_name"] = upload.name
        st.session_state["ps_eval_holdout_path"] = str(path)
        st.session_state["ps_eval_label_col"] = (
            dataset.label_column if dataset.label_column in frame.columns else frame.columns[-1]
        )
        state.flash("Evaluation table loaded.", "success")
        st.rerun()

    holdout = st.session_state.get("ps_eval_holdout_df")
    if holdout is None:
        st.caption("Upload a separate table to enable model comparison.")
        return
    st.markdown(f"**Evaluation file:** {st.session_state.get('ps_eval_holdout_name')} · {len(holdout):,} rows")
    label_options = list(holdout.columns)
    default_label = st.session_state.get("ps_eval_label_col", label_options[-1])
    label_index = label_options.index(default_label) if default_label in label_options else len(label_options) - 1
    label_column = st.selectbox(
        "Evaluation label column",
        label_options,
        index=label_index,
        key="ps_eval_label_select",
    )
    st.dataframe(holdout.head(10), width="stretch", hide_index=True)

    before = _row_hash_overlap(dataset.dataframe, holdout, dataset.label_column, label_column)
    if before:
        callout(
            f"{before:,} evaluation row(s) are identical to training rows across the shared feature "
            "columns. This is not an untouched holdout; results will be optimistic.",
            "danger",
        )
    confirmed_holdout = st.checkbox(
        "I confirm this evaluation table is separate from the training data",
        key="ps_eval_confirm_tab",
    )
    if st.button(
        "Compare original and reviewed models",
        type="primary",
        key="ps_eval_compare_tab",
        disabled=not confirmed_holdout,
    ):
        try:
            cleanup = build_tabular_cleaned(bundle, store)
            result = compare_models(
                dataset.dataframe,
                cleanup.dataframe,
                holdout,
                label_column=label_column,
                feature_transform=bundle.features.preprocessor,
                task=dataset.task,
                modality="tabular",
                random_state=int(st.session_state.get("ps_seed", 42)),
            )
            st.session_state["ps_comparison"] = result
            state.flash("Model comparison completed.", "success")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Model comparison failed: {exc}")
            st.caption(
                "Ensure the evaluation table contains the same feature columns and labels used by the "
                "analysis. The feature transformer is fitted only on the original training data."
            )


def _image_comparison(dataset, bundle, store) -> None:
    upload = st.file_uploader("Untouched evaluation image ZIP", type=["zip"], key="ps_eval_img_zip")
    files = st.file_uploader(
        "…or separate evaluation images",
        type=["jpg", "jpeg", "png", "bmp", "webp", "tif", "tiff", "gif"],
        accept_multiple_files=True,
        key="ps_eval_img_files",
    )
    mapping_upload = st.file_uploader("Evaluation label mapping", type=["csv", "xlsx", "xls"], key="ps_eval_img_map")
    infer_folder = st.checkbox("Infer evaluation labels from folder names", value=True, key="ps_eval_img_folders")
    if upload is None and not files:
        st.caption("Upload a separate labelled image dataset to enable model comparison.")
        return
    if st.button("Load evaluation images", key="ps_eval_load_images"):
        ensure_workspace()
        folder = UPLOAD_DIR / f"eval_img_{timestamp_slug()}"
        folder.mkdir(parents=True, exist_ok=True)
        try:
            mapping = load_mapping_from_bytes(mapping_upload.name, mapping_upload.getvalue()) if mapping_upload else None
            if upload is not None:
                archive_path = folder / "source.zip"
                archive_path.write_bytes(upload.getvalue())
                root = safe_extract_zip(archive_path, folder / "extracted")
                auto_mapping = None
                if mapping is None:
                    candidates = [
                        path for path in root.rglob("*")
                        if path.is_file() and path.suffix.lower() in {".csv", ".xlsx", ".xls"}
                    ]
                    if len(candidates) == 1:
                        try:
                            candidate = load_mapping_from_bytes(candidates[0].name, candidates[0].read_bytes())
                            resolve_mapping_columns(candidate)
                            mapping = candidate
                            auto_mapping = candidates[0].name
                        except Exception:
                            mapping = None
                records, warnings, extra_columns = load_image_folder(
                    root,
                    max_images=MAX_IMAGE_COUNT,
                    mapping=mapping,
                    infer_folder_labels=infer_folder,
                )
            else:
                payloads = [(item.name, item.getvalue()) for item in files or []]
                records, warnings, extra_columns = load_uploaded_images(
                    payloads,
                    folder / "source",
                    max_images=MAX_IMAGE_COUNT,
                    mapping=mapping,
                    infer_folder_labels=False,
                )
            holdout = build_image_dataset(
                records,
                source_name="evaluation image set",
                extra_columns=extra_columns,
                notes=warnings,
            )
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not load the evaluation images: {exc}")
            return
        if not holdout.has_labels:
            st.error(
                "The evaluation image dataset needs at least two labelled images. Upload a mapping with "
                "a label column or use labelled folder names."
            )
            return
        st.session_state["ps_eval_img_dataset"] = holdout
        state.flash("Evaluation images loaded. Run the comparison to extract embeddings.", "success")
        st.rerun()

    holdout = st.session_state.get("ps_eval_img_dataset")
    if holdout is None:
        return
    st.markdown(f"**Evaluation images:** {holdout.profile.n_samples:,} · labels: {holdout.profile.class_distribution}")
    used_extractor = (bundle.features.meta or {}).get("extractor", "auto")
    st.caption(f"Evaluation embeddings will use the same extractor as the analysis: {used_extractor}.")

    confirmed_holdout = st.checkbox(
        "I confirm this evaluation image set is separate from the training data",
        key="ps_eval_confirm_img",
    )
    if confirmed_holdout and st.button(
        "Extract evaluation embeddings and compare models",
        type="primary",
        key="ps_eval_compare_img",
    ):
        progress = st.progress(0.0)
        text = st.empty()

        def report(fraction: float, message: str) -> None:
            progress.progress(float(max(0.0, min(1.0, fraction))))
            text.caption(message)

        try:
            features = extract_image_features(
                holdout.images or [],
                extractor=used_extractor,
                batch_size=int(st.session_state.get("ps_batch", 32)),
                progress=report,
            )
            test_labels = holdout.labels().reindex(features.sample_ids)
            original_labels = dataset.labels().reindex(bundle.features.sample_ids)
            corrected = {
                sid: decision.corrected_label
                for sid, decision in store.decisions.items()
                if decision.action == "approve_correction" and decision.corrected_label is not None
            }
            removed = {sid for sid, decision in store.decisions.items() if decision.action == "remove"}
            training_hashes = {record.sha256 for record in (dataset.images or []) if record.sha256}
            holdout_hashes = {record.sha256 for record in (holdout.images or []) if record.sha256}
            overlap = training_hashes & holdout_hashes
            if overlap:
                callout(
                    f"{len(overlap):,} evaluation image file(s) are byte-identical to training images. "
                    "This is not an untouched holdout; results will be optimistic.",
                    "danger",
                )
            result = compare_embedding_models(
                train_ids=bundle.features.sample_ids,
                train_matrix=bundle.features.matrix,
                original_labels=original_labels,
                test_ids=features.sample_ids,
                test_matrix=features.matrix,
                test_labels=test_labels,
                task=dataset.task,
                modality="image",
                removed_ids=removed,
                corrected_labels=corrected,
                random_state=int(st.session_state.get("ps_seed", 42)),
            )
            st.session_state["ps_comparison"] = result
            state.flash("Image model comparison completed.", "success")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Image model comparison failed: {exc}")
        finally:
            progress.progress(1.0)
            text.empty()


def _row_hash_overlap(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    train_label: str | None,
    holdout_label: str | None,
) -> int:
    common = [
        column
        for column in train.columns
        if column in holdout.columns and column not in {train_label, holdout_label}
    ]
    if not common:
        return 0
    train_hashes = set(pd.util.hash_pandas_object(train[common], index=False).tolist())
    holdout_hashes = pd.util.hash_pandas_object(holdout[common], index=False)
    return int(holdout_hashes.isin(train_hashes).sum())


def _render_quality(dataset, bundle) -> None:
    quality = bundle.risk.quality
    st.markdown("#### Quality summary")
    st.json({key: value for key, value in quality.items() if key not in {"notes"}})
    st.markdown("#### Detector coverage")
    render_detector_status(bundle)
    st.markdown("#### Timings and memory")
    st.write(
        {
            "timings": {key: format_duration(value) for key, value in bundle.timings.items()},
            "memory": bundle.memory,
        }
    )
    st.markdown("#### Ground-truth evaluation caveats")
    st.markdown(
        "- Detector precision/recall require a trustworthy, independent ground-truth column. If the column "
        "was created by the same process that produced the labels, metrics can be circular.\n"
        "- Image feature quality varies with domain; general-purpose embeddings may not capture "
        "specialised imagery.\n"
        "- The relationship engine uses neighbour similarity and label agreement; correlated errors in "
        "the neighbourhood can hide or amplify a signal.\n"
        "- Class-imbalance and non-IID are dataset-level findings and are excluded from per-sample risk.\n"
        "- Sanitisation may reduce model performance even when it removes genuine label noise."
    )
    if dataset.modality == "image":
        st.markdown("#### Image extractor note")
        st.write((bundle.features.meta or {}).get("domain_note"))
