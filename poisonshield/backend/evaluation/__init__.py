"""Evaluation helpers."""
from .metrics import (
    SIGNAL_LABELS,
    available_ground_truth_columns,
    binarize_truth,
    detection_predictions,
    duplicate_accuracy_report,
    evaluate_all_signals,
    evaluate_detection,
    ground_truth_series,
)
from .model_comparison import ModelComparisonResult, compare_embedding_models, compare_models

__all__ = [
    "SIGNAL_LABELS",
    "ModelComparisonResult",
    "available_ground_truth_columns",
    "binarize_truth",
    "compare_embedding_models",
    "compare_models",
    "detection_predictions",
    "duplicate_accuracy_report",
    "evaluate_all_signals",
    "evaluate_detection",
    "ground_truth_series",
]
