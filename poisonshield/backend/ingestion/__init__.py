"""Dataset ingestion: format validation, loading and automatic inspection."""

from .images import (
    build_image_dataset,
    find_image_files,
    load_image_folder,
    load_image_zip,
    load_uploaded_images,
    suggest_image_label_column,
)
from .tabular import (
    build_tabular_dataset,
    infer_task,
    read_tabular,
    suggest_label_column,
)

__all__ = [
    "build_image_dataset",
    "build_tabular_dataset",
    "find_image_files",
    "infer_task",
    "load_image_folder",
    "load_image_zip",
    "load_uploaded_images",
    "read_tabular",
    "suggest_image_label_column",
    "suggest_label_column",
]
