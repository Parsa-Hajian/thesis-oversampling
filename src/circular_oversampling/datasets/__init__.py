"""Dataset loading and metadata registry."""

from src.datasets.loader import download_dataset, load_dataset
from src.datasets.registry import (
    DATASETS,
    filter_datasets,
    get_dataset_info,
    list_datasets,
    list_datasets_by_ir,
)

__all__ = [
    "load_dataset",
    "download_dataset",
    "DATASETS",
    "get_dataset_info",
    "list_datasets",
    "list_datasets_by_ir",
    "filter_datasets",
]
