"""Utilities for generating and processing datasets."""

from .synthetic import (
    gen_regression_data,
    gen_classification_data,
    gen_sparse_regression_problem,
)
from .transforms import unitize_columns


__all__ = [
    "gen_classification_data",
    "gen_regression_data",
    "gen_sparse_regression_problem",
    "unitize_columns",
]
