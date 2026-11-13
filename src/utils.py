"""
utils.py
--------
Small shared helper functions used across the loan default prediction project.
"""

import os
import random
import numpy as np

RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    """Set random seeds for reproducibility across libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def project_root() -> str:
    """Return the absolute path to the project root (parent of src/)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def path_from_root(*parts) -> str:
    """Build an absolute path relative to the project root."""
    return os.path.join(project_root(), *parts)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    """
    Element-wise division that avoids divide-by-zero / inf / NaN.

    Any position where the denominator is 0, NaN, or missing is replaced
    with `fill_value` instead of raising or producing inf/NaN.
    """
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(
            (denominator == 0) | np.isnan(denominator),
            fill_value,
            numerator / denominator,
        )
    result = np.where(np.isnan(result) | np.isinf(result), fill_value, result)
    return result


def print_section(title: str) -> None:
    """Print a visually distinct section header (used for console reports)."""
    bar = "=" * 70
    print(f"\n{bar}\n{title}\n{bar}")
