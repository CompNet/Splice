from __future__ import annotations
import numpy as np


def mean_noNaN(series: list | np.ndarray) -> float:
    """Compute the mean of a series, ignoring NaN values."""
    if isinstance(series, list):
        series = np.array(series)
    has_nan = np.ma.masked_invalid(series).mask
    return series[~has_nan].mean()


def mean_noNaN_dictreduce(d: dict[str, list | np.ndarray]) -> dict[str, float]:
    return {k: mean_noNaN(v) for k, v in d.items()}
