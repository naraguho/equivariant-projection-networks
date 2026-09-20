"""Correlation and scaling-collapse definitions used in the manuscript."""

from __future__ import annotations

import numpy as np


def connected_fft_correlation(field: np.ndarray) -> np.ndarray:
    """Connected periodic 2D autocorrelation before radial averaging."""
    field = np.asarray(field, dtype=float)
    correlation = np.fft.ifft2(np.abs(np.fft.fft2(field)) ** 2).real / field.size
    return correlation - field.mean() ** 2


def radial_average(correlation: np.ndarray, rmax: int | None = None):
    """Average a periodic 2D correlation over integer-radius shells."""
    length = correlation.shape[0]
    if correlation.shape != (length, length):
        raise ValueError("correlation must be square")
    rmax = length // 2 if rmax is None else rmax
    coordinate = np.arange(length)
    coordinate = np.minimum(coordinate, length - coordinate)
    yy, xx = np.meshgrid(coordinate, coordinate, indexing="ij")
    shell = np.rint(np.sqrt(xx**2 + yy**2)).astype(int)
    radii = np.arange(rmax + 1)
    values = np.array([correlation[shell == radius].mean() for radius in radii])
    return radii.astype(float), values


def half_height_length(correlation, radii, target: float = 0.5) -> float:
    """First linearly interpolated crossing of ``C(r)/C(0)=target``."""
    normalized = np.asarray(correlation, float) / correlation[0]
    for i in range(1, len(normalized)):
        if normalized[i - 1] >= target >= normalized[i]:
            fraction = (target - normalized[i - 1]) / (
                normalized[i] - normalized[i - 1]
            )
            return float(radii[i - 1] + fraction * (radii[i] - radii[i - 1]))
    return float("nan")


def collapse_coordinates(correlation, radii):
    """Return ``r/L(t)`` and ``C(r,t)/C(0,t)``."""
    length = half_height_length(correlation, radii)
    return np.asarray(radii) / length, np.asarray(correlation) / correlation[0], length

