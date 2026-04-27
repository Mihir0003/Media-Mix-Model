"""Feature engineering utilities for Media Mix Modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np


EPSILON = 1e-8


@dataclass
class TransformConfig:
    """Holds per-channel transformation parameters."""

    theta: float
    alpha: float
    gamma: float


def geometric_adstock(spend: np.ndarray, theta: float) -> np.ndarray:
    """Apply geometric adstock for carryover effects."""
    if not 0.0 <= theta < 1.0:
        raise ValueError(f"theta must be in [0, 1). Got {theta}")

    spend = np.asarray(spend, dtype=float)
    result = np.zeros_like(spend, dtype=float)
    carry = 0.0
    for i, value in enumerate(spend):
        carry = value + theta * carry
        result[i] = carry
    return result


def hill_saturation(x: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Apply Hill saturation curve for diminishing returns."""
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")

    x = np.asarray(x, dtype=float)
    x = np.clip(x, 0.0, None)
    numerator = np.power(x, alpha)
    denominator = numerator + np.power(gamma, alpha) + EPSILON
    return numerator / denominator


def _resolve_channel_config(
    channels: List[str], channel_params: Dict[str, Dict[str, float]]
) -> Dict[str, TransformConfig]:
    config = {}
    for channel in channels:
        params = channel_params[channel]
        config[channel] = TransformConfig(
            theta=float(params["theta"]),
            alpha=float(params["alpha"]),
            gamma=float(params["gamma"]),
        )
    return config


def transform_channels(
    spend_matrix: np.ndarray,
    channels: List[str],
    channel_params: Dict[str, Dict[str, float]],
    scale_factors: Dict[str, float],
) -> np.ndarray:
    """Apply adstock + Hill transformation channel-wise."""
    spend_matrix = np.asarray(spend_matrix, dtype=float)
    if spend_matrix.ndim != 2:
        raise ValueError("spend_matrix must be 2D")
    if spend_matrix.shape[1] != len(channels):
        raise ValueError("spend_matrix column count must match channels length")

    cfg = _resolve_channel_config(channels, channel_params)
    transformed = np.zeros_like(spend_matrix, dtype=float)

    for idx, channel in enumerate(channels):
        scale = max(scale_factors.get(channel, 1.0), EPSILON)
        scaled_spend = spend_matrix[:, idx] / scale
        adstocked = geometric_adstock(scaled_spend, cfg[channel].theta)
        transformed[:, idx] = hill_saturation(adstocked, cfg[channel].alpha, cfg[channel].gamma)

    return transformed


def compute_scale_factors(spend_matrix: np.ndarray, channels: List[str]) -> Dict[str, float]:
    """Compute channel scaling factors for numerical stability."""
    spend_matrix = np.asarray(spend_matrix, dtype=float)
    if spend_matrix.shape[1] != len(channels):
        raise ValueError("spend_matrix column count must match channels length")

    result: Dict[str, float] = {}
    for idx, channel in enumerate(channels):
        col_max = float(np.nanmax(spend_matrix[:, idx]))
        result[channel] = col_max if col_max > EPSILON else 1.0
    return result


def default_channel_params(channels: List[str]) -> Dict[str, Dict[str, float]]:
    """Deterministic initial values for optimization."""
    return {
        channel: {"theta": 0.40, "alpha": 1.30, "gamma": 0.60}
        for channel in channels
    }
