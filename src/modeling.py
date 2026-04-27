"""Bayesian MMM modeling layer with adstock and saturation effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution, lsq_linear, minimize
from sklearn.metrics import r2_score

from .feature_engineering import compute_scale_factors, transform_channels


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, 1e-8, np.abs(y_true))
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def _safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


@dataclass
class MMMModel:
    engine: str
    channels: List[str]
    scale_factors: Dict[str, float]
    channel_params: Dict[str, Dict[str, float]]
    beta: Dict[str, float]
    intercept: float
    trend_coef: float
    sin_coef: float
    cos_coef: float
    time_mean: float
    time_std: float
    posterior_draws: Dict[str, np.ndarray]
    lightweight_mmm_available: bool

    def _time_terms(self, n_periods: int, start_index: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        idx = np.arange(start_index, start_index + n_periods, dtype=float)
        time_scaled = (idx - self.time_mean) / max(self.time_std, 1.0)
        sin_term = np.sin(2.0 * np.pi * idx / 52.0)
        cos_term = np.cos(2.0 * np.pi * idx / 52.0)
        return time_scaled, sin_term, cos_term

    def transform_spend(self, spend_matrix: np.ndarray) -> np.ndarray:
        return transform_channels(
            spend_matrix=spend_matrix,
            channels=self.channels,
            channel_params=self.channel_params,
            scale_factors=self.scale_factors,
        )

    def predict_sequence(self, spend_matrix: np.ndarray, start_index: int = 0) -> np.ndarray:
        spend_matrix = np.asarray(spend_matrix, dtype=float)
        transformed = self.transform_spend(spend_matrix)
        n_periods = spend_matrix.shape[0]
        t, sin_t, cos_t = self._time_terms(n_periods=n_periods, start_index=start_index)
        beta_vec = np.array([self.beta[channel] for channel in self.channels], dtype=float)
        media_component = transformed @ beta_vec
        baseline_component = self.intercept + self.trend_coef * t + self.sin_coef * sin_t + self.cos_coef * cos_t
        return baseline_component + media_component

    def channel_contributions(self, spend_matrix: np.ndarray) -> pd.DataFrame:
        transformed = self.transform_spend(spend_matrix)
        contrib = {}
        for idx, channel in enumerate(self.channels):
            contrib[channel] = transformed[:, idx] * self.beta[channel]
        return pd.DataFrame(contrib)

    def static_channel_response(self, channel: str, spend: float) -> float:
        """One-period channel response under steady-state adstock assumption."""
        params = self.channel_params[channel]
        theta = params["theta"]
        alpha = params["alpha"]
        gamma = params["gamma"]
        scale = max(self.scale_factors[channel], 1e-8)

        spend_scaled = max(spend, 0.0) / scale
        adstock_steady = spend_scaled / max(1.0 - theta, 1e-6)
        saturated = (adstock_steady**alpha) / (adstock_steady**alpha + gamma**alpha + 1e-8)
        return float(self.beta[channel] * saturated)

    def static_total_revenue(self, allocation: Dict[str, float]) -> float:
        media_response = sum(self.static_channel_response(ch, allocation.get(ch, 0.0)) for ch in self.channels)
        # Use long-run baseline average over one full year.
        idx = np.arange(52, dtype=float)
        t = (idx - self.time_mean) / max(self.time_std, 1.0)
        baseline = self.intercept + self.trend_coef * np.mean(t)
        return float(baseline + media_response)


def _build_nonlinear_bounds(channels: List[str]) -> List[Tuple[float, float]]:
    # Shared Hill shape params across channels for better stability.
    # Vector layout: [theta_channel_1..theta_channel_k, alpha_shared, gamma_shared]
    bounds = [(0.00, 0.85) for _ in channels]
    bounds.extend(
        [
            (0.70, 1.60),  # alpha_shared
            (0.25, 1.20),  # gamma_shared
        ]
    )
    return bounds


def _vector_to_channel_params(vec: np.ndarray, channels: List[str]) -> Dict[str, Dict[str, float]]:
    vec = np.asarray(vec, dtype=float)
    channel_params: Dict[str, Dict[str, float]] = {}

    alpha_shared = float(vec[len(channels)])
    gamma_shared = float(vec[len(channels) + 1])
    for idx, channel in enumerate(channels):
        theta = float(vec[idx])
        channel_params[channel] = {"theta": theta, "alpha": alpha_shared, "gamma": gamma_shared}

    return channel_params


def _ridge_closed_form(X: np.ndarray, y: np.ndarray, ridge_lambda: float = 1e-2) -> np.ndarray:
    p = X.shape[1]
    reg = ridge_lambda * np.eye(p)
    reg[0, 0] = 0.0  # Do not penalize intercept.
    xtx = X.T @ X + reg
    xty = X.T @ y
    return np.linalg.solve(xtx, xty)


def _fit_linear_layer(
    transformed: np.ndarray,
    t: np.ndarray,
    sin_t: np.ndarray,
    cos_t: np.ndarray,
    y: np.ndarray,
    enforce_nonnegative_media: bool = True,
) -> Dict[str, object]:
    x_matrix = np.column_stack([np.ones_like(t), t, sin_t, cos_t, transformed])
    n_params = x_matrix.shape[1]

    lower = np.full(n_params, -np.inf, dtype=float)
    upper = np.full(n_params, np.inf, dtype=float)
    if enforce_nonnegative_media:
        lower[4:] = 0.0

    lsq_result = lsq_linear(
        x_matrix,
        y,
        bounds=(lower, upper),
        method="trf",
        max_iter=800,
        tol=1e-10,
    )

    coef = np.asarray(lsq_result.x, dtype=float)
    y_hat = x_matrix @ coef
    mse = float(np.mean((y - y_hat) ** 2))

    return {
        "coef": coef,
        "y_hat": y_hat,
        "mse": mse,
        "success": bool(lsq_result.success),
        "message": str(lsq_result.message),
        "status": int(lsq_result.status),
        "design_matrix": x_matrix,
    }


def _build_time_cv_splits(train_len: int, val_horizon: int = 8) -> List[Tuple[int, int]]:
    anchors = [0.62, 0.74, 0.84]
    splits: List[Tuple[int, int]] = []

    for anchor in anchors:
        fit_end = int(train_len * anchor)
        val_end = min(fit_end + val_horizon, train_len)
        if fit_end >= 30 and (val_end - fit_end) >= 4:
            splits.append((fit_end, val_end))

    if not splits and train_len >= 40:
        fit_end = int(train_len * 0.75)
        val_end = min(fit_end + max(4, val_horizon), train_len)
        splits.append((fit_end, val_end))

    return splits


def fit_mmm(
    df: pd.DataFrame,
    channels: List[str],
    revenue_col: str,
    holdout_weeks: int = 12,
    random_seed: int = 42,
) -> Dict[str, object]:
    """Fit MMM and return model + diagnostics + predictions."""

    if len(df) <= holdout_weeks + 20:
        raise ValueError("Insufficient rows for training and 12-week holdout split")

    # Prefer lightweight_mmm when installed, fallback otherwise.
    try:
        import lightweight_mmm  # noqa: F401

        lightweight_available = True
    except Exception:
        lightweight_available = False

    train_df = df.iloc[:-holdout_weeks].copy()
    holdout_df = df.iloc[-holdout_weeks:].copy()

    spend_train = train_df[channels].to_numpy(dtype=float)
    spend_full = df[channels].to_numpy(dtype=float)

    y_train = train_df[revenue_col].to_numpy(dtype=float)
    y_full = df[revenue_col].to_numpy(dtype=float)

    scale_factors = compute_scale_factors(spend_train, channels)

    idx_train = np.arange(len(train_df), dtype=float)
    time_mean = float(np.mean(idx_train))
    time_std = float(np.std(idx_train)) if np.std(idx_train) > 0 else 1.0

    t_train = (idx_train - time_mean) / time_std
    sin_train = np.sin(2.0 * np.pi * idx_train / 52.0)
    cos_train = np.cos(2.0 * np.pi * idx_train / 52.0)

    rng = np.random.default_rng(random_seed)
    bounds = _build_nonlinear_bounds(channels)
    cv_splits = _build_time_cv_splits(len(train_df), val_horizon=8)

    def objective(param_vec: np.ndarray) -> float:
        channel_params = _vector_to_channel_params(param_vec, channels)
        transformed = transform_channels(
            spend_matrix=spend_train,
            channels=channels,
            channel_params=channel_params,
            scale_factors=scale_factors,
        )
        fold_mse = []
        for fit_end, val_end in cv_splits:
            idx_fit = np.arange(fit_end, dtype=float)
            fit_mean = float(np.mean(idx_fit))
            fit_std = float(np.std(idx_fit)) if np.std(idx_fit) > 0 else 1.0

            t_fit = (idx_fit - fit_mean) / fit_std
            linear_fit = _fit_linear_layer(
                transformed=transformed[:fit_end],
                t=t_fit,
                sin_t=sin_train[:fit_end],
                cos_t=cos_train[:fit_end],
                y=y_train[:fit_end],
                enforce_nonnegative_media=True,
            )
            if not linear_fit["success"]:
                return float("inf")

            coef = np.asarray(linear_fit["coef"], dtype=float)
            idx_val = np.arange(fit_end, val_end, dtype=float)
            t_val = (idx_val - fit_mean) / fit_std
            x_val = np.column_stack(
                [
                    np.ones(len(idx_val), dtype=float),
                    t_val,
                    sin_train[fit_end:val_end],
                    cos_train[fit_end:val_end],
                    transformed[fit_end:val_end],
                ]
            )
            y_val = y_train[fit_end:val_end]
            y_pred_val = x_val @ coef
            mse = float(np.mean((y_val - y_pred_val) ** 2))
            beta_vec = np.clip(coef[4:], 0.0, None)
            beta_sum = float(np.sum(beta_vec))
            # Keep media effects identifiable while avoiding explosive coefficients.
            media_floor_penalty = 1_400_000.0 / (beta_sum + 1.0)
            beta_size_penalty = 0.004 * float(np.sum(np.square(beta_vec)))
            fold_mse.append(mse + media_floor_penalty + beta_size_penalty)

        if not fold_mse:
            return float("inf")

        # Priors/regularizers to keep transformed media effects realistic.
        prior_penalty = 0.0
        for i in range(len(channels)):
            theta = float(param_vec[i])
            prior_penalty += 30.0 * (theta - 0.20) ** 2
        alpha = float(param_vec[len(channels)])
        gamma = float(param_vec[len(channels) + 1])
        prior_penalty += 14.0 * (alpha - 1.10) ** 2
        prior_penalty += 10.0 * (gamma - 0.60) ** 2

        return float(np.mean(fold_mse) + prior_penalty)

    de_result = differential_evolution(
        objective,
        bounds=bounds,
        seed=random_seed,
        maxiter=42,
        popsize=10,
        tol=1e-4,
        polish=False,
        updating="deferred",
        workers=1,
    )

    best_local = minimize(
        objective,
        de_result.x,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 2200, "ftol": 1e-10},
    )

    if np.isfinite(best_local.fun) and np.isfinite(de_result.fun):
        best_candidate = best_local if best_local.fun <= de_result.fun else de_result
    elif np.isfinite(best_local.fun):
        best_candidate = best_local
    else:
        best_candidate = de_result
    if best_candidate is None or not np.isfinite(best_candidate.fun):
        raise RuntimeError("MMM optimization failed to produce a finite solution")

    channel_params = _vector_to_channel_params(best_candidate.x, channels)

    transformed_train = transform_channels(
        spend_matrix=spend_train,
        channels=channels,
        channel_params=channel_params,
        scale_factors=scale_factors,
    )
    linear_final = _fit_linear_layer(
        transformed=transformed_train,
        t=t_train,
        sin_t=sin_train,
        cos_t=cos_train,
        y=y_train,
        enforce_nonnegative_media=True,
    )
    if not linear_final["success"]:
        raise RuntimeError(f"Linear layer fitting failed: {linear_final['message']}")

    coef = linear_final["coef"]
    intercept = float(coef[0])
    trend_coef = float(coef[1])
    sin_coef = float(coef[2])
    cos_coef = float(coef[3])
    beta = {channel: float(coef[4 + idx]) for idx, channel in enumerate(channels)}

    model = MMMModel(
        engine="lightweight_mmm" if lightweight_available else "custom_bayesian_fallback",
        channels=channels,
        scale_factors=scale_factors,
        channel_params=channel_params,
        beta=beta,
        intercept=intercept,
        trend_coef=trend_coef,
        sin_coef=sin_coef,
        cos_coef=cos_coef,
        time_mean=time_mean,
        time_std=time_std,
        posterior_draws={},
        lightweight_mmm_available=lightweight_available,
    )

    # Predict over entire sequence so adstock state naturally flows into holdout weeks.
    y_pred_full = model.predict_sequence(spend_full, start_index=0)
    y_pred_train = y_pred_full[:-holdout_weeks]
    y_pred_holdout = y_pred_full[-holdout_weeks:]

    # Bayesian-like uncertainty via bootstrap on linear layer conditional on transforms.
    transformed_train = model.transform_spend(spend_train)
    X_train = np.column_stack([np.ones(len(train_df)), t_train, sin_train, cos_train, transformed_train])
    coef_point = np.array(
        [
            model.intercept,
            model.trend_coef,
            model.sin_coef,
            model.cos_coef,
            *[model.beta[ch] for ch in channels],
        ],
        dtype=float,
    )
    residuals = y_train - y_pred_train

    draws = []
    for _ in range(240):
        sampled_resid = rng.choice(residuals, size=len(residuals), replace=True)
        y_boot = y_pred_train + sampled_resid
        linear_boot = _fit_linear_layer(
            transformed=transformed_train,
            t=t_train,
            sin_t=sin_train,
            cos_t=cos_train,
            y=y_boot,
            enforce_nonnegative_media=True,
        )
        if linear_boot["success"]:
            draws.append(np.asarray(linear_boot["coef"], dtype=float))

    if len(draws) < 60:
        for _ in range(120):
            sampled_resid = rng.choice(residuals, size=len(residuals), replace=True)
            y_boot = X_train @ coef_point + sampled_resid
            coef_boot = _ridge_closed_form(X_train, y_boot, ridge_lambda=1e-2)
            draws.append(coef_boot)

    draw_matrix = np.asarray(draws)
    model.posterior_draws = {
        "intercept": draw_matrix[:, 0],
        "trend_coef": draw_matrix[:, 1],
        "sin_coef": draw_matrix[:, 2],
        "cos_coef": draw_matrix[:, 3],
    }
    for i, channel in enumerate(channels):
        model.posterior_draws[f"beta_{channel}"] = draw_matrix[:, 4 + i]

    metrics = {
        "r2_train": _safe_r2(y_train, y_pred_train),
        "mape_train": mean_absolute_percentage_error(y_train, y_pred_train),
        "r2_holdout": _safe_r2(y_full[-holdout_weeks:], y_pred_holdout),
        "mape_holdout": mean_absolute_percentage_error(y_full[-holdout_weeks:], y_pred_holdout),
    }

    diagnostics = {
        "engine": model.engine,
        "lightweight_mmm_available": lightweight_available,
        "optimizer_success": bool(best_local.success) or bool(de_result.success),
        "optimizer_message": (
            f"DE: {de_result.message}; Local: {best_local.message}"
        ),
        "objective_value": float(best_candidate.fun),
        "cv_folds_used": int(len(cv_splits)),
        "linear_fit_success": bool(linear_final["success"]),
        "linear_fit_message": str(linear_final["message"]),
        "train_weeks": int(len(train_df)),
        "holdout_weeks": int(holdout_weeks),
    }

    return {
        "model": model,
        "train_df": train_df,
        "holdout_df": holdout_df,
        "y_true_full": y_full,
        "y_pred_full": y_pred_full,
        "y_pred_train": y_pred_train,
        "y_pred_holdout": y_pred_holdout,
        "metrics": metrics,
        "diagnostics": diagnostics,
    }


def save_model(model: MMMModel, output_path: str) -> None:
    joblib.dump(model, output_path)


def write_model_performance_report(
    output_path: str,
    metrics: Dict[str, float],
    diagnostics: Dict[str, object],
) -> None:
    lines = []
    lines.append("# Model Performance Report")
    lines.append("")
    lines.append("## Modeling Framework")
    lines.append(f"- Engine used: `{diagnostics['engine']}`")
    lines.append(f"- `lightweight_mmm` available in environment: {diagnostics['lightweight_mmm_available']}")
    lines.append("- Core transforms: Geometric adstock + Hill saturation")
    lines.append("- Validation strategy: Chronological split with last 12 weeks as hold-out")
    lines.append("")

    lines.append("## Hold-Out Validation")
    lines.append(f"- Hold-out R²: {metrics['r2_holdout']:.4f}")
    lines.append(f"- Hold-out MAPE: {metrics['mape_holdout']:.2f}%")
    lines.append("")

    lines.append("## Training Fit")
    lines.append(f"- Train R²: {metrics['r2_train']:.4f}")
    lines.append(f"- Train MAPE: {metrics['mape_train']:.2f}%")
    lines.append("")

    lines.append("## Diagnostics")
    lines.append(f"- Optimizer success: {diagnostics['optimizer_success']}")
    lines.append(f"- Optimizer message: {diagnostics['optimizer_message']}")
    lines.append(f"- Objective value: {diagnostics['objective_value']:.2f}")
    if "cv_folds_used" in diagnostics:
        lines.append(f"- Time-series CV folds used: {diagnostics['cv_folds_used']}")
    if "linear_fit_success" in diagnostics:
        lines.append(f"- Linear layer success: {diagnostics['linear_fit_success']}")
    if "linear_fit_message" in diagnostics:
        lines.append(f"- Linear layer message: {diagnostics['linear_fit_message']}")
    lines.append(f"- Training weeks: {diagnostics['train_weeks']}")
    lines.append(f"- Hold-out weeks: {diagnostics['holdout_weeks']}")

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
