"""Budget optimization for MMM-derived response functions."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .modeling import MMMModel
from .visualization import plot_budget_comparison


def _select_channels(available_channels: List[str]) -> List[str]:
    lower_map = {ch.lower(): ch for ch in available_channels}
    preferred = ["search", "social", "tv"]
    selected = [lower_map[p] for p in preferred if p in lower_map]

    if len(selected) < 3:
        for channel in available_channels:
            if channel not in selected:
                selected.append(channel)
            if len(selected) == 3:
                break

    if not selected:
        raise ValueError("No channels available for optimization")

    return selected


def optimize_budget_allocation(
    model: MMMModel,
    df: pd.DataFrame,
    total_budget: float = 200000.0,
) -> Dict[str, object]:
    channels = _select_channels(model.channels)

    recent = df.tail(12)
    avg_spend = recent[channels].mean().to_numpy(dtype=float)
    if np.sum(avg_spend) <= 0:
        current_alloc = np.ones(len(channels)) * (total_budget / len(channels))
    else:
        current_shares = avg_spend / np.sum(avg_spend)
        current_alloc = current_shares * total_budget

    bounds = [(0.0, total_budget) for _ in channels]
    constraints = [{"type": "eq", "fun": lambda x: np.sum(x) - total_budget}]

    def objective(x: np.ndarray) -> float:
        allocation = {channel: float(x[idx]) for idx, channel in enumerate(channels)}
        revenue = sum(model.static_channel_response(channel, allocation[channel]) for channel in channels)
        return -float(revenue)

    result = minimize(
        objective,
        x0=current_alloc,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 1200, "ftol": 1e-10},
    )

    if not result.success:
        raise RuntimeError(f"Budget optimization failed: {result.message}")

    optimized_alloc = result.x

    current_response = sum(
        model.static_channel_response(channels[i], current_alloc[i]) for i in range(len(channels))
    )
    optimized_response = sum(
        model.static_channel_response(channels[i], optimized_alloc[i]) for i in range(len(channels))
    )

    uplift_pct = (optimized_response - current_response) / max(abs(current_response), 1e-8) * 100.0

    comparison_df = pd.DataFrame(
        {
            "channel": channels,
            "current_spend": current_alloc,
            "optimized_spend": optimized_alloc,
        }
    )
    comparison_df["delta_spend"] = comparison_df["optimized_spend"] - comparison_df["current_spend"]
    comparison_df["delta_pct"] = comparison_df["delta_spend"] / comparison_df["current_spend"].replace(0, np.nan) * 100.0

    return {
        "channels": channels,
        "comparison_df": comparison_df,
        "current_predicted_revenue": float(current_response),
        "optimized_predicted_revenue": float(optimized_response),
        "uplift_pct": float(uplift_pct),
        "optimizer_message": str(result.message),
        "optimizer_success": bool(result.success),
    }


def write_optimization_report(output_path: Path, optimization_result: Dict[str, object], budget: float) -> None:
    comparison_df = optimization_result["comparison_df"]

    lines = []
    lines.append("# Budget Optimization Results")
    lines.append("")
    lines.append("## Optimization Setup")
    lines.append(f"- Total budget constraint: ${budget:,.0f}")
    lines.append("- Objective: maximize predicted media-driven revenue")
    lines.append("- Solver: SLSQP")
    lines.append(f"- Optimizer success: {optimization_result['optimizer_success']}")
    lines.append(f"- Optimizer message: {optimization_result['optimizer_message']}")
    lines.append("")

    lines.append("## Current vs Optimized Allocation")
    lines.append("| Channel | Current Spend | Optimized Spend | Delta Spend | Delta % |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in comparison_df.iterrows():
        lines.append(
            "| {channel} | ${current:,.0f} | ${optimized:,.0f} | ${delta:,.0f} | {delta_pct:.2f}% |".format(
                channel=row["channel"],
                current=row["current_spend"],
                optimized=row["optimized_spend"],
                delta=row["delta_spend"],
                delta_pct=(0.0 if np.isnan(row["delta_pct"]) else row["delta_pct"]),
            )
        )

    lines.append("")
    lines.append("## Predicted Outcome")
    lines.append(
        "- Current-allocation predicted media response: "
        f"{optimization_result['current_predicted_revenue']:,.2f}"
    )
    lines.append(
        "- Optimized-allocation predicted media response: "
        f"{optimization_result['optimized_predicted_revenue']:,.2f}"
    )
    lines.append(f"- Estimated uplift: {optimization_result['uplift_pct']:.2f}%")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_budget_plot(project_root: Path, optimization_result: Dict[str, object]) -> None:
    comparison_df = optimization_result["comparison_df"]
    output_path = project_root / "visuals" / "budget_comparison.png"
    plot_budget_comparison(comparison_df, output_path)
