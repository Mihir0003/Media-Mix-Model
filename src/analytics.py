"""ROI and contribution analytics for trained MMM."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .modeling import MMMModel
from .visualization import plot_response_curve


def calculate_channel_metrics(model: MMMModel, df: pd.DataFrame, channels: List[str]) -> pd.DataFrame:
    spend_matrix = df[channels].to_numpy(dtype=float)
    contributions = model.channel_contributions(spend_matrix)

    rows = []
    for channel in channels:
        total_spend = float(df[channel].sum())
        total_contrib = float(contributions[channel].sum())
        roas = total_contrib / total_spend if total_spend > 0 else np.nan
        avg_weekly_contrib = float(contributions[channel].mean())
        rows.append(
            {
                "channel": channel,
                "total_spend": total_spend,
                "modeled_contribution": total_contrib,
                "roas": roas,
                "avg_weekly_contribution": avg_weekly_contrib,
            }
        )

    metrics_df = pd.DataFrame(rows).sort_values("roas", ascending=False).reset_index(drop=True)
    return metrics_df


def _diminishing_returns_point(spend_grid: np.ndarray, response_grid: np.ndarray) -> float:
    if len(spend_grid) < 3:
        return float(spend_grid[-1])

    marginal = np.diff(response_grid) / np.diff(spend_grid)
    if np.all(~np.isfinite(marginal)):
        return float(spend_grid[-1])

    finite_marginal = marginal[np.isfinite(marginal)]
    if len(finite_marginal) == 0:
        return float(spend_grid[-1])

    initial = float(max(finite_marginal[0], 1e-8))
    threshold = 0.2 * initial
    idx = np.where(marginal <= threshold)[0]
    if len(idx) == 0:
        return float(spend_grid[-1])
    return float(spend_grid[idx[0]])


def generate_response_curves(
    project_root: Path,
    model: MMMModel,
    df: pd.DataFrame,
    channels: List[str],
) -> Dict[str, Dict[str, np.ndarray]]:
    output = {}

    for channel in channels:
        p95 = float(np.percentile(df[channel], 95))
        max_spend = max(p95 * 1.8, 1.0)
        spend_grid = np.linspace(0.0, max_spend, 150)
        response_grid = np.array([model.static_channel_response(channel, spend) for spend in spend_grid])

        dim_point = _diminishing_returns_point(spend_grid, response_grid)

        plot_path = project_root / "visuals" / f"response_curves_{channel.lower()}.png"
        plot_response_curve(
            spend_grid=spend_grid,
            response_grid=response_grid,
            channel=channel,
            diminishing_point=dim_point,
            output_path=plot_path,
        )

        output[channel] = {
            "spend_grid": spend_grid,
            "response_grid": response_grid,
            "diminishing_point": np.array([dim_point]),
        }

    return output


def write_roas_report(
    output_path: Path,
    metrics_df: pd.DataFrame,
    response_curves: Dict[str, Dict[str, np.ndarray]],
    model: MMMModel,
) -> None:
    lines = []
    lines.append("# ROAS and Channel Insights")
    lines.append("")
    lines.append("## Channel Coefficients")
    for channel in model.channels:
        beta = model.beta[channel]
        draws = model.posterior_draws.get(f"beta_{channel}")
        if draws is not None and len(draws) > 0:
            low = np.percentile(draws, 5)
            high = np.percentile(draws, 95)
            lines.append(f"- {channel}: beta={beta:.2f} (approx 90% interval: {low:.2f} to {high:.2f})")
        else:
            lines.append(f"- {channel}: beta={beta:.2f}")

    lines.append("")
    lines.append("## ROAS Summary")
    lines.append("| Channel | Total Spend | Modeled Contribution | ROAS | Avg Weekly Contribution |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in metrics_df.iterrows():
        lines.append(
            "| {channel} | {spend:,.0f} | {contrib:,.0f} | {roas:.3f} | {avg:,.0f} |".format(
                channel=row["channel"],
                spend=row["total_spend"],
                contrib=row["modeled_contribution"],
                roas=row["roas"],
                avg=row["avg_weekly_contribution"],
            )
        )

    lines.append("")
    lines.append("## Diminishing Returns")
    lines.append("- Diminishing point is defined where marginal response falls below 20% of initial marginal response.")
    for channel, payload in response_curves.items():
        dim_point = float(payload["diminishing_point"][0])
        lines.append(f"- {channel}: estimated diminishing-returns spend level ~ {dim_point:,.0f} per week")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
