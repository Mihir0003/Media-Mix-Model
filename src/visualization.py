"""Visualization utilities for MMM artifacts."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_actual_vs_predicted(
    dates: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    holdout_start: pd.Timestamp,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))
    plt.plot(dates, y_true, label="Actual Revenue", linewidth=2.0, color="#0f172a")
    plt.plot(dates, y_pred, label="Predicted Revenue", linewidth=2.0, linestyle="--", color="#2563eb")
    plt.axvline(holdout_start, color="#dc2626", linestyle=":", linewidth=2, label="Hold-out Start")
    plt.title("Actual vs Predicted Revenue")
    plt.xlabel("Week")
    plt.ylabel("Revenue")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_response_curve(
    spend_grid: np.ndarray,
    response_grid: np.ndarray,
    channel: str,
    diminishing_point: float,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(spend_grid, response_grid, color="#0f766e", linewidth=2.2)
    plt.axvline(diminishing_point, color="#b45309", linestyle="--", linewidth=2, label="Diminishing Returns")
    plt.title(f"Response Curve: {channel}")
    plt.xlabel(f"{channel} Spend")
    plt.ylabel("Predicted Revenue Contribution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_budget_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x = np.arange(len(comparison_df))
    width = 0.36

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, comparison_df["current_spend"], width=width, label="Current", color="#475569")
    plt.bar(x + width / 2, comparison_df["optimized_spend"], width=width, label="Optimized", color="#0d9488")
    plt.xticks(x, comparison_df["channel"])
    plt.ylabel("Budget")
    plt.title("Current vs Optimized Budget Allocation")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
