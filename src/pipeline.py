"""End-to-end pipeline for MBA-level Media Mix Modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from .analytics import calculate_channel_metrics, generate_response_curves, write_roas_report
from .data_processing import process_data
from .modeling import fit_mmm, save_model, write_model_performance_report
from .optimization import optimize_budget_allocation, save_budget_plot, write_optimization_report
from .visualization import plot_actual_vs_predicted


@dataclass
class ExecutionLogger:
    lines: List[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.lines.append(f"- [{timestamp}] {message}")

    def save(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        content = ["# Execution Log", "", *self.lines]
        output_path.write_text("\n".join(content), encoding="utf-8")


def _ensure_structure(project_root: Path) -> None:
    for folder in ["docs", "data", "models", "reports", "src", "visuals"]:
        (project_root / folder).mkdir(parents=True, exist_ok=True)


def _write_final_report(
    output_path: Path,
    *,
    synthetic_data_generated: bool,
    model_metrics: Dict[str, float],
    model_diagnostics: Dict[str, object],
    roas_metrics,
    optimization_result: Dict[str, object],
    channels: List[str],
) -> None:
    best_channel = roas_metrics.iloc[0]["channel"] if len(roas_metrics) else "N/A"

    lines = []
    lines.append("# Final MBA-Level MMM Report")
    lines.append("")

    lines.append("## 1. Executive Summary")
    lines.append(
        "This project delivered an end-to-end Media Mix Model (MMM) pipeline that transforms weekly marketing spend and "
        "revenue data into channel-level ROI insights and a constrained budget optimization recommendation. "
        "The model uses adstock and saturation dynamics to reflect carryover and diminishing returns."
    )
    lines.append(
        f"Hold-out performance reached R²={model_metrics['r2_holdout']:.4f} and MAPE={model_metrics['mape_holdout']:.2f}%, "
        "providing a credible basis for decision support."
    )
    if synthetic_data_generated:
        lines.append(
            "Note: input files were not present at run time, so deterministic synthetic data was generated for reproducible "
            "pipeline demonstration. Replace with actual business data for production decisions."
        )
    lines.append("")

    lines.append("## 2. Methodology")
    lines.append("- Data Engineering: schema validation, weekly standardization, merge, imputation, collinearity diagnostics.")
    lines.append("- Feature Engineering: geometric adstock and Hill saturation transforms per channel.")
    lines.append(
        "- Modeling: Bayesian MMM preference via `lightweight_mmm`; custom Bayesian-style fallback with bootstrap "
        "uncertainty when package is unavailable."
    )
    lines.append("- Validation: chronological split with last 12 weeks held out.")
    lines.append("- Optimization: constrained nonlinear allocation for fixed budget ($200,000).")
    lines.append("")

    lines.append("## 3. Model Performance")
    lines.append(f"- Engine used: `{model_diagnostics['engine']}`")
    lines.append(f"- Hold-out R²: {model_metrics['r2_holdout']:.4f}")
    lines.append(f"- Hold-out MAPE: {model_metrics['mape_holdout']:.2f}%")
    lines.append(f"- Train R²: {model_metrics['r2_train']:.4f}")
    lines.append(f"- Train MAPE: {model_metrics['mape_train']:.2f}%")
    lines.append("")

    lines.append("## 4. Key Insights")
    lines.append(f"- Highest modeled ROAS channel: {best_channel}")
    lines.append("- Channel effects exhibit diminishing returns, visible in response curve artifacts.")
    lines.append("- Modeled contributions and ROAS are reported in `reports/roas_analysis.md`.")
    lines.append("")

    lines.append("## 5. Budget Optimization Results")
    lines.append(f"- Optimized channel set: {', '.join(optimization_result['channels'])}")
    lines.append(
        f"- Predicted media-response uplift vs current allocation: {optimization_result['uplift_pct']:.2f}%"
    )
    lines.append(
        f"- Current response: {optimization_result['current_predicted_revenue']:,.2f}; "
        f"Optimized response: {optimization_result['optimized_predicted_revenue']:,.2f}"
    )
    lines.append("")

    lines.append("## 6. Strategic Recommendations")
    lines.append("1. Reallocate spend toward higher marginal-response channels up to the identified diminishing-return thresholds.")
    lines.append("2. Track weekly model drift and retrain monthly to absorb changing market response patterns.")
    lines.append("3. Run incrementality experiments on top-spend channels to calibrate MMM attribution confidence.")
    lines.append("4. Use optimizer output as a decision baseline, then apply practical business constraints (brand goals, inventory, seasonality).")
    lines.append("")

    lines.append("## Artifact Index")
    lines.append("- data/processed_data.csv")
    lines.append("- models/mmm_model.pkl")
    lines.append("- reports/data_quality_report.md")
    lines.append("- reports/model_performance.md")
    lines.append("- reports/roas_analysis.md")
    lines.append("- reports/optimization_results.md")
    lines.append("- visuals/correlation_matrix.png")
    lines.append("- visuals/actual_vs_predicted.png")
    for channel in channels:
        lines.append(f"- visuals/response_curves_{channel.lower()}.png")
    lines.append("- visuals/budget_comparison.png")
    lines.append("- reports/execution_log.md")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _validate_artifacts(project_root: Path, channels: List[str]) -> None:
    required = [
        project_root / "data" / "processed_data.csv",
        project_root / "models" / "mmm_model.pkl",
        project_root / "reports" / "data_quality_report.md",
        project_root / "reports" / "model_performance.md",
        project_root / "reports" / "roas_analysis.md",
        project_root / "reports" / "optimization_results.md",
        project_root / "docs" / "final_report.md",
        project_root / "visuals" / "correlation_matrix.png",
        project_root / "visuals" / "actual_vs_predicted.png",
        project_root / "visuals" / "budget_comparison.png",
        project_root / "reports" / "execution_log.md",
    ]
    required.extend(project_root / "visuals" / f"response_curves_{channel.lower()}.png" for channel in channels)

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required artifacts: {missing}")


def run_pipeline(project_root: Path) -> None:
    _ensure_structure(project_root)
    logger = ExecutionLogger()

    try:
        logger.log("Pipeline started")

        np.random.seed(42)
        logger.log("Random seed set to 42")

        logger.log("Step 1: Data processing initiated")
        data_result = process_data(project_root=project_root)
        logger.log(
            f"Data processing completed with {len(data_result.df)} rows and channels: {', '.join(data_result.spend_columns)}"
        )
        if data_result.synthetic_data_generated:
            logger.log("Input CSV files were missing; deterministic synthetic inputs were generated")

        logger.log("Step 2: Bayesian MMM training initiated")
        model_result = fit_mmm(
            df=data_result.df,
            channels=data_result.spend_columns,
            revenue_col=data_result.revenue_column,
            holdout_weeks=12,
            random_seed=42,
        )
        model = model_result["model"]
        logger.log(
            "Model training completed: "
            f"engine={model_result['diagnostics']['engine']}, "
            f"holdout_r2={model_result['metrics']['r2_holdout']:.4f}, "
            f"holdout_mape={model_result['metrics']['mape_holdout']:.2f}%"
        )

        model_path = project_root / "models" / "mmm_model.pkl"
        save_model(model, str(model_path))
        logger.log(f"Model saved to {model_path}")

        holdout_start = data_result.df.iloc[-12]["Date"]
        plot_actual_vs_predicted(
            dates=data_result.df["Date"],
            y_true=model_result["y_true_full"],
            y_pred=model_result["y_pred_full"],
            holdout_start=holdout_start,
            output_path=project_root / "visuals" / "actual_vs_predicted.png",
        )
        logger.log("Actual-vs-predicted visualization generated")

        write_model_performance_report(
            output_path=str(project_root / "reports" / "model_performance.md"),
            metrics=model_result["metrics"],
            diagnostics=model_result["diagnostics"],
        )
        logger.log("Model performance report saved")

        logger.log("Step 3: ROI and channel analytics initiated")
        roas_metrics = calculate_channel_metrics(
            model=model,
            df=data_result.df,
            channels=data_result.spend_columns,
        )
        response_curves = generate_response_curves(
            project_root=project_root,
            model=model,
            df=data_result.df,
            channels=data_result.spend_columns,
        )
        write_roas_report(
            output_path=project_root / "reports" / "roas_analysis.md",
            metrics_df=roas_metrics,
            response_curves=response_curves,
            model=model,
        )
        logger.log("ROAS analysis and response curves generated")

        logger.log("Step 4: Budget optimization initiated")
        optimization_result = optimize_budget_allocation(
            model=model,
            df=data_result.df,
            total_budget=200000.0,
        )
        write_optimization_report(
            output_path=project_root / "reports" / "optimization_results.md",
            optimization_result=optimization_result,
            budget=200000.0,
        )
        save_budget_plot(project_root=project_root, optimization_result=optimization_result)
        logger.log(
            "Budget optimization complete with estimated uplift "
            f"{optimization_result['uplift_pct']:.2f}%"
        )

        logger.log("Step 5: Final report generation initiated")
        _write_final_report(
            output_path=project_root / "docs" / "final_report.md",
            synthetic_data_generated=data_result.synthetic_data_generated,
            model_metrics=model_result["metrics"],
            model_diagnostics=model_result["diagnostics"],
            roas_metrics=roas_metrics,
            optimization_result=optimization_result,
            channels=data_result.spend_columns,
        )
        logger.log("Final report saved")

        logger.save(project_root / "reports" / "execution_log.md")
        _validate_artifacts(project_root=project_root, channels=data_result.spend_columns)

        logger.log("Artifact validation completed successfully")
        logger.save(project_root / "reports" / "execution_log.md")

    except Exception as exc:  # pragma: no cover
        logger.log(f"Pipeline failed: {exc}")
        logger.save(project_root / "reports" / "execution_log.md")
        raise


if __name__ == "__main__":
    run_pipeline(project_root=Path(__file__).resolve().parents[1])
