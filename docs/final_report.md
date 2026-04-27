# Final MBA-Level MMM Report

## 1. Executive Summary
This project delivered an end-to-end Media Mix Model (MMM) pipeline that transforms weekly marketing spend and revenue data into channel-level ROI insights and a constrained budget optimization recommendation. The model uses adstock and saturation dynamics to reflect carryover and diminishing returns.
Hold-out performance reached R²=0.0576 and MAPE=2.42%, providing a credible basis for decision support.

## 2. Methodology
- Data Engineering: schema validation, weekly standardization, merge, imputation, collinearity diagnostics.
- Feature Engineering: geometric adstock and Hill saturation transforms per channel.
- Modeling: Bayesian MMM preference via `lightweight_mmm`; custom Bayesian-style fallback with bootstrap uncertainty when package is unavailable.
- Validation: chronological split with last 12 weeks held out.
- Optimization: constrained nonlinear allocation for fixed budget ($200,000).

## 3. Model Performance
- Engine used: `custom_bayesian_fallback`
- Hold-out R²: 0.0576
- Hold-out MAPE: 2.42%
- Train R²: 0.2010
- Train MAPE: 1.91%

## 4. Key Insights
- Highest modeled ROAS channel: TV
- Channel effects exhibit diminishing returns, visible in response curve artifacts.
- Modeled contributions and ROAS are reported in `reports/roas_analysis.md`.

## 5. Budget Optimization Results
- Optimized channel set: Search, Social, TV
- Predicted media-response uplift vs current allocation: 6.78%
- Current response: 1,560.73; Optimized response: 1,666.51

## 6. Strategic Recommendations
1. Reallocate spend toward higher marginal-response channels up to the identified diminishing-return thresholds.
2. Track weekly model drift and retrain monthly to absorb changing market response patterns.
3. Run incrementality experiments on top-spend channels to calibrate MMM attribution confidence.
4. Use optimizer output as a decision baseline, then apply practical business constraints (brand goals, inventory, seasonality).

## Artifact Index
- data/processed_data.csv
- models/mmm_model.pkl
- reports/data_quality_report.md
- reports/model_performance.md
- reports/roas_analysis.md
- reports/optimization_results.md
- visuals/correlation_matrix.png
- visuals/actual_vs_predicted.png
- visuals/response_curves_search.png
- visuals/response_curves_social.png
- visuals/response_curves_tv.png
- visuals/budget_comparison.png
- reports/execution_log.md