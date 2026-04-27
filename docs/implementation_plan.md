# Implementation Plan: MBA-Level Media Mix Model (MMM)

## 1) Objective and Scope
Build a reproducible, end-to-end Bayesian Media Mix Modeling pipeline that:
- Ingests weekly media spend and weekly revenue data
- Engineers MMM-ready channel features
- Trains a Bayesian MMM with adstock and saturation effects
- Evaluates predictive performance on a 12-week hold-out window
- Produces ROAS/channel contribution insights
- Optimizes allocation of a fixed $200,000 budget across Search, Social, TV
- Delivers business and technical reports with visual evidence

## 2) Phase-by-Phase Plan

### Phase A: Data Engineering
Inputs:
- `Weekly_Media_Spend.csv`
- `Weekly_Revenue.csv`

Processing:
- Validate required columns and parse dates
- Normalize to weekly granularity (`W-MON` week start)
- Aggregate duplicate weekly entries if needed
- Merge media + revenue on week key
- Handle missing values using explicit rules
- Build data quality checks and log outcomes

Outputs:
- `/data/processed_data.csv`
- `/reports/data_quality_report.md`
- `/visuals/correlation_matrix.png`
- `/reports/execution_log.md` (run-level logs)

Validation:
- Date uniqueness and monotonic ordering
- No null leakage in model-ready columns
- Schema and type consistency
- Correlation matrix generated and interpretable

---

### Phase B: Feature Engineering
Inputs:
- `/data/processed_data.csv`

Processing:
- Identify channel columns (excluding date/revenue)
- Construct reusable transformations:
  - Geometric adstock
  - Hill saturation
- Scale spend features where required for numerical stability
- Persist feature metadata used by model and analytics

Outputs:
- In-memory transformed arrays for model training
- Feature metadata in model object for reproducibility

Validation:
- Transformation unit checks (shape, value ranges)
- No negative transformed response where not expected
- Stable parameter defaults across reruns

---

### Phase C: Bayesian Modeling
Inputs:
- Processed dataset + transformed features

Processing:
- Hold-out design: last 12 weeks as validation
- Preferred engine: `lightweight_mmm`
- Fallback engine if unavailable: Bayesian-style nonlinear regression with constrained optimization and bootstrap uncertainty approximation
- Fit channel effects and baseline
- Generate in-sample + hold-out predictions

Outputs:
- `/models/mmm_model.pkl`
- `/visuals/actual_vs_predicted.png`
- `/reports/model_performance.md`

Validation:
- Metrics: R², MAPE
- Visual fit (actual vs predicted)
- No train/test leakage (strict chronological split)

---

### Phase D: ROI & Insights
Inputs:
- Trained model object
- Processed data

Processing:
- Extract channel-level coefficients/effects
- Estimate channel revenue contribution by period
- Compute ROAS per channel
- Generate response curves (spend -> predicted revenue)
- Estimate diminishing-returns point from marginal gains

Outputs:
- `/reports/roas_analysis.md`
- `/visuals/response_curves_<channel>.png`

Validation:
- Contributions sum approximately to modeled media component
- ROAS formula consistency and unit checks
- Response curve monotonicity where expected

---

### Phase E: Budget Optimization
Inputs:
- Trained response functions
- Total budget = `$200,000`

Processing:
- Objective: maximize predicted revenue from channel response functions
- Constraint: spend_Search + spend_Social + spend_TV = 200000
- Non-negativity bounds and practical caps
- Compare optimized allocation vs recent average current allocation

Outputs:
- `/reports/optimization_results.md`
- `/visuals/budget_comparison.png`

Validation:
- Constraint satisfaction within tolerance
- Optimization convergence diagnostics
- Revenue uplift computed from consistent model equations

---

### Phase F: Reporting
Inputs:
- All reports + model metrics + visual artifacts

Processing:
- Compose MBA-level narrative:
  1. Executive Summary
  2. Methodology
  3. Model Performance
  4. Key Insights
  5. Budget Optimization Results
  6. Strategic Recommendations

Outputs:
- `/docs/final_report.md`

Validation:
- All sections complete
- References point to generated artifacts
- Business recommendations linked to quantitative evidence

## 3) Code Architecture
Implemented modules:
```
/src
  data_processing.py
  feature_engineering.py
  modeling.py
  analytics.py
  optimization.py
  visualization.py
  pipeline.py
```

Execution entrypoint:
- `python -m src.pipeline`

## 4) Libraries and Tools
Core:
- `pandas`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`

Bayesian MMM:
- Preferred: `lightweight_mmm`
- Fallback path: deterministic nonlinear fit with uncertainty bands if package unavailable

Reproducibility:
- Fixed random seed(s)
- Serialized model and run log
- Deterministic data simulation fallback when required files are absent

## 5) Validation Strategy (Global)
- Data validation checks at ingestion and post-merge
- Leakage prevention via chronological split
- Metric validation (R², MAPE) and sanity checks on predictions
- Artifact existence check at pipeline end
- Run log records timestamps, warnings, decisions, and fallback paths

## 6) Risks and Assumptions
Risks:
- Missing input files prevent factual business inference
- Small sample size may produce unstable channel effects
- High channel collinearity can blur attribution
- `lightweight_mmm` installation may fail in local environment

Assumptions:
- Weekly data is correctly timestamped and comparable across files
- Revenue is primarily influenced by included channels + baseline trend/noise
- Search, Social, TV are available for optimization; if not, optimizer uses first three channels and logs substitution

Mitigations:
- Strict schema checks and transparent warnings
- Correlation diagnostics and documented attribution caveats
- Fallback modeling path with documented limitations
- Deterministic synthetic demo data only when inputs are missing, explicitly flagged in reports

## 7) Completion Criteria
A run is considered complete only when all required artifacts exist:
- `/data/processed_data.csv`
- `/models/mmm_model.pkl`
- `/reports/data_quality_report.md`
- `/reports/model_performance.md`
- `/reports/roas_analysis.md`
- `/reports/optimization_results.md`
- `/docs/final_report.md`
- `/visuals/correlation_matrix.png`
- `/visuals/actual_vs_predicted.png`
- `/visuals/response_curves_<channel>.png` (for each optimized channel)
- `/visuals/budget_comparison.png`
- `/reports/execution_log.md`

