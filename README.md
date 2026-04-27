# Media Mix Modeling for Marketing Budget Optimization

MBA-level Media Mix Modeling (MMM) pipeline in Python using Bayesian methods.

## Project Structure
```
/docs
  implementation_plan.md
  final_report.md
/data
  processed_data.csv (generated)
/models
  mmm_model.pkl (generated)
/reports
  data_quality_report.md
  model_performance.md
  roas_analysis.md
  optimization_results.md
/src
  data_processing.py
  modeling.py
  optimization.py
  visualization.py
  pipeline.py
```

## Quick Start
1. Place input files in the repository root:
   - Weekly_Media_Spend.csv
   - Weekly_Revenue.csv
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the pipeline:
   ```bash
   python -m src.pipeline
   ```

## Outputs
- /data/processed_data.csv
- /models/mmm_model.pkl
- /visuals/correlation_matrix.png
- /visuals/actual_vs_predicted.png
- /visuals/response_curves_<channel>.png
- /visuals/budget_comparison.png
- /reports/*.md
- /docs/final_report.md

## Notes
- Uses `lightweight_mmm` for Bayesian MMM. If unavailable, the pipeline will stop with a clear error.