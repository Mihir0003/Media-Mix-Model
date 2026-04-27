# 🚀 Media Mix Modeling (MMM) for Marketing Optimization

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![MBA-Level](https://img.shields.io/badge/Level-MBA--Grade-brightgreen.svg)]()

An end-to-end, high-fidelity **Media Mix Modeling (MMM)** pipeline designed to provide data-driven insights into marketing effectiveness. This project leverages Bayesian-inspired methodologies, adstock transformations, and non-linear optimization to help marketing teams maximize their Return on Ad Spend (ROAS).

---

## 🌟 Key Features

-   **End-to-End Pipeline**: Automates everything from raw data ingestion to final strategic reporting.
-   **Advanced Feature Engineering**: Implements **Geometric Adstock** (carryover effects) and **Hill Saturation** (diminishing returns).
-   **Robust Modeling Engine**: Prefers `lightweight_mmm` (Bayesian) with a fallback to a custom bootstrap-validated regression for maximum reliability.
-   **Channel Attribution**: Calculates granular ROAS and marginal effectiveness for every media channel.
-   **Budget Optimization**: Uses constrained non-linear optimization to suggest the ideal budget allocation for maximum revenue.
-   **Executive Reporting**: Generates automated Markdown reports and rich visualizations ready for boardroom presentations.

---

## 📂 Project Structure

```text
├── data/               # Processed datasets and quality diagnostics
├── docs/               # Final executive reports and implementation plans
├── models/             # Serialized model artifacts (.pkl)
├── reports/            # Granular analysis (ROAS, Performance, Optimization)
├── src/                # Core logic (Processing, Modeling, Optimization)
│   ├── analytics.py
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── modeling.py
│   ├── optimization.py
│   ├── pipeline.py     <-- Main Entry Point
│   └── visualization.py
├── visuals/            # Generated charts and diagnostic plots
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone <repository-url>
cd Media-Mix-Model
```

### 2. Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage

### Data Preparation
Place your weekly marketing data in the root directory:
- `Weekly_Media_Spend.csv`: Columns should include `Date` and spend per channel (e.g., `TV`, `Digital`, `Social`).
- `Weekly_Revenue.csv`: Columns should include `Date` and `Revenue`.

*Note: If these files are missing, the pipeline will automatically generate synthetic data for demonstration purposes.*

### Running the Pipeline
Execute the full suite of analysis with a single command:
```bash
python3 -m src.pipeline
```

---

## 📊 Methodology

### 1. Data Engineering
The pipeline performs automated schema validation, missing value imputation, and collinearity checks to ensure model stability.

### 2. Adstock & Saturation
-   **Adstock**: Captures the lagged effect of advertising over time.
-   **Hill Function**: Models the "S-curve" of diminishing returns, identifying where additional spend stops being efficient.

### 3. Optimization
The optimizer identifies the global maximum for revenue given a fixed budget constraint, reallocating funds from low-performing to high-performing channels based on marginal returns.

---

## 📈 Sample Artifacts

Upon completion, the pipeline populates:
-   `visuals/actual_vs_predicted.png`: Model accuracy validation.
-   `visuals/response_curves_*.png`: Visualizing diminishing returns per channel.
-   `reports/roas_analysis.md`: Detailed ROI breakdown.
-   `docs/final_report.md`: Executive summary of findings and recommendations.

---

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
