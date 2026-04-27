# 📖 How to Run: Media Mix Model Pipeline

This guide provides step-by-step instructions for setting up and running the Media Mix Model (MMM) pipeline.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.9 or higher**
- **pip** (Python package installer)
- (Optional) **Virtualenv** for environment isolation

---

## 🛠️ Step 1: Environment Setup

1. **Open your terminal** and navigate to the project root:
   ```bash
   cd path/to/Media-Mix-Model
   ```

2. **Create a virtual environment** (highly recommended):
   ```bash
   python3 -m venv mmm_env
   ```

3. **Activate the environment**:
   - **MacOS / Linux**:
     ```bash
     source mmm_env/bin/activate
     ```
   - **Windows**:
     ```bash
     mmm_env\Scripts\activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Step 2: Prepare Your Data

The pipeline expects two CSV files in the root directory. If you don't have them, the pipeline will generate sample data for you.

### File 1: `Weekly_Media_Spend.csv`
Required columns:
- `Date`: Format `YYYY-MM-DD` (Weekly frequency)
- `Channel_Name_1`: Spend amount (e.g., `TV`)
- `Channel_Name_2`: Spend amount (e.g., `Digital`)
- ...

### File 2: `Weekly_Revenue.csv`
Required columns:
- `Date`: Format `YYYY-MM-DD`
- `Revenue`: The target metric to optimize.

---

## 🚀 Step 3: Run the Pipeline

Execute the following command to start the end-to-end process:

```bash
python3 -m src.pipeline
```

### What happens during execution?
1. **Data Loading**: Loads and merges spend and revenue data.
2. **Feature Engineering**: Applies Adstock and Hill transforms.
3. **Model Training**: Fits a Bayesian MMM model to your data.
4. **ROI Analysis**: Calculates Return on Ad Spend (ROAS).
5. **Optimization**: Runs a budget allocator to find the best spend mix.
6. **Report Generation**: Saves all charts and documents.

---

## 🔍 Step 4: Reviewing Results

Once the script finishes, explore the output folders:

- **`visuals/`**: Open `actual_vs_predicted.png` to see how well the model fits your historical data.
- **`reports/`**: Read `roas_analysis.md` for channel performance.
- **`docs/`**: The `final_report.md` contains the executive summary and strategic recommendations.

---

## ❓ Troubleshooting

- **ModuleNotFoundError**: Ensure you have activated your virtual environment and run `pip install -r requirements.txt`.
- **Data Inconsistency**: If your dates don't match or are not weekly, the pipeline might fail or produce poor results. Use the generated `reports/data_quality_report.md` to debug.
- **Missing Dependencies**: If `lightweight_mmm` is missing, the model will automatically fall back to a robust custom implementation. You don't need to manually install it unless you want Bayesian specificities.

---

## 💡 Pro Tip
To run a quick test without your own data, simply delete any existing `Weekly_*.csv` files in the root and run the command. The system will create a perfect synthetic dataset for you to explore the features!
