# LNG Portfolio: Sensor to Decision (Senior Data Scientist — Venture Global LNG)

An end-to-end data engineering and machine learning portfolio tailored for the **Senior Data Scientist** role at **Venture Global LNG** ([`jd.md`](jd.md)). Python, DuckDB, PySpark, and Scikit-Learn run locally or in GitHub Actions CI/CD with no cloud account required. PySpark stands in for Databricks and DuckDB stands in for SQL warehouse reconciliation across Bronze, Silver, and Gold medallion tables.

## Role & Architecture Alignment

| Venture Global Stack (`jd.md` & `data_usage.md`) | Portfolio Stand-In | Purpose in Pipeline |
|---|---|---|
| **OSIsoft PI Historian** (flows, pressures, temperatures, valves) | Tennessee Eastman `d00`–`d21` (`52` tags, `3`-min steps) | Train-level sensor telemetry; `d00` is normal operation, `d01`–`d21` are process faults (switched on at step 160 in test runs) |
| **Azure Data Lake Gen2 & Data Factory** | `download.py` → `data/raw` (Bronze) → `data/silver` → `data/gold` | Automated monthly ingestion of EIA LNG exports, Gulf terminal volumes, Henry Hub gas prices, and NOAA Lake Charles weather/storms |
| **Databricks (PySpark & SQL)** | `prepare.py` (`PySpark` + `DuckDB`) | Distributed $3\sigma$ operating-envelope checks in PySpark and multi-source SQL joins in DuckDB |
| **Predictive Modeling & Classification** | `model.py` (`Scikit-Learn` RF, Deep MLP, Probability Calibration) | Compares heuristic rules, Random Forests, Calibrated Random Forests, and Deep Neural Networks (MLP) |
| **Executive BI & Stakeholder Communication** | `dashboard.py` → `site/index.html` | Self-contained interactive Plotly dashboard written in clear language for both non-technical stakeholders and engineers |

## Live Dashboard

**Live:** https://mhadeli.github.io/lng-sensor-to-decision/

`dashboard.py` compiles the Gold tables and ML evaluations into a single static page, `site/index.html`. The GitHub Action in `.github/workflows/dashboard.yml` runs `download.py`, `prepare.py`, `model.py`, and `dashboard.py` on every push to `main`, on the 15th of each month (after EIA's monthly release), and on demand.

## Setup & Run

```bash
source .venv/bin/activate
pip install -r requirements.txt

python download.py    # Ingest EIA, NOAA, TEP historian, and Microsoft PdM raw files
python prepare.py     # Build Silver & Gold tables with PySpark and DuckDB SQL
python model.py       # Train & evaluate Heuristic, Random Forest, Calibrated RF, and Deep MLP models
python dashboard.py   # Generate site/index.html
```

### Gold Layer Outputs (`data/gold/`)

- `monthly.csv` — Reconciled monthly exports, Henry Hub price, Lake Charles temperature, Louisiana storms, and Gulf terminal volumes
- `alerts.csv` — PySpark $3\sigma$ historian operating-envelope flags (fitted strictly on normal training run `d00.dat`)
- `fault_predictions.csv` — Out-of-sample test predictions and probabilities across 21,120 rows for the 3-Sigma Rule, Random Forest, Calibrated Random Forest, and Deep Neural Network (MLP)
- `feature_importance.csv` — Random Forest feature importances aggregated by physical sensor tag (level vs. 3-minute step change)
- `calibration_curve.csv` — Reliability diagram bins comparing predicted fault probabilities against observed fault frequencies
- `volume_forecast.csv` — 24-month walk-forward U.S. LNG export forecasts comparing Naive Baseline, Physics Day-Count Scaling, Seasonal Factor, Random Forest Regressor, Domain Heuristic, and Deep Neural Network (MLP Regressor)
- `volume_gap.csv` & `maintenance.csv` — National-vs-Gulf volume reconciliation check and 2015 equipment work-order/failure counts

## Results Summary

### 1. Sensor Fault Detection & Probability Calibration (21,120 unseen test rows)

| Detection Method | Faults Caught (Sensitivity) | Normal Cleared (Specificity) | False Alarm Rate | ROC-AUC | Brier Score (↓) |
|---|---|---|---|---|---|
| **1. 3-Sigma Historian Rule (Baseline)** | 79.2% | 81.8% | 18.2% | — | — |
| **2. Random Forest (Uncalibrated)** | 82.7% | 72.2% | 27.8% | 0.864 | 0.125 |
| **3. Calibrated Random Forest (Prior + 15m Persistence)** | 78.5% | **81.9%** | **18.1%** | 0.877 | 0.132 |
| **4. Deep Neural Network (MLP + 15m Window)** | **84.3%** | **81.0%** | 19.0% | **0.914** | **0.107** |

- **Root-Cause Feature Importance:** The Random Forest identifies the exact physical loops driving fault alerts—led by the **Reactor cooling-water valve (`x51`, 10.6%)**, **Separator pressure (`x13`, 8.6%)**, **Stripper steam flow (`x19`, 8.3%)**, and **Stripper temperature (`x18`, 8.3%)**.
- **Fixing Alarm Fatigue via Calibration:** Because the training archive contains 20 fault runs for every 1 normal run, uncalibrated Random Forest probabilities trigger false alarms on 27.8% of normal test rows. Applying Bayesian prior calibration and a 5-step (15-minute) persistence window cuts false alarms to 18.1%.
- **Deep Learning Gain:** A 2-layer Multi-Layer Perceptron (`64 → 32`) trained on normal-standardized z-scores with a 15-minute smoothing window achieves the best overall discrimination (**0.914 ROC-AUC**, **84.3%** faults caught, **81.0%** normal cleared). Faults 3, 9, and 15 barely move the physical sensors and remain unobservable without additional instrumentation.

### 2. Next-Month U.S. LNG Export Forecast (24-month walk-forward evaluation)

| Method | Mean Absolute Error (Bcf) | MAPE | vs. Baseline |
|---|---|---|---|
| **Naive Baseline (Same as last month)** | 25.31 Bcf | 5.2% | Baseline |
| **1. Physics Day-Count Scaling** (`last_daily_rate × days_in_month`) | 21.37 Bcf | 4.4% | -16% |
| **2. Full Calendar-Month Factor** | 22.84 Bcf | 4.6% | -10% |
| **3. Random Forest Regressor (L1 Loss)** | 20.50 Bcf | 4.1% | -19% |
| **4. Domain Heuristic (50/50 Blend of 1 & 2)** | 19.69 Bcf | 3.9% | -22% |
| **5. Deep Neural Network (MLP Regressor)** | **19.35 Bcf** | **3.8%** | **-24%** |

## What a Production Engineer Would Do

1. **Before raising a train's daily target:** Check the calibrated fault probability and inspect the top contributing tags (`x51` cooling-water valve, `x13` separator pressure, `x19` stripper steam flow) rather than reacting to isolated single-step spikes.
2. **When setting the monthly export plan:** Account for calendar day counts and summer ambient temperature dips first (worth a 22% error reduction), then layer in the MLP forecast alongside internal plant schedules (Plaquemines train ramp-ups, planned maintenance windows, and cargo liftings).

