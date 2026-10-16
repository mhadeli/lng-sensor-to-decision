# LNG: from data to decisions

A data science portfolio built around two operational questions: **which process readings deserve investigation, and how accurately can next month’s LNG exports be forecast?** The dashboard explains the results for an operations audience, with technical details available on demand.

This project supports my application for a senior data scientist role. [jd.md](jd.md) describes the role; [data_usage.md](data_usage.md) records the original research and proposed plant architecture. The implementation uses public stand-ins, not company operating data.

## View and run

Generate and open `site/index.html`. The page bundles Plotly locally and works without a CDN. The existing GitHub Pages destination is https://mhadeli.github.io/lng-sensor-to-decision/; local changes appear there only after deployment.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python download.py       # Tennessee Eastman simulation + monthly EIA LNG exports
python prepare.py        # Validate inputs, build model-ready tables with pandas / SQL
python model.py          # Validate settings, refit, evaluate baseline / MLP / XGBoost
python -m unittest discover -s tests -v
python dashboard.py      # Generate site/index.html + plotly.min.js
```

On macOS, XGBoost needs the OpenMP runtime (`brew install libomp` if Homebrew is available). This workspace’s XGBoost library has been linked to the OpenMP runtime already bundled with scikit-learn; reinstalling XGBoost may require restoring that runtime linkage or installing libomp.

Downloads use cached raw files by default. `python download.py --refresh` refreshes EIA exports and regenerates its CSV. GitHub Actions runs this core workflow on pushes to `main`, monthly, or on demand; a failing build prevents deployment.

## A smaller workflow

```text
Public source files → validated tables → causal features → three approaches
                                            ↓
                              validation chooses settings
                                            ↓
                       separate test evaluation → dashboard
```

| Question | Baseline | MLP | XGBoost |
|---|---|---|---|
| Detect abnormal sensor readings | Any sensor beyond its normal mean ± 3 standard deviations | Regularized 64 → 32 network | Shallow boosted trees |
| Forecast monthly U.S. LNG exports | Last month’s daily export rate × next month’s days | Small regularized 8-unit network | Shallow boosted trees |

Removed the Random Forest variants, hand-set “calibration,” and competing forecast heuristics from the core comparison. Only the two necessary datasets run by default. The original supporting EDA sources remain available with `python download.py --extras` and `python prepare.py --extras`.

Optional Spark demonstration: install `requirements-spark.txt` and Java, then run `python prepare.py --spark`. The default workflow needs no Java. The original `eda.ipynb` is an exploratory archive: its old model outputs are not the current evaluation; use `model.py` and the regenerated dashboard for results.

## What I did, step by step

1. **Ingest:** `download.py` acquires the Tennessee Eastman ZIP and EIA monthly export workbook, keeps raw source files, extracts sensor runs, and writes a consistent monthly CSV. Cached data supports repeatable local builds; `--refresh` updates the export source.
2. **Preprocess:** `prepare.py` checks sensor completeness and duplicate run steps, assigns fault labels, validates positive monthly volumes, and orders monthly data with DuckDB SQL. Model preparation rejects missing months. Sensor runs and national export data remain separate datasets.
3. **Engineer features:** `model.py` builds run-local sensor changes and trailing means, fits a normal-operation reference inside each fit, and adds deviation magnitudes. Export features use month length, lagged growth, and past-only seasonal history.
4. **Model and validate:** compare a baseline, MLP, and XGBoost with a small validation search. Refit with the chosen settings, then evaluate separate sensor runs and 24 months of walk-forward forecasts.
5. **Evaluate operational relevance:** `evaluation.py` measures confirmed event detection, delay, and false-alert episodes from saved predictions using one fixed policy. It does not retune models or thresholds. Run `python evaluation.py` to refresh this analysis without retraining; `model.py` also generates it automatically.
6. **Communicate and maintain:** `dashboard.py` presents the recommendation, trade-offs, workflow, and proposed pilot. Automated tests verify causality and evaluation calculations before the scheduled CI deployment.

## Evaluation design

**Sensor detection.** Tennessee Eastman provides 52 process signals across normal operation and 21 fault scenarios. Training faults are active from the beginning; test faults activate at zero-based step 160. Features use current levels, one-step changes, and trailing five-reading means, resetting at run boundaries. Learned models scale features against normal training observations and include absolute deviation magnitudes. Class weights address the fault-heavy training mix. Scores use a causal five-reading average; this may delay detection and is not probability calibration.

Within each training run, the first 75% supplies fitting data and the last 25% supplies validation, separated by a five-reading purge. Three regularization/depth settings and thresholds from 0.20 to 0.80 are compared by balanced accuracy. A normal reference is fitted separately inside each model fit. After selection, each model refits on the complete training archive. The selected approach and its threshold are fixed before scoring the separate test runs in each build.

**Export forecasting.** The baseline adjusts for month length. Learned models predict the next change in daily rate from calendar seasonality, lagged growth, and historical seasonal changes. Each row’s seasonal feature uses only earlier years—training targets cannot contribute to their own feature. Models train on observations from 2019 onward. A 12-month walk-forward validation period precedes the final 24-month backtest; three settings per model are compared by mean absolute error. Settings and the headline approach are frozen before the test period. Models refit each month using only preceding observations.

**Limits.** The benchmark test data has been examined during project development; it is not an untouched prospective trial. Sensor validation has only 125 normal readings from one normal run, and neighboring readings are correlated. Training and testing cover the same fault scenarios, so performance on novel faults is unknown. Scores and thresholds need recalibration on independent operating history. The forecast uses revised EIA data and assumes the prior month’s volume is available; release delays and historical data vintages are not reconstructed. Neither task demonstrates reduced downtime, safety risk, or operating cost at a real facility.

## Current local results

Snapshot from the current data rebuild (export holdout August 2024–July 2026). The dashboard always calculates its numbers from the newly generated artifacts.

| Detection approach | Abnormal readings detected | Normal readings falsely flagged | Balanced accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline | 79.3% | 18.2% | 80.5% | — |
| MLP — selected on validation | 61.2% | 0.2% | 80.5% | 0.904 |
| XGBoost | 71.8% | 8.8% | 81.5% | 0.903 |

**Recommendation: retain existing approved operational rules during a pilot.** The validation-selected MLP is a research candidate, not an approved replacement. It greatly reduces false flags, **but misses more abnormal readings**. It does not beat the baseline on every criterion. XGBoost has the highest test balanced accuracy, but we do not switch the selected approach after seeing the test. This is a decision-support demonstration; operators would need to agree on an acceptable missed-event / false-alarm trade-off before deployment.

### Event-level results and alert burden

The same fixed policy requires three consecutive flagged readings to confirm a new alert. A fault event is a contiguous fault-labelled interval within a run. Detection requires an alert that starts during the event and confirms before it ends; an alert already active before onset is not credited. Delay is measured from event onset to the third flagged reading, at three minutes per step. A clear reading resets an episode.

| Approach | Events flagged within 30 minutes | Median delay, detected events only | False alerts per 24 normal hours | Never flagged |
|---|---:|---:|---:|---:|
| Baseline | 10 / 21 | 36 min | 7.78 | 0 |
| MLP | 7 / 21 | 105 min | 0.22 | 0 |
| XGBoost | 9 / 21 | 60 min | 3.22 | 0 |

All methods eventually flag all events over the long simulated runs, but eventual detection is not timely intervention. The MLP's low alert burden comes with slower detection. The 30-minute window is an illustrative analysis target, not an approved operating requirement. These descriptive results do not change model selection or establish incident prevention.

False-alert episodes have their first three readings entirely in normal operation. Rates aggregate 216 normal simulation hours across runs, then scale to 24 hours; they are not observed calendar-day workloads. Undetected events are excluded from the median delay and reported separately. There is no acknowledgement or operator-response history.

| Forecast approach | Average absolute error | Average percentage error |
|---|---:|---:|
| Day-adjusted baseline | 21.37 Bcf | 4.58% |
| MLP | 19.34 Bcf | 4.11% |
| XGBoost — selected on validation | 19.53 Bcf | 4.28% |

XGBoost reduces forecast error by **8.6%** against the day-adjusted baseline. MLP has the lowest test error, but XGBoost won the earlier validation period. A smaller model set and honest selection are more useful than presenting whichever test result happens to look best.

## Outputs and checks

`data/gold/` contains:

- `fault_predictions.csv`: test labels, binary predictions, and uncalibrated model scores.
- `fault_events.csv`: one row per model and fault event, with confirmation delay and timely-detection flags.
- `fault_run_alerts.csv`: normal exposure and false-alert episode counts by run and model.
- `fault_operations.json`: fixed alert-policy definitions and aggregate event / burden metrics.
- `feature_importance.csv`: XGBoost importance aggregated by sensor; not a causal explanation.
- `volume_forecast.csv`: 24 months of actual volumes and the three forecasts, in million cubic feet.
- `evaluation.json`: validation scores, selected settings / thresholds, test periods, and final metrics.
- `alerts.csv`: baseline sensor flags from preparation.

Old CSVs from earlier runs can remain in an existing data directory; the new dashboard reads only the current artifacts listed above. Raw data, generated outputs, and the local site are ignored by git; the source pipeline regenerates them.

Tests check causal feature construction, run-boundary resets, missing-month rejection, validation-based model selection, and agreement between saved predictions and displayed metrics. The CI build runs these checks before generating and publishing the dashboard.

## How this translates to plant work

Start with operations: define the decision owner, actionable fault labels, forecast horizon, and acceptable alert burden. Connect historian readings to train status and maintenance events; connect forecasts to feed-gas constraints, availability, outages, and cargo schedules. Validate on later periods and separate trains, measure detection delay and incident-level outcomes, and run alongside existing processes before rollout.

The local architecture demonstrates Python, SQL, feature engineering, model evaluation, reproducibility, and stakeholder communication. A proposed Azure implementation would use Data Factory ingestion, Data Lake storage, and Databricks preparation / scoring, with governed outputs and drift monitoring. Those integrations are proposed, not implemented here.

## Proposed pilot and acceptance gates

These are proposed next steps, not implemented plant integrations. Start with one train; agree on duration and staffing after checking data access, event frequency, and engineering review availability.

| Stage | Accountable partners | Gate before continuing |
|---|---|---|
| Define the decision | Operations lead, reliability engineer, production planner | Approve actionable event definitions, maximum acceptable delay, missed-event tolerance, false-alert burden, forecast target and horizon before model selection. |
| Establish data readiness | Data engineering and data science | Align timestamps, train IDs, units, available-at times and quality flags; resolve critical missing inputs and agree on freshness limits. |
| Validate offline | Data scientist and reliability engineer | Meet approved detection criteria on later periods and separate trains; reduce alert burden without unacceptable delay. Forecasts must improve on the existing planning baseline. |
| Run a shadow pilot | Operations owner and planner | Review candidate outputs alongside existing processes through representative operating modes; document useful alerts, misses, response times and forecast error before approving expansion. |
| Maintain and review | Data science and IT | Monitor freshness, missingness, operating-mode changes, drift, alert rate, delayed outcomes and forecast error. Route breaches to the owner and review before retraining or promotion. |

**Fallback:** keep approved plant alerts and the established planning process. Suppress candidate outputs when critical inputs are stale or missing and show the last successful run. Version data, features, thresholds, and evaluation artifacts for review and rollback. Candidate scores never change control setpoints.

**Plant forecasting adaptation:** the current national export target is not plant production. Agree on daily LNG rundown per train in plant units such as tonnes per day, and a forecast cutoff. Combine historian production with availability, operating mode, feed-gas supply / quality, ambient conditions, and planned maintenance. Include cargo schedules when forecasting shipments or inventory; use schedule versions available at the cutoff. Rebuild a baseline from the last comparable operating period, then evaluate all three approaches on future periods and held-out trains, including outages and ramp-ups. Do not directly convert national export forecasts into plant production estimates.
