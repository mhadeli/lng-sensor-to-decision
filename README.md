# Equipment reliability and LNG terminal planning — runs locally

A complete laptop-based data science workflow for a senior data scientist portfolio: ingest operational tables, validate and join them with SQL, predict failures, evaluate a bounded review queue, replay historical inputs through a saved model, and explain results in a simple dashboard. A separate terminal-export forecast supplies real LNG context.

**No Azure account, Java, paid services or server is required.** Python, DuckDB and Parquet provide the local implementation. [jd.md](jd.md) is the target role; [docs/pilot_plan.md](docs/pilot_plan.md) maps the implemented workflow to its lifecycle and platform requirements without claiming cloud deployment.

## Run and view

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py
```

Open **`site/index.html`**. Charts, styling and the replay are bundled locally. Share the whole `site/` folder for an offline demonstration. The existing GitHub Pages destination is https://mhadeli.github.io/lng-sensor-to-decision/; local edits become public only after a push and successful deployment.

After the first download:

```bash
python pipeline.py --offline
```

This verifies cached source checksums and rebuilds the entire project. Per-stage logs and run audits are stored in `data/runs/`; `data/gold/latest_run.json` points to the latest successful execution. The dashboard is generated only after the tests pass. On this Mac, XGBoost is already linked to scikit-learn's bundled OpenMP runtime; a fresh macOS environment needs `libomp` (typically `brew install libomp`).

## What I implemented

| Step | Code | Output |
|---|---|---|
| Ingest | `download.py` | Five machine CSVs, DOE workbook, source manifest and checksums |
| Validate and join | `prepare.py`, `sql/features.sql` | Quarantine records, clean Parquet tables, local DuckDB warehouse, causal feature table |
| Train and evaluate | `model.py`, `evaluation.py` | Baseline / MLP / XGBoost, validation search, saved model registry, event and alert metrics |
| Forecast exports | `forecasting.py` | Reporting-aware walk-forward forecasts for Sabine Pass |
| Replay scoring | `replay.py` | Time-filtered historical batches, review requests and monitoring logs |
| Check and present | `tests/`, `dashboard.py`, `templates/`, `assets/` | Automated checks and an offline interactive dashboard |
| Orchestrate | `pipeline.py` | One command, stage logs and a run audit |

The default pipeline deliberately uses two independent demonstrations. Simulated equipment failures are never joined to real terminal exports. The obsolete Tennessee Eastman notebook, Spark requirements and generated analysis outputs have been removed; the earlier implementation remains available in git history. `data_usage.md` and `jd.md` are retained as research references. Original cached source files are preserved but only the sources in the current manifest are consumed.

The dashboard footer links to `build_metadata.json`, which records its build time and source revision. GitHub Pages builds use the deployed commit SHA so the online version can be checked against the repository.

## Public data

- **Microsoft predictive maintenance:** 876,100 hourly sensor records for 100 simulated machines, plus errors, component replacements, failures and machine metadata. The shared machine IDs allow meaningful operational joins. [Data files](https://github.com/microsoft/sqlworkshops/tree/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data)
- **DOE LNG export details:** a pinned September 2026 workbook covering January 2016–July 2026. Retain domestic LNG exports by vessel and aggregate Sabine Pass transactions from January 2017 onward. Preserve split cargo allocations; don't assume each source row is a complete vessel shipment. [Source report](https://www.energy.gov/hgeo/articles/natural-gas-imports-and-exports-monthly-2026)

A normal run uses cached files. `python download.py --refresh` checks the pinned sources and stops if content changed rather than silently accepting revisions. Updating the DOE vintage requires reviewing and versioning the source and its coverage, not simply rerunning on a schedule.

## Maintenance problem and evaluation

**Decision:** which machines should an engineer review because a component may fail in the next 24 hours?

At six-hour decision times, use the preceding 24 hours of sensor means/variability and error counts, equipment age/model, and time since past component replacement. Current and future sensor measurements are excluded from the feature window. Future failures create labels only. Require complete history and a complete future observation window before using a row for training/evaluation.

Use January–August 2015 for training, September–October for validation, and November–December for testing. Purge overlapping 24-hour target windows across splits. Every fifth machine ID is excluded from all fitting and tuning, then tested separately. Only 26 features enter the models; machine identifiers and timestamps do not.

Compare a normal-reference rule, a regularized MLP (16 → 8), and shallow XGBoost. Select model settings and thresholds by validation incident recall subject to an illustrative limit of two false reviews per 100 machine-days. Emit at most one review per machine every 24 hours. Simultaneous component failures count as one incident; evaluate only incidents with four preceding six-hour opportunities. Lead-time medians exclude missed incidents, which remain in the event-recall denominator.

The dashboard distinguishes later-period tests on known machines from tests on unseen machines. Model scores are not calibrated probabilities. Excellent performance on this simulated data is not evidence of effectiveness on real LNG equipment. [Full model card](docs/model_card.md)

## Terminal planning problem and evaluation

Forecast monthly vessel exports from **Sabine Pass, LA**, selected for its long history before modeling. This is shipped gas volume, not plant production or production efficiency.

The baseline carries the latest available daily rate into the target month's calendar. MLP and XGBoost use seasonality and lagged growth. At target month t, both features and training targets stop at t−3 to approximate public reporting delay. A 12-month walk-forward validation period selects settings before the 24-month test. This fixed lag does not reconstruct historical publication dates or revisions; the backtest uses the pinned revised workbook.

## Current local result snapshot

Maintenance candidate selected on validation: **XGBoost**. Later-period test on **20 unseen machines**:

| Method | Eligible incidents warned | Median lead, warned only | False reviews / 100 machine-days |
|---|---:|---:|---:|
| Rule baseline | 14 / 22 | 21 h | 5.32 |
| MLP | 22 / 22 | 18 h | 1.83 |
| XGBoost | 22 / 22 | 24 h | 0.33 |

The small incident count and synthetic source limit the conclusion. The model remains a research candidate; a plant pilot needs independent labels and agreed acceptance criteria.

Sabine Pass backtest, August 2024–July 2026:

| Method | Average absolute error | Average percentage error |
|---|---:|---:|
| Daily-rate baseline | 12.03 Bcf | 10.39% |
| MLP | 13.91 Bcf | 11.77% |
| XGBoost — selected on validation | 6.24 Bcf | 5.57% |

XGBoost's average error is about 48% lower than this reporting-aware baseline. These numbers describe the pinned dataset; the dashboard always derives metrics from generated artifacts.

## Replay and monitoring

`python replay.py` reloads the registered model and reconstructs features for a held-out historical week, one six-hour batch at a time. Scoring reads telemetry, errors, maintenance and machine metadata only; it does not read future failure outcomes. A selected historical period determines the replay window, not the recommendations.

The dashboard slider displays the precomputed batches. Missing/stale sensor history suppresses a score. Logs include scored/suppressed counts, emitted reviews, and feature-mean shifts relative to the training reference. This lightweight shift check is diagnostic, not a validated production drift test. A 24-hour cooldown limits duplicate requests. Tests compare replay outputs against offline scores.

## Evidence beyond model accuracy

- [Data contract](docs/data_contract.md): table keys, quality rules, available-at assumptions, features and labels.
- [Model card](docs/model_card.md): selection policy, holdouts, metrics and limitations.
- [Runbook](docs/runbook.md): reproducibility, warehouse inspection, demonstration and recovery.
- [Pilot plan](docs/pilot_plan.md): owners, dependencies, acceptance gates, fallback and a mentoring/handoff exercise.

The local implementation demonstrates Python/SQL, ETL requirements, future-event modeling, testing, maintainability and stakeholder communication. Azure Data Factory, Data Lake and Databricks are documented future counterparts, not tools claimed to be running here. GitHub Actions runs the same local workflow on pushes and manual triggers; only a successful build is deployed.
