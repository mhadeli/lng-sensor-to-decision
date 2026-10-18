# Run locally and review the result

## First build

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py
```

Open `site/index.html` in a browser. Plotly and the replay data are bundled locally, so no server or internet connection is needed to view the built dashboard. Share the entire `site/` directory, including CSS, JavaScript, charts, downloads and documentation.

On macOS, XGBoost needs OpenMP. This workspace's installed XGBoost library already links to scikit-learn's bundled OpenMP runtime. For a fresh installation, install `libomp` using Homebrew if available. This is an environment dependency, not a cloud requirement.

## Repeatable and partial runs

```bash
python pipeline.py --offline             # Verify cached sources and rebuild everything
python prepare.py                       # Rebuild validated tables and features
python model.py                         # Train/evaluate maintenance candidates
python forecasting.py                   # Backtest terminal exports
python replay.py --start 2015-12-01 --days 7
python -m unittest discover -s tests -v
python dashboard.py                     # Render from the latest saved outputs
```

Only the first build needs downloads. The DOE snapshot is pinned for reproducibility. The standard pipeline does not claim to retrieve the latest data every month.

## Demonstration for an interviewer

1. Show the dashboard's ingestion/preparation/modeling steps and source manifest.
2. Compare models on machines absent from training and explain the review budget.
3. Move the replay slider. It shows precomputed historical batches; use `replay.py` to actually regenerate those scores from time-filtered source tables.
4. Explain why terminal exports are separate from simulated maintenance and how the publication cutoff works.
5. Show `sql/features.sql`, a leakage test, the model card and pilot acceptance gates.

## Inspect the warehouse

```python
import duckdb
con = duckdb.connect("data/operations.duckdb", read_only=True)
print(con.sql("SELECT machineID, count(*) AS hours FROM telemetry GROUP BY machineID LIMIT 5").df())
print(con.sql("SELECT * FROM features LIMIT 5").df())
```

## Failures and recovery

Every pipeline run stores per-stage logs and an audit in `data/runs/<run_id>/`. A failure stops subsequent stages, including dashboard generation. The last successful run audit is `data/gold/latest_run.json`. Intermediate tables/model files may have been rebuilt before failure: fix the error and rerun the complete pipeline before sharing outputs.

For schema errors or quarantines, inspect the raw source and `data/quarantine/`; do not silently invent sensor values or missing export months. A modified source checksum requires deliberate review and versioning. If model and registry hashes differ, rerun `model.py`, `replay.py`, tests, and the dashboard in order. Keep joblib inputs local and trusted.

Replay suppresses scoring for incomplete/stale sensor histories. Monitor `replay_monitoring.csv` for scored/suppressed counts, review requests and shifted feature means. The shift diagnostic is not a statistically validated drift detector. Retraining and promotion are deliberate actions; nothing automatically modifies operating rules.

The GitHub Actions workflow builds the same local pipeline and can publish `site/` to Pages. No Azure account, credentials, distributed cluster or paid service is needed.
