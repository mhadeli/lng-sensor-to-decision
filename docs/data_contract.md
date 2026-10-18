# Local data contract

This is a demonstration contract, not a specification approved by an operating plant.

## Inputs

| Table | Key and contents | Validation / availability |
|---|---|---|
| telemetry | machineID + datetime; volt, rotate, pressure, vibration | Hour-aligned times, finite positive measurements, unique key, known machine |
| machines | machineID; model and age | Positive integer ID, nonnegative age, one of four documented model categories |
| errors | machineID + datetime + errorID | Known machine, error1–error5, hour-aligned times |
| maintenance | machineID + datetime + comp | Known machine, comp1–comp4; only replacements strictly before scoring time become features |
| failures | machineID + datetime + failure | Known machine, comp1–comp4; used only to construct future labels and evaluate outcomes |
| DOE transactions | source file + workbook row number | Valid date, terminal and nonnegative volume; preserve allocations rather than deduplicating by vessel/date |

Machine data is Microsoft's simulated 2015 equipment history, with earlier maintenance records. No timezone or arrival timestamps are supplied; retain the source's naive timestamps and explicitly assume recorded-time availability. Sensor units are source units; do not silently reinterpret these as LNG instruments. `sql/features.sql` excludes current-time observations as well as future values.

DOE's pinned September 2026 workbook covers January 2016–July 2026. The model uses Sabine Pass, LA from January 2017, selected for history length before evaluation. Include domestic LNG exports by vessel; exclude re-exports and ISO containers. Volume is million cubic feet of gas, not liquid volume or tonnes. Missing terminal months fail validation rather than becoming assumed zeroes.

## Layers and quality

- `data/raw/`: original source files and a manifest containing URLs, checksums, byte sizes, schema version and verification time. `verified_at` is not a claim about the original download time of a cached file.
- `data/quarantine/`: rejected rows with a generic reason; the raw original remains available for investigation. A missing DOE date is visible here rather than repaired by guessing.
- `data/silver/`: validated Parquet tables and the feature table; `exports.csv` is the selected terminal series.
- `data/operations.duckdb`: queryable copies of machine sources, feature rows and LNG transactions. Preparation replaces tables, so reruns do not append duplicates.
- `data/gold/`: predictions, evaluations, review logs and quality results.
- `data/models/`: locally fitted model files and a registry. Only load trusted locally generated joblib files.

Wrong schemas or empty valid tables stop the build. Invalid records and duplicate keys are quarantined. A scoring window must contain exactly 24 prior hourly telemetry records plus a current timestamp anchor. No forward/back fill repairs a missing hour. The dashboard exposes quarantine counts; production error-rate tolerances would need agreement with a data owner.

## Feature and label definitions

At each six-hour decision time t, sensor means/standard deviations and error counts use [t−24h, t). Replacement age uses only maintenance before t. Missing replacement history is represented by a separate unknown flag and a fixed 365-day placeholder. Machine age and model are treated as known static metadata.

The binary target is any component failure in (t, t+24h]. Require all 24 future hourly observations before assigning a label. Exclude decisions at an actual failure instant. Preserve unlabeled tails for scoring, but never treat them as negative training examples. Machine and timestamp are identifiers, not model features.

## Source changes

Downloads are cached. `download.py --refresh` verifies the pinned source; if its content changes, the replacement stays in `.part` and the build stops for review. Version the source URL/file and update the contract before accepting a new report. Normal pipeline runs do not silently advance the DOE reporting vintage.
