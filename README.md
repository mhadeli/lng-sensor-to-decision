# LNG portfolio: sensor to decision

A local stand-in for the Venture Global path from plant readings to a production decision. Python, DuckDB, and PySpark run on this machine. No cloud account is required. PySpark stands in for Databricks and needs a local Java runtime.

Plant sensor logs from an operating LNG terminal are not public. Each public file stands in for one source:

| File | Stands in for |
|---|---|
| Tennessee Eastman `d00`–`d21` | Historian tags. `d00` is normal operation. The rest are faults. Training runs `dXX.dat` are faulty from the first row. Test runs `dXX_te.dat` switch the fault on at step 160. |
| EIA `lng_exports.csv` | National LNG volume, million cubic feet |
| EIA `henry_hub.csv` | Gas supply price |
| NOAA `lake_charles_weather.csv` | Ambient temperature at Lake Charles |
| NOAA storm files | Louisiana storm counts |
| EIA `gulf_lng_exports.csv` | Ship and terminal schedule: monthly exports from Sabine Pass, Cameron, Calcasieu Pass, Plaquemines, Corpus Christi, Freeport, and Golden Pass |
| Microsoft `PdM_*.csv` | Work orders and failures for 2015. This sample does not share months with the export series, so it stays in its own table. |

## Dashboard

**Live:** https://mhadeli.github.io/lng-sensor-to-decision/

`dashboard.py` turns the gold tables into one static page, `site/index.html`. The GitHub Action in `.github/workflows/dashboard.yml` runs `download.py`, `prepare.py`, `model.py`, and `dashboard.py` on GitHub's servers and publishes the page to GitHub Pages. It runs on every push to `main`, on the 15th of each month after EIA's monthly update, and from the Actions tab on demand. If a source is down and a step fails, nothing is deployed and the last good page stays online. No local machine needs to be on.

One-time setup: in the repository, open **Settings → Pages** and set **Source** to **GitHub Actions**.

## Setup

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python download.py
python prepare.py
python model.py
python dashboard.py   # optional: open site/index.html
```

`data/raw` is the landed layer. `prepare.py` writes cleaned tables to `data/silver` and the joined tables to `data/gold`:

- `monthly.csv` joins exports, gas price, temperature, Louisiana storms, and Gulf terminal volume for the months the storm files cover
- `alerts.csv` flags a sensor row that leaves the normal-operation band (mean ± 3 standard deviations on the normal training run)
- `volume_gap.csv` is national exports minus the Gulf terminal total
- `maintenance.csv` is the 2015 work-order and failure count
- `fault_predictions.csv` compares that rule with a random forest that sees the current readings and how they just changed. Both are scored on the test runs, which the model never trains on
- `volume_forecast.csv` is the next-month export forecast for the last 24 months, next to the actual value and the baseline "same as last month"

## Results

Fault detection on the 21 test runs plus the normal test run, 21,120 rows:

| Method | Faults caught | Normal rows cleared |
|---|---|---|
| Rule: any sensor outside the normal band | 79% | 82% |
| Random forest | 83% | 72% |

The random forest catches a few more faults and raises more false alarms. It is not clearly better than the rule. Faults 3, 9, and 15 are close to a coin flip for both. The benchmark literature reports the same thing: those faults barely move the measurements.

Next-month U.S. LNG export forecast, scored on the last 24 months. Each month is forecast from earlier months only.

| Method | Mean miss | vs baseline |
|---|---|---|
| Same as last month (baseline) | 25.3 Bcf | |
| 1. Same daily rate: last month's daily rate times the days in the coming month | 21.4 Bcf | -16% |
| 2. Step 1 times the median change into that calendar month since 2019 | 22.8 Bcf | -10% |
| 3. Average of 1 and 2 (final) | 19.7 Bcf | -22% |

The EDA drove each step. Months have different lengths, and exports dip every summer. The calendar-month pattern is real but noisy, so the final forecast averages it with step 1 using fixed equal weights. Nothing is tuned on the test months. An earlier linear regression on gas price, temperature, and storm counts did not beat the baseline, so it was removed. The remaining misses come from plants ramping up and from cargo timing, which need plant-level inputs.

## What a production engineer would do

An alert means inspect that train before raising its daily target. A forecast below recent exports means check the gas price, the Lake Charles temperature, and the Gulf terminal volume for that month before setting the plan.
