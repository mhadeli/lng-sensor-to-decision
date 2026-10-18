# Calcasieu Pass LNG export monitoring

A focused, local data science project for **Venture Global’s Calcasieu Pass terminal**. One public DOE workbook powers export monitoring, destination analysis, simple forecasts, and an offline dashboard designed for nontechnical readers.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py
```

Open **[site/index.html](site/index.html)** in your browser. No server, API key or paid service is required. The entire `site/` folder can be shared for offline viewing. To rebuild from the verified cached workbook:

```bash
python pipeline.py --offline
```

## What the dashboard answers

- How much gas did the terminal export in the latest reported month?
- How many vessel departures and destination countries were reported?
- How have volumes changed, and which destinations received the most?
- What is the estimate for the first month after the dataset ends?
- How close were the method’s estimates on 12 held-out historical months?

The period selector updates export and destination charts. The latest-month cards and forecast retain their clearly labeled dates. Hover over charts for values and download the underlying totals or allocation records.

**Data cutoff: July 2026.** The next-unreported-month estimate is August 2026, not a live forecast for today. Bcf means billion cubic feet of natural gas equivalent. Shipment volumes are not plant production or efficiency.

## Small, readable codebase

| File | Responsibility |
|---|---|
| `pipeline.py` | Run preparation, forecasting, tests and dashboard generation |
| `lng/data.py` | Verify one source; filter and validate allocations; build CSVs and DuckDB tables |
| `lng/forecast.py` | Compare three simple methods; chronological evaluation and estimated ranges |
| `lng/dashboard.py` | Package validated outputs into an offline page |
| `lng/dashboard.html`, `.css`, `.js` | Plain-language layout, styling and interactive charts |
| `tests/test_lng.py` | Allocation counting, missing coverage, calendar adjustments and leakage checks |

Generated data and the warehouse are in `data/lng/`. The reviewed source is cached in `data/raw/`. A newer source requires an explicit URL, hash and coverage review.

## Forecasting choices

Compare the latest month’s daily export rate, the recent three-month average daily rate, and the same month last year. Select using six earlier validation months, calibrate an approximate 80% range on the following 12 months, and evaluate on the final 12 months. Each prediction uses only prior observations. Show average absolute error in the same units as exports, along with actual range coverage.

The backtest assumes the preceding observation is available. It does not reproduce DOE reporting delays or original publication vintages. This makes it a retrospective planning benchmark, not evidence of live trading performance. [Full methodology and limitations](docs/lng_methodology.md)

## Source and scope

[DOE Natural Gas Imports and Exports Monthly 2026](https://www.energy.gov/hgeo/articles/natural-gas-imports-and-exports-monthly-2026), pinned January 2016–July 2026 LNG export details workbook. Select domestic vessel LNG exports from `Cameron (Calcasieu Pass), LA`; observed terminal history begins March 2022. Preserve split allocations and count distinct vessel/date combinations as approximate departures. This covers terminal activity across reported exporters, not all Venture Global facilities or consolidated sales.

The GitHub Pages workflow runs this pipeline when changes are pushed to `main`. Building locally does not publish changes.
