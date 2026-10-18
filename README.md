# Venture Global LNG analytics, from public data

An end-to-end data science project about one company — **Venture Global LNG** and its two operating Louisiana terminals, **Calcasieu Pass** and **Plaquemines**. Seventeen pinned public sources feed a local medallion data platform, five analyses and an offline, multi-page dashboard. Every answer states its method and limits.

**Live site:** https://mhadeli.github.io/lng-sensor-to-decision/ · **Data through July 2026**

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python pipeline.py          # tests → bronze checks → silver → gold → site/  (~2 minutes)
open site/index.html
```

No cloud account, API key or network access is needed to build: raw inputs are committed as checksummed snapshots in `bronze/`.

## Questions and answers

| Page | Question | Answer (public data) |
|---|---|---|
| Exports | Where does the LNG go; what comes next month? | Destination mix by terminal; Calcasieu Pass month-ahead estimate with a 12-month held-out track record |
| Production | How close to capacity; does heat cut output? | Both terminals run above nameplate. Plaquemines reached 90% of nameplate in 9 months. Calcasieu Pass loses ≈0.9% of nameplate per °C of monthly mean temperature (p = 0.002) |
| Reliability | When did output drop; was weather the cause? | 5 low-output windows from cargo cadence (≈46 Bcf shortfall). 2 match weather (Dec 2022 freeze, Hurricane Beryl) against 25% expected by chance; 3 are listed for review |
| Commercial | Do cargoes follow the best netback; can feed gas be forecast? | Europe share rises with the modeled Europe advantage (ρ = 0.40). SEC-reported prices track market netbacks before COD and drop to contract levels after. Conv1D + LSTM beats a naive Henry Hub forecast by ~3%, which is not significant |
| Fleet | Who carries cargoes; which routes; how long at berth? | Own ships carry ~12% of cargoes. Asia round trips (median 79 days) fit Suez/Cape far better than Panama. AIS berth stays (~26 h) matched all 14 DOE departures in October 2023 |
| Data | Can the numbers be trusted? | Contract checks on every table; DOE volumes within 3% of SEC filings every quarter, with cargo counts matching exactly |

## Code

| Path | Responsibility |
|---|---|
| `pipeline.py` | One command: tests, build, render, run audit (`data/gold/run.json`) |
| `vg/bronze.py` | Source registry, deterministic snapshots, checksum manifest, `--refresh` |
| `vg/ais.py` | Streams a month of national AIS (~11 GB) down to a 0.6 MB extract |
| `vg/silver.py`, `vg/sec.py`, `vg/contracts.py` | Validated tables, SEC table parsing, data contracts |
| `vg/production.py`, `vg/reliability.py`, `vg/commercial.py`, `vg/price_model.py`, `vg/fleet.py`, `vg/shipping.py`, `vg/forecast.py` | Analyses |
| `vg/gold.py`, `vg/site.py`, `vg/web/` | Marts, DuckDB warehouse, page payloads and templates |
| `reference/` | Curated, cited inputs: terminal capacities, events, destination ports, shipping assumptions |
| `tests/` | Leakage, contracts, parsing, detection, voyage arithmetic, port-call matching |
| `docs/` | Methodology and limits for each page and for the data platform |

## Sources

DOE LNG export workbook · Venture Global 10-K/10-Q (SEC EDGAR) · EIA Henry Hub · IMF prices via FRED · Open-Meteo / ERA5 weather and marine data · NHC HURDAT2 · NOAA Storm Events · NOAA MarineCadastre AIS. Licenses and hashes are listed on the Data page and in `bronze/manifest.json`.

Venture Global's own ships, both terminals and every price series are covered through mid-2026. Ship positions are public only through 2023, so berth and weather-speed analysis covers Calcasieu Pass in October 2023. Spot JKM and charter rates are not public, so they appear as proxies and stated assumptions.

*Independent portfolio project. Not affiliated with or endorsed by Venture Global.*
