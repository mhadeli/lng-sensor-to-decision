# Data platform

A local medallion architecture with no cloud services: Python, DuckDB and Parquet.

```
bronze/   committed raw snapshots + manifest.json (SHA-256, URL, license, retrieval time)
   │  vg/bronze.py verify()  – any changed or missing file stops the build
   ▼
silver    validated tables keyed by terminal, vessel and date      data/silver/*.parquet
   │  vg/silver.py + vg/contracts.py check()
   ▼
gold      one mart per question                                    data/gold/*.parquet
   │  vg/gold.py
   ▼
warehouse data/warehouse.duckdb (schemas silver, gold) and site/ (static pages + CSV downloads)
```

## Why commit bronze

Several sources change in place: FRED, EIA, Open-Meteo and SEC company data. Pinning snapshots in git makes every build reproduce the same numbers, including in CI, with no network access. Accepting new data is an explicit, reviewable step:

```bash
python -m vg.bronze --refresh              # all sources
python -m vg.bronze --refresh fred_asia_lng
git diff bronze/manifest.json              # review changed hashes, then commit
python -m vg.ais --month 2023-10           # rebuild the AIS extract (streams ~11 GB)
```

Large upstream sources are reduced at ingestion:

- NOAA Storm Events is filtered to the terminal parishes and marine zones.
- AIS is filtered to the relevant ships, with upstream hashes in a provenance file.

Total bronze size is about 4 MB.

## Contracts

`vg/contracts.check()` validates required columns, nulls, unique keys, value ranges and minimum rows for every reference, silver and gold table. A failure raises and stops the build before `site/` is replaced. Non-fatal findings, such as the one Calcasieu Pass row listing Cameron LNG, are recorded as warnings. Results appear on the Data page and in `data_quality.csv`.

## Reconciliation

DOE volumes are converted at 1,038 Btu/cf and compared with the "LNG volumes exported" in each 10-Q and 10-K, per project and quarter. Cargo counts are compared too.

## Build

`python pipeline.py` runs, in order:

1. unit tests;
2. bronze verification;
3. silver;
4. gold;
5. site.

It writes `data/gold/run.json` with step timings and code hashes. The site is rendered into a staging folder and swapped in only when every page succeeds. GitHub Actions runs the same command and deploys `site/` to GitHub Pages.

## Cloud mapping

| Local | Azure / Databricks counterpart |
|---|---|
| `bronze/` + manifest | ADLS Gen2 raw zone, ADF copy activities with checksums |
| silver/gold Parquet + DuckDB | Delta tables in Unity Catalog |
| `vg/contracts.py` | Delta constraints / DLT expectations |
| `pipeline.py` + GitHub Actions | Databricks Workflows or ADF pipeline triggered from Azure DevOps |
