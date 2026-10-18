# Calcasieu Pass export monitoring

One terminal operated by Venture Global; one primary source: the DOE LNG exports workbook, January 2016–July 2026 vintage. Its SHA-256 is pinned in `lng/data.py`. Filter exactly to Cameron (Calcasieu Pass), LA, LNG, Exports, Vessel. Terminal history begins with reported shipments in March 2022. This is terminal activity, not consolidated company sales.

## Data rules

Keep every allocation row and its original spreadsheet row number. Aggregate volume in MMcf and divide by 1,000 to display Bcf (natural gas equivalent, not liquid LNG cubic feet). Count distinct vessel/date pairs as approximate departures; spelling changes can affect counts. Do not collapse split cargo allocations. Fail on nonpositive/nonfinite volumes, missing dates/vessels/destinations, missing monthly coverage or source checksum changes. A missing month is not assumed to mean zero shipments. The DuckDB warehouse stores cleaned allocations and monthly totals; CSV downloads also include review flags.

## Forecast and evaluation

Compare three transparent daily-rate methods: latest month, trailing three-month mean, and same month of the previous year. Multiply the daily rate by target-month days. At each origin use only earlier observations. Reserve the last 12 months for testing, the preceding 12 for range calibration, and the six months before those for model selection by mean absolute error (MAE). Ties favor the first, simplest method. At least 12 initial months remain before validation.

The 80% range uses an absolute-residual order statistic from the calibration period: rank ceil((n+1) × 0.8). Clip its lower bound at zero. Freeze model choice and interval width before testing. Display actual test coverage rather than assuming 80% is achieved. Temporal changes mean nominal interval coverage is not guaranteed. Fit the final estimate using all observed history, for the first month after the source cutoff.

These are retrospective next-observation estimates. They assume the previous month’s export totals are available and use a revised workbook; they do not reconstruct historical publication delays or original releases. The August 2026 estimate from a July 2026 cutoff is not a current-calendar forecast. Operational forecasting requires release timestamps, archived vintages, and evaluation at the actual decision date.

A review flag means volume is more than 25% below the previous six-month mean. It is a descriptive rule, not a statistically validated detector or evidence of equipment failure. Start-up and commercial-operation transitions can change export patterns. No causal attribution, revenue estimate, plant efficiency, live tracking or maintenance diagnosis is inferred.

## Rebuild and update

Run `python pipeline.py` or `python pipeline.py --offline` after installing requirements. Outputs live in `data/lng/`; the offline dashboard is `site/index.html`. The build runs focused data/forecast tests before replacing the dashboard. To adopt a newer DOE workbook, review schema, coverage and source hash, then explicitly change the source constants and rerun tests. No live refresh is implied.
