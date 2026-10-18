# Exports

**Question.** How much LNG did each Venture Global terminal export, where did it go, and what is a reasonable estimate for the first unreported month?

## Data rules

- Source: DOE LNG export workbook, January 2016–July 2026 vintage (`doe_exports` in `bronze/manifest.json`).
- Keep every row with Activity = Exports, Mode = Vessel, Gas Type = LNG at `Cameron (Calcasieu Pass), LA` or `Plaquemines, LA`. Terminals are mapped in `reference/terminals.csv`.
- One Calcasieu Pass row lists Cameron LNG, LLC as the company. It is kept because it is a physical export from the terminal, and recorded as a warning in the quality report.
- Split allocations are legitimate: one cargo can serve several buyers. Volumes add every allocation. A **cargo** is a distinct terminal + vessel + date; its main destination is the largest allocation.
- MMcf / 1,000 = Bcf of natural gas equivalent.
- Every month from a terminal's first cargo through July 2026 must have exports. A missing month stops the build instead of being treated as zero.
- Every destination country must map to a region in `reference/destinations.csv`.

## Calcasieu Pass estimate

Compare three daily-rate methods: latest month, trailing three-month mean, and same month last year. Multiply the daily rate by the days in the target month. At every origin only earlier months are used.

- Last 12 months: test.
- Preceding 12 months: calibrate an 80% range (absolute-residual order statistic, rank ⌈(n+1)·0.8⌉).
- Six months before that: choose the method by mean absolute error.

Method choice and range width are frozen before testing, and the page reports actual test coverage.

Plaquemines (first cargo December 2024) is still ramping up and has too little stable history for the same design.

## Limits

These are retrospective next-observation estimates on a revised workbook. They do not reproduce DOE publication delays or earlier data vintages. The August 2026 estimate from a July 2026 cutoff is not a current-calendar forecast. Review flags (months more than 25% below the prior six-month mean) are prompts, not diagnoses.
