# Reliability

**Questions.** When did output fall below its normal pace, and how often does weather explain it?

There is no public sensor data or outage log for either terminal. The cargo schedule is the best public signal: when a plant slows, ships stop leaving.

## Detection (`vg/reliability.py`)

Two transparent detectors. Thresholds are constants at the top of the module.

1. **Cargo gap:** days since the previous departure ≥ max(6 days, 3 × the median of the previous 20 gaps).
2. **Volume dip:** 30-day export pace < 75% of the pace over the 90 days before those 30 days. The baseline never includes the current window.

Flags that overlap merge into one window. The first 120 days after each terminal's first cargo are skipped as start-up.

**Shortfall** is the expected volume at the prior 90-day pace minus the actual volume over the window.

## Labels

- **Weather hazards**
  - NHC HURDAT2 storms passing within 300 km of a terminal.
  - NOAA Storm Events in the terminal parishes and adjacent marine zones: tropical, winter and cold, fog, and strong-wind types. Routine thunderstorm wind is excluded.
  - ERA5 daily reanalysis: hard freezes (minimum ≤ −2 °C) and gale gusts (≥ 75 km/h). This catches storms that are missing from Storm Events, such as Winter Storm Elliott in December 2022.
- **Documented events:** `reference/events.csv`, curated from SEC filings, the IPO prospectus and an IEEFA report citing Louisiana DEQ. Each row has a source URL.

## Matching and evaluation

A window is explained by a hazard active from 3 days before to 10 days after the window begins. This onset rule stops long windows collecting credit for later coincidences.

The **chance rate** places windows of the same length at 2,000 random dates within each terminal's history and applies the same rule.

The reverse check counts how many weather hazards were followed by a window beginning under the same rule.

## Limits

- Five windows is a small sample, so the weather-versus-chance comparison is suggestive, not significant.
- A weather match is coincidence in time, not proof of cause.
- Unexplained windows may be planned maintenance, upstream feed-gas limits, shipping congestion or unreported equipment issues. They are listed for investigation.
- Production can also fall without changing cargo timing (smaller cargoes), which the gap detector cannot see.
