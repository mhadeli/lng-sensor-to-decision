# Fleet

**Questions.** Who carries the cargoes? How long do round trips take, and which routes do Asia voyages use? How long do ships wait and berth? Do waves and wind slow them?

## Own fleet

Own ships are the seven *Venture*-named tankers in the DOE data. The FY2025 10-K says nine are contracted and seven delivered.

## Round trips and route inference (DOE, 2022–2026)

- A round trip runs from a Venture Global loading to the same vessel's next loading at **any** U.S. terminal; the full DOE workbook is kept as `silver.us_loadings` for this purpose.
- Gaps over 150 days are treated as other trading and excluded.
- For each Asia voyage, the implied average speed for each route is `2 × distance / ((days − port days) × 24)`. A route is plausible at 13–20 knots. A voyage is labeled Panama or Suez/Cape only when exactly one route is plausible.
- Waiting, slow steaming and intermediate calls lengthen trips, so labels are inferences, not observations.

## Port calls (AIS, Calcasieu Pass, October 2023)

- `vg/ais.py` streams 31 daily NOAA MarineCadastre files (~11 GB) and keeps two sets of positions in the Gulf:
  - ships whose names match DOE Venture Global loadings;
  - large ships near the terminal.
- It resamples to 10 minutes and stores a 0.6 MB extract plus upstream file hashes (`bronze/ais/*.provenance.json`).
- **Berth stay:** at least 6 hours within 1 km of the berths (29.7735 N, 93.3415 W, located from stationary AIS positions) at under 1 knot.
- **Waiting:** stopped (under 1 knot) within 60 km before berthing.
- **DOE match:**
  - by vessel name within one day;
  - otherwise the only unmatched DOE cargo that day, labeled "date only". This handles renamed ships (*Meridian Spirit* → *Seapeak Meridian*) and abbreviated AIS names.
- AIS for 2024 onward is not downloadable, so Plaquemines and Venture Global's own ships are not covered.

## Speed versus weather

- Positions: over 80 km from the terminal, above 6 knots, within six days after (laden) or before (ballast) a berth stay.
- Joined to hourly Open-Meteo wave height and ERA5 wind on a 1° grid.
- Model: OLS `sog ~ wave + wind + leg + ship fixed effects`, with errors clustered by ship.
- One calm month and a dozen ships cannot measure storm effects. Fuel burn is not public and is not estimated.

## Voyage costs

Same formula as `commercial.md`. The low case uses low charter rates and fees with the highest speed; the high case uses the reverse.
