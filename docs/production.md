# Production

**Questions.** How close does each terminal run to nameplate? How fast did each ramp up? Does hot weather reduce output?

## Utilization

- Export pace (Bcf/d) = monthly DOE volume ÷ days in month. LNG storage is small relative to output, so monthly exports are a close proxy for production.
- Nameplate and authorized capacity come from the FY2025 10-K (`reference/terminals.csv`).
  - Calcasieu Pass: 10.0 MTPA nameplate, 12.4 MTPA authorized.
  - Plaquemines: 20.0 MTPA nameplate, 27.2 MTPA FERC-permitted.
- Conversion: 1 MTPA ≈ 48 Bcf a year (`vg/common.py`). Published conversions range about 46–52 depending on gas composition; utilization ratios move proportionally.
- Nameplate is the vendor-guaranteed minimum. Venture Global sells output above nameplate as excess capacity, so values over 100% are expected.

## Ramp-up

Months from first cargo until the three-month rolling utilization first reaches 90% of nameplate. Commissioning cargoes count, because they are real output.

## Heat effect (Calcasieu Pass)

OLS: `bcfd ~ temp_mean_c + trend_years + after_cod`, on monthly data from October 2022 (ramp-up complete) to July 2026.

- Temperature is the monthly mean of ERA5 daily mean temperature at the site (Open-Meteo).
- Standard errors are Newey–West with 3 lags.
- Robustness: refit excluding months that overlap a detected low-output window (see `reliability.md`).

**Interpretation.** Gas-turbine-driven liquefaction loses compressor power as inlet air warms, so a negative coefficient is physically expected. Seasonal confounders remain: maintenance is often scheduled in shoulder seasons, and hurricanes arrive in summer. The estimate is an association, not a controlled engineering measurement.

Plaquemines is not estimated: its ramp-up dominates any weather signal.
