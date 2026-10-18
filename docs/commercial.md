# Commercial

**Questions.** Which market paid more after shipping? Did cargoes follow the better netback? How do reported prices compare? Can Henry Hub be forecast weeks ahead?

## Prices

| Use | Series | Caveat |
|---|---|---|
| Feed gas | EIA Henry Hub daily spot (RNGWHHD) | — |
| Europe | FRED `PNGASEUUSDM` (IMF, EU natural gas, TTF-linked) | Monthly average |
| Asia | FRED `PNGASJPUSDM` (IMF, Japan LNG import price) | Mostly oil-linked contracts; lags and smooths spot JKM, which is paid data |

## Netbacks (`vg/shipping.py`, `vg/commercial.py`)

FOB netback at Calcasieu Pass = P × (1 − boil-off) − (round-trip days × charter rate + canal fees) / cargo MMBtu − regasification fee.

- Benchmarks: Rotterdam (Europe) and Tokyo (Asia), the delivery points behind TTF and JKM.
- Sea distances come from `searoute` (open-source maritime network) with three route options:
  - shortest (Panama or Suez allowed);
  - no Panama;
  - Cape (no Panama, no Suez).
- Cargo size: 3.6 Bcf.
- Assumptions with low/base/high values are in `reference/assumptions.csv`. Monte Carlo uses 4,000 triangular draws per month.

## Destination choice

Monthly Europe share of Europe + Asia volume against the modeled Europe-minus-Asia netback. Reported as a Spearman rank correlation plus direction agreement (majority destination = the region with the better netback).

## Realized prices

The weighted average price of LNG sold, per project and quarter, is parsed from the 10-Q and 10-K filings (`vg/sec.py`).

- When filings overlap, the latest filing wins for each field. Q4 = annual total minus Q1–Q3. An em dash in SEC tables means zero.
- The modeled comparison is volume-weighted over that quarter's actual Europe and Asia cargoes.
- Interpretation: Plaquemines sold commissioning cargoes at market before COD. Calcasieu Pass moved to long-term contracts (fixed fee + 115% of Henry Hub) after COD on 15 April 2025.

## Henry Hub forecasts (`vg/price_model.py`)

- Weekly (Friday) averages of daily prices. The target is the log-price change 1, 4 and 8 weeks ahead.
- Methods:
  - naive (no change);
  - last year's seasonal change;
  - ridge regression on 12 lagged weekly changes plus a seasonal cycle;
  - Conv1D (16 filters, kernel 5) → LSTM (32) → linear head, with 52-week input windows, an ensemble of 5 seeds, Adam and early stopping on the last 15% of the training period.
- Walk-forward: for each test year 2022–2026, refit using only origins whose 8-week targets were observed before 1 January.
- Diebold–Mariano test on absolute errors versus naive (Newey–West, h − 1 lags).

**Result.** No method reliably beats the naive forecast. Henry Hub behaves close to a random walk at these horizons, which is itself a useful planning finding.
