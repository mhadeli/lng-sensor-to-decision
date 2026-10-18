# Model card: maintenance review and terminal forecasting

## Purpose and scope

Support an engineer's review queue by predicting a component failure in the next 24 hours on simulated equipment. The second, independent task forecasts monthly LNG vessel exports from Sabine Pass. Neither model controls plant equipment. There is no connection between the simulated machines and the real terminal.

## Maintenance methods

Exactly three approaches: a normal-operation sensor rule, a regularized MLP (16 → 8 units), and XGBoost. The rule scores the maximum standardized deviation of four trailing sensor means, fitted on negative training observations. The learned methods use 26 causal features. Class weighting handles imbalance. Scores are not calibrated probabilities.

Training uses known machines in January–August 2015; validation uses September–October; testing uses November–December. Remove decisions whose 24-hour target window reaches a split boundary. Every fifth machine ID is excluded from fitting and tuning across all dates, then assessed separately during the test period. IDs never enter the feature matrix.

Tune only two settings per learned model and a short threshold grid. Select the largest validation event recall while staying at or below two false review requests per 100 machine-days. Break ties with fewer false requests. If no candidate meets the budget, select the least noisy candidate and mark the validation gate unmet. Refit on known machines before the test cutoff; keep chosen settings and thresholds fixed. The illustrative budget needs an operations owner before production use.

## Event metrics

Issue at most one review per machine in 24 hours, using scores only. An incident is one machine/failure timestamp, combining simultaneous component failures. Count an incident only if all four preceding six-hour decision opportunities are available. A warning must precede the incident by up to 24 hours. Earliest qualifying warning defines lead time. Missed incidents remain in the recall denominator and do not become zero lead times.

A false review has no recorded failure in the following 24 hours. Exposure is six hours per eligible scoring row, converted to machine-days. This measures a simulated review workload, not actual operator work or normal-only runtime. Reports retain both known-machine and unseen-machine results, sample sizes, precision/recall and average precision.

## Terminal forecasting

Fit a daily-rate carry-forward baseline, a small MLP, and XGBoost. Calendar seasonality and lagged growth predict a change from the baseline. For target month t, features and training targets stop at t−3. This represents a conservative fixed availability assumption, not exact historical publication vintages. The baseline obeys the same restriction.

A 12-month walk-forward validation period selects settings and the headline method. Refit at each origin during the subsequent 24-month test without using unavailable targets. Report MAE in Bcf and mean absolute percentage error. Keep the simple baseline if validation favors it; do not select a method using final test results.

## Limits and intended interpretation

Microsoft's data is synthetic and may contain easier patterns than real equipment. Few failure incidents on held-out machines make apparently excellent recall uncertain. Source timestamps lack arrival delays, machine operating modes, repair durations and intervention outcomes. Validation budgets are proposed, not business-approved. No savings, avoided incidents or safety improvement can be inferred.

DOE data is revised, and shipment volume also reflects inventory and vessel schedules. Export forecasting does not establish plant production efficiency. Model confidence intervals and a real-time weather forecast are not implemented.

Each new evaluation after development is retrospective; a plant pilot requires a genuinely new period and independent event labels. The registry records input hashes, code hashes, versions and fitted-model checksums. Historical replay checks deployment consistency, not prospective effectiveness.
