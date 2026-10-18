"""Transparent, chronological forecasts with a separate calibration period."""
import numpy as np
import pandas as pd

METHODS = {'last_month': 'Latest month’s daily pace', 'three_month': 'Recent three-month daily pace',
           'seasonal': 'Same month last year’s daily pace'}


def predict(history, target, method):
    """Use only history supplied by the caller; adjust for target month length."""
    rates = history.volume_bcf.to_numpy() / pd.to_datetime(history.month).dt.days_in_month.to_numpy()
    rate = {'last_month': lambda: rates[-1], 'three_month': lambda: rates[-3:].mean(),
            'seasonal': lambda: rates[-12]}[method]()
    return float(rate * pd.Period(target, freq='M').days_in_month)


def analyze(monthly):
    if len(monthly) < 42:
        raise ValueError('Need at least 42 months for training, validation, calibration and testing.')
    # First 12+ months establish history, then 6 validation, 12 calibration, 12 test.
    test_start = len(monthly) - 12
    calibration_start = test_start - 12
    validation_start = calibration_start - 6
    rows = []
    for i in range(validation_start, len(monthly)):
        row = {'month': monthly.iloc[i].month, 'actual': float(monthly.iloc[i].volume_bcf)}
        for method in METHODS:
            row[method] = predict(monthly.iloc[:i], row['month'], method)
        rows.append(row)
    results = pd.DataFrame(rows)
    validation = results.iloc[:6]
    chosen = min(METHODS, key=lambda m: (validation[m] - validation.actual).abs().mean())
    calibration = results.iloc[6:18]
    residuals = (calibration[chosen] - calibration.actual).abs().to_numpy()
    # Finite-sample 80% absolute-residual interval, fixed before held-out testing.
    rank = min(len(residuals), int(np.ceil((len(residuals) + 1) * .8)))
    width = float(np.sort(residuals)[rank - 1])
    test = results.iloc[18:].copy()
    test['prediction'] = test[chosen]
    test['lower'] = np.maximum(0, test.prediction - width)
    test['upper'] = test.prediction + width
    scores = {m: float((test[m] - test.actual).abs().mean()) for m in METHODS}
    next_month = str(pd.Period(monthly.month.iloc[-1], freq='M') + 1)
    estimate = predict(monthly, next_month, chosen)
    report = {'method': chosen, 'method_label': METHODS[chosen], 'mae_bcf': scores[chosen],
              'baseline_mae_bcf': scores['last_month'], 'comparison': scores,
              'test_first': test.month.iloc[0], 'test_last': test.month.iloc[-1],
              'interval_coverage': float(((test.actual >= test.lower) & (test.actual <= test.upper)).mean()),
              'target_month': next_month, 'estimate_bcf': estimate,
              'lower_bcf': max(0, estimate - width), 'upper_bcf': estimate + width,
              'calibration_months': 12, 'nominal_coverage': .8}
    # A review flag uses a trailing window only; it is not a diagnosis.
    monthly = monthly.copy()
    monthly['prior_six_month_average'] = monthly.volume_bcf.shift(1).rolling(6).mean()
    monthly['review_flag'] = monthly.volume_bcf < .75 * monthly.prior_six_month_average
    return report, test, monthly
