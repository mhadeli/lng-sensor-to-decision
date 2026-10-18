"""Terminal export backtest with an explicit three-month observation lag."""
import json

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from common import GOLD, MODELS, SILVER, write_json

REPORT_LAG = 3  # Forecast month t can use volumes through t-3; conservative publication-delay assumption.
TEST_MONTHS = 24
VALID_MONTHS = 12
FEATURES = ["sin_month", "cos_month", "recent_growth", "annual_growth"]


def export_features(exports):
    frame = exports.sort_values("month").reset_index(drop=True).copy()
    date = pd.to_datetime(frame.month)
    if frame.month.duplicated().any() or list(frame.month) != list(pd.period_range(date.min(), date.max(), freq="M").astype(str)):
        raise ValueError("Expected consecutive unique export months")
    days = date.dt.days_in_month
    rate = frame.exports_mmcf / days
    frame["baseline"] = rate.shift(REPORT_LAG) * days
    frame["target"] = frame.exports_mmcf / frame.baseline - 1
    frame["sin_month"] = np.sin(2 * np.pi * date.dt.month / 12)
    frame["cos_month"] = np.cos(2 * np.pi * date.dt.month / 12)
    frame["recent_growth"] = rate.shift(REPORT_LAG) / rate.shift(REPORT_LAG + 1) - 1
    frame["annual_growth"] = rate.shift(REPORT_LAG) / rate.shift(REPORT_LAG + 12) - 1
    return frame


def make_model(kind, setting):
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8,), solver="lbfgs",
            alpha=setting, max_iter=1000, random_state=42))
    return XGBRegressor(n_estimators=setting, max_depth=2, learning_rate=.03,
        min_child_weight=6, reg_lambda=20, n_jobs=2, random_state=42, base_score=0.0)


def predict_month(frame, i, kind, setting):
    # Even training targets after t-3 are unavailable at this forecast origin.
    history = frame.iloc[:i - REPORT_LAG + 1].dropna(subset=FEATURES + ["target"])
    if len(history) < 24:
        raise ValueError("Need 24 complete training months before forecast origin")
    model = make_model(kind, setting)
    model.fit(history[FEATURES], history.target)
    change = model.predict(frame.iloc[[i]][FEATURES])[0]
    return max(0.0, float(frame.iloc[i].baseline * (1 + change)))


def main():
    frame = export_features(pd.read_csv(SILVER / "exports.csv"))
    test_start = len(frame) - TEST_MONTHS
    valid_start = test_start - VALID_MONTHS
    validation, settings = {}, {}
    valid = frame.iloc[valid_start:test_start]
    validation["baseline"] = float((valid.baseline - valid.exports_mmcf).abs().mean() / 1000)
    with threadpool_limits(limits=2):
        for kind, candidates in [("mlp", [1.0, 10.0, 100.0]), ("xgboost", [40, 100, 200])]:
            errors = {}
            for setting in candidates:
                estimates = [predict_month(frame, i, kind, setting) for i in range(valid_start, test_start)]
                errors[setting] = float(np.mean(np.abs(np.array(estimates) - valid.exports_mmcf)) / 1000)
            settings[kind] = min(errors, key=errors.get)
            validation[kind] = errors[settings[kind]]
        selected = min(validation, key=validation.get)
        output = frame.iloc[test_start:][["month", "exports_mmcf", "baseline"]].rename(columns={"exports_mmcf": "actual"}).copy()
        for kind, setting in settings.items():
            output[kind] = [predict_month(frame, i, kind, setting) for i in range(test_start, len(frame))]
    output["latest_available_month"] = [frame.iloc[i - REPORT_LAG].month for i in range(test_start, len(frame))]
    output.to_csv(GOLD / "terminal_forecast.csv", index=False)
    metrics = {kind: {"mae_bcf": float((output[kind] - output.actual).abs().mean() / 1000),
                      "mape": float(((output[kind] - output.actual).abs() / output.actual).mean())} for kind in MODELS}
    report = {"terminal": "Sabine Pass, LA", "selected": selected, "settings": settings,
              "report_lag_months": REPORT_LAG, "validation_mae_bcf": validation, "test": metrics,
              "validation_first": valid.month.iloc[0], "validation_last": valid.month.iloc[-1],
              "test_first": output.month.iloc[0], "test_last": output.month.iloc[-1], "test_months": len(output)}
    write_json(GOLD / "forecast_evaluation.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
