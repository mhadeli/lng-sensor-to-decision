"""Classify process faults (Rule vs Random Forest vs Calibrated RF vs Deep MLP) and forecast next month's LNG exports."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import brier_score_loss, classification_report, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
# Several Gulf terminals were running by 2019. Earlier years are the start-up ramp.
SEASON_START = "2019-01"
TEST_MONTHS = 24

TAGS = (
    [
        "A feed", "D feed", "E feed", "A and C feed", "Recycle flow", "Reactor feed rate",
        "Reactor pressure", "Reactor level", "Reactor temperature", "Purge rate",
        "Separator temperature", "Separator level", "Separator pressure", "Separator underflow",
        "Stripper level", "Stripper pressure", "Stripper underflow", "Stripper temperature",
        "Stripper steam flow", "Compressor work", "Reactor cooling-water outlet temp",
        "Condenser cooling-water outlet temp",
    ]
    + [f"Reactor feed analysis {c}" for c in "ABCDEF"]
    + [f"Purge gas analysis {c}" for c in "ABCDEFGH"]
    + [f"Product analysis {c}" for c in "DEFGH"]
    + [
        "D feed valve", "E feed valve", "A feed valve", "A and C feed valve",
        "Compressor recycle valve", "Purge valve", "Separator liquid valve",
        "Stripper product valve", "Stripper steam valve", "Reactor cooling-water valve",
        "Condenser cooling-water valve",
    ]
)
TAG_NAMES = {f"x{i}": name for i, name in enumerate(TAGS, start=1)}


def add_changes(sensors):
    """Add each reading's change from the previous step and 5-step rolling mean of the same run."""
    measures = [column for column in sensors.columns if column.startswith("x")]
    change = sensors.groupby(["split", "fault_id"])[measures].diff().fillna(0)
    change.columns = [f"d{column}" for column in measures]
    roll_mean = sensors.groupby(["split", "fault_id"])[measures].transform(
        lambda s: s.rolling(5, min_periods=1).mean()
    )
    roll_mean.columns = [f"m5_{column}" for column in measures]
    basic_features = measures + list(change.columns)
    temporal_features = basic_features + list(roll_mean.columns)
    return pd.concat([sensors, change, roll_mean], axis=1), measures, basic_features, temporal_features


def calibrate_probability(prob_series, fault_ids, prior_weight=0.72, window=5):
    """Apply Bayesian prior adjustment and a 5-step (15-min) persistence window to raw fault probabilities."""
    smoothed = prob_series.groupby(fault_ids).transform(lambda s: s.rolling(window, min_periods=1).mean()).clip(0, 1)
    calibrated = (smoothed * prior_weight) / (smoothed * prior_weight + (1.0 - smoothed) + 1e-12)
    return calibrated.clip(0, 1)


def report(name, actual, predicted):
    print(name)
    print(classification_report(actual, predicted, target_names=["normal", "fault"], digits=3))


def classify_faults():
    """Train on the training runs. Score on the separate test runs, which the models never see."""
    sensors, measures, basic_features, temporal_features = add_changes(pd.read_csv(SILVER / "sensors.csv"))
    alerts = pd.read_csv(GOLD / "alerts.csv")
    train = sensors[sensors["split"] == "train"]
    test = sensors[sensors["split"] == "test"].copy()
    y_train = train["is_fault"]
    y_test = test["is_fault"]

    # 1. Random Forest (balanced class weights for the 20:1 fault-to-normal training ratio)
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=0, n_jobs=-1
    )
    rf.fit(train[basic_features], y_train)
    rf_prob = pd.Series(rf.predict_proba(test[basic_features])[:, 1], index=test.index)
    rf_pred = (rf_prob >= 0.5).astype(int)

    # 2. Probability-Calibrated Random Forest (Bayesian prior shift + 15-minute persistence filter)
    rf_cal_prob = calibrate_probability(rf_prob, test["fault_id"], prior_weight=0.72, window=5)
    rf_cal_pred = (rf_cal_prob >= 0.5).astype(int)

    # 3. Deep Learning: Multi-Layer Perceptron (MLP) trained on normal-standardized z-scores + temporal window
    normal_train = train[y_train == 0]
    scaler = StandardScaler().fit(normal_train[basic_features])
    X_train_scaled = scaler.transform(train[basic_features])
    X_test_scaled = scaler.transform(test[basic_features])

    idx_norm = np.where(y_train == 0)[0]
    idx_fault = np.where(y_train == 1)[0]
    rng = np.random.default_rng(0)
    idx_bal = np.concatenate([rng.choice(idx_norm, size=len(idx_fault), replace=True), idx_fault])

    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32), activation="relu", alpha=8.0, max_iter=100, random_state=0
    )
    mlp.fit(X_train_scaled[idx_bal], y_train.iloc[idx_bal])
    dl_raw_prob = pd.Series(mlp.predict_proba(X_test_scaled)[:, 1], index=test.index)
    dl_prob = dl_raw_prob.groupby(test["fault_id"]).transform(lambda s: s.rolling(5, min_periods=1).mean()).clip(0, 1)
    dl_pred = (dl_prob >= 0.5).astype(int)

    # Save predictions and probabilities
    result = test[["split", "fault_id", "step", "is_fault"]].copy()
    result["model_predicted"] = rf_pred
    result["rf_prob"] = rf_prob.round(4)
    result["calibrated_predicted"] = rf_cal_pred
    result["calibrated_prob"] = rf_cal_prob.round(4)
    result["dl_predicted"] = dl_pred
    result["dl_prob"] = dl_prob.round(4)
    result = result.merge(alerts.drop(columns="is_fault"), on=["split", "fault_id", "step"])
    result = result.rename(columns={"outside_limits": "rule_predicted"}).drop(columns="split")
    result.to_csv(GOLD / "fault_predictions.csv", index=False)

    # Save Random Forest feature importances aggregated by physical sensor tag
    imp_df = pd.DataFrame({
        "feature": basic_features,
        "sensor": [col.lstrip("d") for col in basic_features],
        "kind": ["step_change" if col.startswith("dx") else "level" for col in basic_features],
        "importance": rf.feature_importances_,
    })
    by_sensor = (
        imp_df.pivot(index="sensor", columns="kind", values="importance")
        .fillna(0.0)
        .assign(total_importance=lambda d: d["level"] + d["step_change"])
        .reset_index()
    )
    by_sensor["tag_name"] = by_sensor["sensor"].map(TAG_NAMES)
    by_sensor["category"] = np.where(
        by_sensor["sensor"].str.slice(1).astype(int) <= 22,
        "Process measurement (XMEAS)",
        np.where(by_sensor["sensor"].str.slice(1).astype(int) <= 41, "Composition analyzer", "Control valve (XMV)"),
    )
    by_sensor = by_sensor.sort_values("total_importance", ascending=False)
    by_sensor[["sensor", "tag_name", "category", "level", "step_change", "total_importance"]].to_csv(
        GOLD / "feature_importance.csv", index=False
    )

    # Save reliability / calibration curve data for dashboard
    cal_rows = []
    for name, col in [
        ("Uncalibrated Random Forest", "rf_prob"),
        ("Calibrated RF (Prior + 15m Persistence)", "calibrated_prob"),
        ("Deep Neural Network (MLP)", "dl_prob"),
    ]:
        frac_pos, mean_pred = calibration_curve(result["is_fault"], result[col], n_bins=8, strategy="uniform")
        for mp, fp in zip(mean_pred, frac_pos):
            cal_rows.append({"model": name, "mean_predicted_prob": round(mp, 4), "observed_fault_rate": round(fp, 4)})
    pd.DataFrame(cal_rows).to_csv(GOLD / "calibration_curve.csv", index=False)

    print(f"fault test rows: {len(result):,}")
    for label, pred_col, prob_col in [
        ("rule: any sensor outside 3-sigma band", "rule_predicted", None),
        ("random forest (uncalibrated)", "model_predicted", "rf_prob"),
        ("calibrated random forest (prior + 15m persistence)", "calibrated_predicted", "calibrated_prob"),
        ("deep neural network (MLP + 15m window)", "dl_predicted", "dl_prob"),
    ]:
        c1 = recall_score(result["is_fault"], result[pred_col])
        c0 = recall_score(result["is_fault"], result[pred_col], pos_label=0)
        extra = ""
        if prob_col:
            auc = roc_auc_score(result["is_fault"], result[prob_col])
            brier = brier_score_loss(result["is_fault"], result[prob_col])
            extra = f" | ROC-AUC: {auc:.3f} | Brier: {brier:.3f}"
        print(f"  {label:52s} -> Faults caught: {c1:.1%}, Normal cleared: {c0:.1%}{extra}")


def forecast_exports():
    """Forecast next month's exports using walk-forward validation (strictly earlier months only).

    Compares:
    - same_as_last_month: naive baseline
    - same_daily_rate: physics/calendar day-count scaling
    - calendar_month: daily rate times historical median seasonal factor
    - forecast: domain heuristic (50/50 blend of daily rate and seasonal factor)
    - rf_forecast: Random Forest Regressor (L1 loss) on seasonal & temperature features
    - dl_forecast: Deep Neural Network (MLP Regressor) on standardized seasonal, temperature & momentum features
    """
    exports = pd.read_csv(SILVER / "exports.csv")
    weather = pd.read_csv(SILVER / "weather.csv")
    exports = exports.merge(weather, on="month", how="left")
    exports["date"] = pd.to_datetime(exports["month"])
    exports["days"] = exports["date"].dt.days_in_month
    exports["cal_month"] = exports["date"].dt.month
    exports["daily_rate"] = exports["exports_mmcf"] / exports["days"]
    exports["change"] = exports["daily_rate"] / exports["daily_rate"].shift(1)
    exports["sin_m"] = np.sin(2 * np.pi * exports["cal_month"] / 12)
    exports["cos_m"] = np.cos(2 * np.pi * exports["cal_month"] / 12)
    exports["yoy_growth"] = (exports["daily_rate"].shift(1) / exports["daily_rate"].shift(13)).fillna(1.0)
    exports["mom_lag1"] = exports["change"].shift(1).fillna(1.0)

    rows = []
    feat_cols = ["seasonal_prior", "sin_m", "cos_m", "expected_temp", "yoy_growth", "mom_lag1"]

    for i in range(len(exports) - TEST_MONTHS, len(exports)):
        month, last = exports.iloc[i], exports.iloc[i - 1]
        history = exports.iloc[:i].copy()
        history = history[history["month"] >= SEASON_START].dropna(subset=["change"]).copy()

        med_by_month = history.groupby("cal_month")["change"].median().to_dict()
        temp_by_month = history.groupby("cal_month")["temp_c"].mean().to_dict()
        history["seasonal_prior"] = history["cal_month"].map(med_by_month)
        history["expected_temp"] = history["cal_month"].map(temp_by_month)

        typical_change = med_by_month[month["cal_month"]]
        same_daily_rate = last["daily_rate"] * month["days"]
        calendar_month = same_daily_rate * typical_change
        heuristic = (same_daily_rate + calendar_month) / 2

        x_te = pd.DataFrame([{
            "seasonal_prior": typical_change,
            "sin_m": month["sin_m"],
            "cos_m": month["cos_m"],
            "expected_temp": temp_by_month[month["cal_month"]],
            "yoy_growth": month["yoy_growth"],
            "mom_lag1": month["mom_lag1"],
        }])

        rf = RandomForestRegressor(
            n_estimators=100, max_depth=3, min_samples_leaf=4, criterion="absolute_error", random_state=0
        )
        rf.fit(history[feat_cols], history["change"])
        rf_change = rf.predict(x_te)[0]

        mlp = make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(16, 8), solver="lbfgs", alpha=10.0, max_iter=200, random_state=0),
        )
        mlp.fit(history[feat_cols], history["change"])
        dl_change = mlp.predict(x_te)[0]

        rows.append(
            {
                "month": month["month"],
                "actual": month["exports_mmcf"],
                "same_as_last_month": last["exports_mmcf"],
                "same_daily_rate": round(same_daily_rate),
                "calendar_month": round(calendar_month),
                "forecast": round(heuristic),
                "rf_forecast": round(same_daily_rate * rf_change),
                "dl_forecast": round(same_daily_rate * dl_change),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(GOLD / "volume_forecast.csv", index=False)

    methods = ["same_as_last_month", "same_daily_rate", "calendar_month", "rf_forecast", "forecast", "dl_forecast"]
    error = result[methods].sub(result["actual"], axis=0).abs().mean()
    print(f"\nexport forecast, {TEST_MONTHS} months to {result['month'].iloc[-1]}, mean absolute error in Bcf:")
    print((error / 1000).round(2).to_string())


def main():
    classify_faults()
    forecast_exports()


if __name__ == "__main__":
    main()

