"""Classify a process fault and forecast next month's LNG exports."""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parent
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
# Several Gulf terminals were running by 2019. Earlier years are the start-up ramp.
SEASON_START = "2019-01"
TEST_MONTHS = 24


def add_changes(sensors):
    """Add each reading's change from the previous step of the same run."""
    measures = [column for column in sensors.columns if column.startswith("x")]
    change = sensors.groupby(["split", "fault_id"])[measures].diff().fillna(0)
    change.columns = [f"d{column}" for column in measures]
    return pd.concat([sensors, change], axis=1), measures + list(change.columns)


def report(name, actual, predicted):
    print(name)
    print(classification_report(actual, predicted, target_names=["normal", "fault"], digits=3))


def classify_faults():
    """Train on the training runs. Score on the separate test runs, which the model never sees."""
    sensors, features = add_changes(pd.read_csv(SILVER / "sensors.csv"))
    alerts = pd.read_csv(GOLD / "alerts.csv")
    train = sensors[sensors["split"] == "train"]
    test = sensors[sensors["split"] == "test"]

    # Normal rows are about 1 in 20 of the training runs, so weight the classes.
    model = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight="balanced", random_state=0, n_jobs=-1
    )
    model.fit(train[features], train["is_fault"])

    result = test[["split", "fault_id", "step", "is_fault"]].copy()
    result["model_predicted"] = model.predict(test[features])
    result = result.merge(alerts.drop(columns="is_fault"), on=["split", "fault_id", "step"])
    result = result.rename(columns={"outside_limits": "rule_predicted"}).drop(columns="split")
    result.to_csv(GOLD / "fault_predictions.csv", index=False)

    print(f"fault test rows: {len(result):,}")
    report("rule: any sensor outside the normal band", result["is_fault"], result["rule_predicted"])
    report("random forest", result["is_fault"], result["model_predicted"])
    caught = result[result["is_fault"] == 1].groupby("fault_id")["model_predicted"].mean()
    print("share of fault rows caught, by fault")
    print(caught.round(2).to_string())


def forecast_exports():
    """Forecast next month's exports. Each month is scored using only the months before it.

    same_daily_rate: last month's daily rate times the days in the coming month.
    calendar_month: same_daily_rate times the typical change into that calendar month.
    forecast: the average of the two.
    """
    exports = pd.read_csv(SILVER / "exports.csv")
    exports["date"] = pd.to_datetime(exports["month"])
    exports["daily_rate"] = exports["exports_mmcf"] / exports["date"].dt.days_in_month
    exports["change"] = exports["daily_rate"] / exports["daily_rate"].shift(1)

    rows = []
    for i in range(len(exports) - TEST_MONTHS, len(exports)):
        month, last = exports.iloc[i], exports.iloc[i - 1]
        history = exports.iloc[:i]
        history = history[history["month"] >= SEASON_START]
        typical_change = history.loc[history["date"].dt.month == month["date"].month, "change"].median()
        same_daily_rate = last["daily_rate"] * month["date"].days_in_month
        calendar_month = same_daily_rate * typical_change
        rows.append(
            {
                "month": month["month"],
                "actual": month["exports_mmcf"],
                "same_as_last_month": last["exports_mmcf"],
                "same_daily_rate": round(same_daily_rate),
                "calendar_month": round(calendar_month),
                "forecast": round((same_daily_rate + calendar_month) / 2),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(GOLD / "volume_forecast.csv", index=False)

    methods = ["same_as_last_month", "same_daily_rate", "calendar_month", "forecast"]
    error = result[methods].sub(result["actual"], axis=0).abs().mean()
    print(f"export forecast, {TEST_MONTHS} months to {result['month'].iloc[-1]}, mean absolute error in Bcf")
    print((error / 1000).round(1).to_string())


def main():
    classify_faults()
    forecast_exports()


if __name__ == "__main__":
    main()
