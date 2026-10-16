"""Three models per question; settings chosen before the final holdout is scored."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import balanced_accuracy_score, recall_score, roc_auc_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parent
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
TEST_MONTHS = 24
VALID_MONTHS = 12

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
    """Causal features; windows reset at each independent run."""
    sensors = sensors.sort_values(["split", "fault_id", "step"]).reset_index(drop=True)
    measures = [c for c in sensors if c.startswith("x")]
    grouped = sensors.groupby(["split", "fault_id"])[measures]
    changes = grouped.diff().fillna(0).add_prefix("d")
    means = grouped.transform(lambda x: x.rolling(5, min_periods=1).mean()).add_prefix("m5_")
    features = measures + list(changes) + list(means)
    return pd.concat([sensors, changes, means], axis=1), measures, features


def smooth(probabilities, frame):
    return pd.Series(probabilities, index=frame.index).groupby(
        [frame["split"], frame["fault_id"]]
    ).transform(lambda x: x.rolling(5, min_periods=1).mean())


class NormalReference(TransformerMixin, BaseEstimator):
    """Express each signal relative to normal operation, including deviation magnitude."""
    def fit(self, X, y):
        self.scaler_ = StandardScaler().fit(X[np.asarray(y) == 0])
        return self

    def transform(self, X):
        scaled = self.scaler_.transform(X)
        return np.concatenate([scaled, np.abs(scaled)], axis=1)


def fault_model(kind, setting):
    if kind == "mlp":
        return make_pipeline(NormalReference(), MLPClassifier(
            hidden_layer_sizes=(64, 32), alpha=setting, max_iter=250,
            random_state=42, tol=1e-4))
    return make_pipeline(NormalReference(), XGBClassifier(n_estimators=220, max_depth=setting, learning_rate=0.04,
                         min_child_weight=5, reg_lambda=10, subsample=0.85,
                         colsample_bytree=0.85, tree_method="hist", n_jobs=2,
                         random_state=42))


def fit_fault(model, frame, features):
    weights = compute_sample_weight("balanced", frame["is_fault"])
    kwargs = {f"{model.steps[-1][0]}__sample_weight": weights}
    model.fit(frame[features], frame["is_fault"], **kwargs)
    return model


def fault_metrics(actual, prediction, probability=None):
    return {"caught": float(recall_score(actual, prediction)),
            "false_alarm": float(1 - recall_score(actual, prediction, pos_label=0)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
            "auc": float(roc_auc_score(actual, probability)) if probability is not None else None}


def classify_faults():
    sensors, measures, features = add_changes(pd.read_csv(SILVER / "sensors.csv"))
    train = sensors[sensors.split == "train"].copy()
    test = sensors[sensors.split == "test"].copy()
    cutoff = train.groupby("fault_id")["step"].transform("max").add(1).mul(0.75).astype(int)
    fit = train[train.step < cutoff - 5]
    valid = train[train.step >= cutoff]
    # A five-reading purge separates fitting and validation windows.
    normal = train[train.is_fault == 0][measures]
    baseline = ((test[measures] - normal.mean()).abs() > 3 * normal.std()).any(axis=1).astype(int)
    normal_fit = fit[fit.is_fault == 0][measures]
    baseline_valid = ((valid[measures] - normal_fit.mean()).abs() > 3 * normal_fit.std()).any(axis=1).astype(int)
    result = test[["fault_id", "step", "is_fault"]].copy()
    result["baseline_predicted"] = baseline
    validation = {"baseline": fault_metrics(valid.is_fault, baseline_valid)}
    selection = {}
    for kind, settings in [("mlp", [1.0, 8.0, 30.0]), ("xgboost", [2, 4, 6])]:
        candidates = []
        for setting in settings:
            model = fit_fault(fault_model(kind, setting), fit, features)
            probability = smooth(model.predict_proba(valid[features])[:, 1], valid)
            for threshold in np.arange(0.2, 0.81, 0.05):
                metrics = fault_metrics(valid.is_fault, probability >= threshold, probability)
                candidates.append((metrics["balanced_accuracy"], setting, float(threshold), metrics))
        _, setting, threshold, metrics = max(candidates, key=lambda x: x[0])
        validation[kind] = metrics
        selection[kind] = {"setting": setting, "threshold": threshold}
        model = fit_fault(fault_model(kind, setting), train, features)
        probability = smooth(model.predict_proba(test[features])[:, 1], test)
        result[f"{kind}_score"] = probability
        result[f"{kind}_predicted"] = (probability >= threshold).astype(int)
        if kind == "xgboost":
            importance = pd.DataFrame({"sensor": [c.replace("m5_", "").removeprefix("d") for c in features * 2],
                                       "importance": model.steps[-1][1].feature_importances_})
            importance = importance.groupby("sensor", as_index=False).importance.sum()
            importance["tag_name"] = importance.sensor.map(TAG_NAMES)
            importance.sort_values("importance", ascending=False).to_csv(GOLD / "feature_importance.csv", index=False)
    result.to_csv(GOLD / "fault_predictions.csv", index=False)
    metrics = {kind: fault_metrics(result.is_fault, result[f"{kind}_predicted"],
                                  result[f"{kind}_score"] if kind != "baseline" else None)
               for kind in ["baseline", "mlp", "xgboost"]}
    selected = max(validation, key=lambda k: validation[k]["balanced_accuracy"])
    return {"selected": selected, "settings": selection, "validation": validation, "test": metrics,
            "test_rows": len(test), "train_rows": len(train), "fit_rows": len(fit), "validation_rows": len(valid)}


def export_features(exports):
    exports = exports.sort_values("month").copy()
    date = pd.to_datetime(exports.month)
    if exports.month.duplicated().any() or not np.array_equal(date.dt.to_period("M").astype(str),
            pd.period_range(date.min(), date.max(), freq="M").astype(str)):
        raise ValueError("Exports must have one row per consecutive month")
    days = date.dt.days_in_month
    rate = exports.exports_mmcf / days
    change = rate.pct_change()
    exports["baseline"] = rate.shift(1) * days
    exports["target"] = change
    exports["sin_month"] = np.sin(2 * np.pi * date.dt.month / 12)
    exports["cos_month"] = np.cos(2 * np.pi * date.dt.month / 12)
    exports["last_change"] = change.shift(1)
    exports["recent_change"] = change.shift(1).rolling(3).mean()
    exports["year_change"] = rate.shift(1) / rate.shift(13) - 1
    # Each row's seasonal estimate uses only earlier years, including during fitting.
    exports["seasonal_change"] = change.groupby(date.dt.month).transform(
        lambda x: x.shift(1).expanding(min_periods=2).median())
    return exports, ["sin_month", "cos_month", "last_change", "recent_change", "year_change", "seasonal_change"]


def export_model(kind, setting):
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(8,),
            solver="lbfgs", alpha=setting, max_iter=1000, random_state=42))
    return XGBRegressor(n_estimators=setting, max_depth=2, learning_rate=0.025,
        min_child_weight=6, reg_lambda=20, base_score=0.0, n_jobs=2, random_state=42)


def predict_export(frame, features, i, kind, setting):
    history = frame.iloc[:i].loc[lambda x: x.month >= "2019-01"].dropna(subset=features + ["target"])
    model = export_model(kind, setting)
    model.fit(history[features], history.target)
    change = model.predict(frame.iloc[[i]][features])[0]
    return max(0.0, float(frame.iloc[i].baseline * (1 + change)))


def forecast_exports():
    frame, features = export_features(pd.read_csv(SILVER / "exports.csv"))
    start = len(frame) - TEST_MONTHS
    if start - VALID_MONTHS < 36:
        raise ValueError("At least 72 monthly observations are required")
    validation = {"baseline": float((frame.iloc[start-VALID_MONTHS:start].baseline -
                                    frame.iloc[start-VALID_MONTHS:start].exports_mmcf).abs().mean() / 1000)}
    settings = {}
    for kind, candidates in [("mlp", [1.0, 10.0, 100.0]), ("xgboost", [40, 100, 200])]:
        errors = {}
        for setting in candidates:
            predictions = [predict_export(frame, features, i, kind, setting) for i in range(start-VALID_MONTHS, start)]
            errors[setting] = float(np.mean(np.abs(np.array(predictions) - frame.iloc[start-VALID_MONTHS:start].exports_mmcf)) / 1000)
        settings[kind] = min(errors, key=errors.get)
        validation[kind] = errors[settings[kind]]
    selected = min(validation, key=validation.get)
    result = frame.iloc[start:][["month", "exports_mmcf", "baseline"]].rename(columns={"exports_mmcf": "actual"}).copy()
    for kind, setting in settings.items():
        result[kind] = [predict_export(frame, features, i, kind, setting) for i in range(start, len(frame))]
    result.to_csv(GOLD / "volume_forecast.csv", index=False)
    metrics = {kind: {"mae_bcf": float((result[kind] - result.actual).abs().mean() / 1000),
                      "mape": float(((result[kind] - result.actual).abs() / result.actual).mean())}
               for kind in ["baseline", "mlp", "xgboost"]}
    return {"selected": selected, "settings": settings, "validation_mae_bcf": validation, "test": metrics,
            "validation_first": frame.iloc[start-VALID_MONTHS].month, "validation_last": frame.iloc[start-1].month,
            "test_first": result.month.iloc[0], "test_last": result.month.iloc[-1], "test_months": len(result)}


def main():
    GOLD.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=2):
        results = {"faults": classify_faults(), "exports": forecast_exports()}
    (GOLD / "evaluation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
