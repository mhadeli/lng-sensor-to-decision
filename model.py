"""Fit three maintenance models; select on earlier months, test once on later months."""
import json
import platform

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.base import BaseEstimator
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

from common import ARTIFACTS, GOLD, MODELS, RAW, ROOT, SILVER, file_hash, read_parquet, write_json, write_parquet
from evaluation import FALSE_ALERT_BUDGET, evaluate, selection_key

VALID_START = pd.Timestamp("2015-09-01")
TEST_START = pd.Timestamp("2015-11-01")
SENSOR_MEANS = [f"{s}_mean" for s in ("volt", "rotate", "pressure", "vibration")]


class SensorRule(BaseEstimator):
    """Largest deviation from normal training sensor means; no learned failure model."""
    def fit(self, X, y):
        normal = X.loc[np.asarray(y) == 0, SENSOR_MEANS]
        self.center_ = normal.mean()
        self.scale_ = normal.std().replace(0, 1)
        return self

    def score_samples(self, X):
        return ((X[SENSOR_MEANS] - self.center_) / self.scale_).abs().max(axis=1).to_numpy()


def make_model(kind, setting):
    if kind == "baseline":
        return SensorRule()
    if kind == "mlp":
        return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16, 8), alpha=setting,
            max_iter=120, random_state=42, tol=0.0005))
    return XGBClassifier(n_estimators=140, max_depth=setting, learning_rate=0.05,
        min_child_weight=10, reg_lambda=10, n_jobs=2, tree_method="hist", random_state=42)


def fit_model(kind, setting, frame, columns):
    model = make_model(kind, setting)
    y = frame.failure_24h.astype(int)
    weights = compute_sample_weight("balanced", y)
    arguments = {} if kind == "baseline" else {
        "mlpclassifier__sample_weight" if kind == "mlp" else "sample_weight": weights}
    model.fit(frame[columns], y, **arguments)
    return model


def score_model(model, frame, columns):
    if isinstance(model, SensorRule):
        return model.score_samples(frame[columns])
    return model.predict_proba(frame[columns])[:, 1]


def split_data(frame):
    labeled = frame.dropna(subset=["failure_24h"])
    known = labeled.machineID % 5 != 0  # Deterministic held-out machines, never used to fit or tune.
    train = labeled[known & (labeled.datetime + pd.Timedelta(hours=24) < VALID_START)]
    valid = labeled[known & (labeled.datetime >= VALID_START) &
                    (labeled.datetime + pd.Timedelta(hours=24) < TEST_START)]
    refit = labeled[known & (labeled.datetime + pd.Timedelta(hours=24) < TEST_START)]
    test = labeled[labeled.datetime >= TEST_START].copy()
    test["cohort"] = np.where(test.machineID % 5 == 0, "unseen_machines", "known_machines")
    return train, valid, refit, test


def main():
    frame = read_parquet(SILVER / "features.parquet")
    failures = read_parquet(SILVER / "failures.parquet")
    columns = [c for c in frame if c not in ("machineID", "datetime", "failure_24h", "history_hours")]
    train, valid, refit, test = split_data(frame)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    report = {"target": "Any component failure in the next 24 hours", "features": columns,
              "train_rows": len(train), "validation_rows": len(valid), "refit_rows": len(refit),
              "valid_start": str(VALID_START), "test_start": str(TEST_START),
              "false_alert_budget": FALSE_ALERT_BUDGET, "settings": {}, "validation": {}, "test": {}}
    all_predictions, all_alerts, all_events, search = [], [], [], []
    with threadpool_limits(limits=2):
        for kind, settings in [("baseline", [0]), ("mlp", [1.0, 10.0]), ("xgboost", [2, 4])]:
            candidates = []
            for setting in settings:
                model = fit_model(kind, setting, train, columns)
                scored = valid[["machineID", "datetime", "failure_24h"]].copy()
                scored["score"] = score_model(model, valid, columns)
                thresholds = [1.5, 2, 2.5, 3, 4, 5] if kind == "baseline" else [0.1, 0.3, 0.5, 0.7, 0.9]
                for threshold in thresholds:
                    metrics, _, _ = evaluate(scored, failures, threshold)
                    candidates.append((selection_key(metrics), setting, threshold, metrics))
                    search.append({"model": kind, "setting": setting, "threshold": threshold, **metrics})
            _, setting, threshold, metrics = max(candidates, key=lambda x: x[0])
            report["settings"][kind] = {"setting": setting, "threshold": threshold}
            report["validation"][kind] = metrics
            model = fit_model(kind, setting, refit, columns)
            bundle = {"model": model, "kind": kind, "features": columns, "threshold": threshold,
                      "training_end": str(refit.datetime.max())}
            joblib.dump(bundle, ARTIFACTS / f"{kind}.joblib")
            scored = test[["machineID", "datetime", "failure_24h", "cohort"]].copy()
            scored["score"] = score_model(model, test, columns)
            scored["model"] = kind
            all_predictions.append(scored)
            report["test"][kind] = {}
            for cohort, group in scored.groupby("cohort"):
                metric, alerts, events = evaluate(group, failures, threshold)
                report["test"][kind][cohort] = metric
                all_alerts.append(alerts.assign(model=kind, cohort=cohort))
                all_events.append(events.assign(model=kind, cohort=cohort))
            print(f"Evaluated {kind}; validation event recall {metrics['event_recall']:.1%}", flush=True)
            if kind == "xgboost":
                pd.DataFrame({"feature": columns, "importance": model.feature_importances_}).sort_values(
                    "importance", ascending=False).to_csv(GOLD / "maintenance_importance.csv", index=False)
    report["selected"] = max(MODELS, key=lambda k: selection_key(report["validation"][k]))
    report["validation_budget_met"] = report["validation"][report["selected"]]["false_alerts_per_100_machine_days"] <= FALSE_ALERT_BUDGET
    write_parquet(pd.concat(all_predictions, ignore_index=True), GOLD / "maintenance_predictions.parquet")
    pd.concat(all_alerts, ignore_index=True).to_csv(GOLD / "maintenance_alerts.csv", index=False)
    pd.concat(all_events, ignore_index=True).to_csv(GOLD / "maintenance_events.csv", index=False)
    pd.DataFrame(search).to_csv(GOLD / "validation_search.csv", index=False)
    write_json(GOLD / "maintenance_evaluation.json", report)
    versions = {"python": platform.python_version(), "sklearn": sklearn.__version__, "xgboost": xgboost.__version__}
    source_hashes = {p: file_hash(ROOT / p) for p in ["model.py", "prepare.py", "evaluation.py", "sql/features.sql"]}
    write_json(ARTIFACTS / "registry.json", {"selected": report["selected"], "versions": versions,
        "source_hashes": source_hashes, "inputs": json.loads((RAW / "manifest.json").read_text()),
        "model_hashes": {kind: file_hash(ARTIFACTS / f"{kind}.joblib") for kind in MODELS},
        "settings": report["settings"], "test_start": str(TEST_START), "features": columns})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    from model import main
    main()
