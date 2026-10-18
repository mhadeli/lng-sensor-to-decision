"""Guard against future-data leakage and verify artifacts used by the dashboard."""
import json
import unittest

import duckdb
import numpy as np
import pandas as pd

from common import GOLD, SILVER, read_parquet
from forecasting import FEATURES, export_features, predict_month
from model import split_data
from prepare import add_labels, build_features, validate_table
from evaluation import evaluate, selection_key


def fixture_tables():
    times = pd.date_range("2015-01-01", periods=50, freq="h")
    telemetry = pd.DataFrame({"machineID": 1, "datetime": times,
                              **{s: np.arange(50) + 100.0 for s in ["volt", "rotate", "pressure", "vibration"]}})
    return {"telemetry": telemetry, "machines": pd.DataFrame({"machineID": [1], "model": ["model1"], "age": [5]}),
            "errors": pd.DataFrame({"machineID": [1, 1], "datetime": times[[10, 30]], "errorID": ["error1", "error2"]}),
            "maintenance": pd.DataFrame({"machineID": [1, 1], "datetime": times[[1, 30]], "comp": ["comp1", "comp2"]})}


def features_for(tables):
    with duckdb.connect() as con:
        for name, frame in tables.items():
            con.register(name, frame)
        return build_features(con)


class DataWorkflow(unittest.TestCase):
    def test_sql_features_exclude_current_and_future_values(self):
        tables = fixture_tables()
        before = features_for(tables)
        target = pd.Timestamp("2015-01-02")
        row = before[before.datetime == target].iloc[0]
        self.assertEqual(row.volt_mean, 111.5)
        self.assertEqual(row.error1_count, 1)
        self.assertEqual(row.error2_count, 0)
        self.assertAlmostEqual(row.comp1_days, 23 / 24)
        self.assertEqual(row.comp2_unknown, 1)
        tables["telemetry"].loc[tables["telemetry"].datetime >= target, "volt"] = 99999
        tables["maintenance"].loc[1, "datetime"] = target
        after = features_for(tables)
        pd.testing.assert_frame_equal(before[before.datetime <= target], after[after.datetime <= target])

    def test_missing_hour_suppresses_complete_history(self):
        tables = fixture_tables()
        tables["telemetry"] = tables["telemetry"].drop(index=10)
        result = features_for(tables)
        self.assertNotIn(pd.Timestamp("2015-01-02"), result.datetime.to_list())

    def test_labels_exclude_present_failure_and_unknown_tail(self):
        tables = fixture_tables()
        features = features_for(tables)
        failures = pd.DataFrame({"machineID": [1], "datetime": [pd.Timestamp("2015-01-02 06:00")]})
        labeled = add_labels(features, failures, tables["telemetry"])
        self.assertEqual(labeled.iloc[0].failure_24h, 1)
        self.assertTrue(pd.isna(labeled[labeled.datetime == failures.iloc[0].datetime].iloc[0].failure_24h))
        self.assertTrue(pd.isna(labeled.iloc[-1].failure_24h))

    def test_invalid_rows_are_quarantined(self):
        machines = pd.DataFrame({"machineID": [1, 1, 2], "model": ["model1"] * 3, "age": [4, 4, -2]})
        good, bad = validate_table(machines, "machines")
        self.assertEqual(len(good), 1)
        self.assertEqual(len(bad), 2)

    def test_time_purge_and_unseen_machine_split(self):
        times = pd.date_range("2015-08-30", "2015-11-03", freq="6h")
        frame = pd.concat([pd.DataFrame({"machineID": machine, "datetime": times, "failure_24h": 0}) for machine in [1, 5]])
        train, valid, refit, test = split_data(frame)
        self.assertLess(train.datetime.max() + pd.Timedelta(hours=24), valid.datetime.min())
        self.assertLess(refit.datetime.max() + pd.Timedelta(hours=24), test.datetime.min())
        self.assertEqual(set(refit.machineID), {1})
        self.assertEqual(set(test[test.cohort == "unseen_machines"].machineID), {5})

    def test_forecast_respects_publication_lag_in_features_and_training(self):
        exports = pd.DataFrame({"month": pd.period_range("2017-01", periods=100, freq="M").astype(str),
                                "exports_mmcf": 100000 + np.arange(100) * 500.0})
        before = export_features(exports)
        exports.loc[78:, "exports_mmcf"] *= 10  # At month 80, only month 77 and earlier are known.
        after = export_features(exports)
        pd.testing.assert_frame_equal(before.loc[:80, FEATURES + ["baseline"]], after.loc[:80, FEATURES + ["baseline"]])
        self.assertEqual(predict_month(before, 80, "xgboost", 40), predict_month(after, 80, "xgboost", 40))


@unittest.skipUnless((GOLD / "maintenance_evaluation.json").exists(), "Build pipeline for artifact checks")
class ArtifactChecks(unittest.TestCase):
    def test_validation_selection_and_saved_metrics(self):
        report = json.loads((GOLD / "maintenance_evaluation.json").read_text())
        self.assertEqual(report["selected"], max(report["validation"], key=lambda k: selection_key(report["validation"][k])))
        predictions = read_parquet(GOLD / "maintenance_predictions.parquet")
        failures = read_parquet(SILVER / "failures.parquet")
        for (model, cohort), group in predictions.groupby(["model", "cohort"]):
            metrics, _, _ = evaluate(group, failures, report["settings"][model]["threshold"])
            self.assertEqual(metrics, report["test"][model][cohort])

    def test_replay_scores_match_batch_scores(self):
        replay = pd.read_csv(GOLD / "replay_scores.csv", parse_dates=["datetime"])
        selected = json.loads((GOLD / "maintenance_evaluation.json").read_text())["selected"]
        batch = read_parquet(GOLD / "maintenance_predictions.parquet")
        joint = replay.merge(batch[batch.model == selected], on=["machineID", "datetime"], suffixes=("_replay", "_batch"))
        self.assertGreater(len(joint), 0)
        np.testing.assert_allclose(joint.score_replay, joint.score_batch, rtol=1e-6)
        alerts = replay[replay.review_requested].sort_values(["machineID", "datetime"])
        self.assertTrue((alerts.groupby("machineID").datetime.diff().dropna() >= pd.Timedelta(hours=24)).all())

    def test_terminal_metrics_and_availability(self):
        report = json.loads((GOLD / "forecast_evaluation.json").read_text())
        forecasts = pd.read_csv(GOLD / "terminal_forecast.csv")
        self.assertEqual(report["selected"], min(report["validation_mae_bcf"], key=report["validation_mae_bcf"].get))
        for kind in ["baseline", "mlp", "xgboost"]:
            self.assertAlmostEqual((forecasts[kind] - forecasts.actual).abs().mean() / 1000, report["test"][kind]["mae_bcf"])
        self.assertTrue((pd.PeriodIndex(forecasts.month, freq="M") - 3 == pd.PeriodIndex(forecasts.latest_available_month, freq="M")).all())
