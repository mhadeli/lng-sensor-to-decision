"""Guard against temporal leakage and misleading evaluation artifacts."""
import json
import unittest

import numpy as np
import pandas as pd

from model import GOLD, add_changes, export_features, fault_metrics, predict_export


class TemporalFeatures(unittest.TestCase):
    def test_sensor_windows_reset_and_ignore_future(self):
        raw = pd.DataFrame({"split": ["train"] * 8, "fault_id": [0] * 4 + [1] * 4,
                            "step": list(range(4)) * 2, "x1": [1, 2, 3, 4, 100, 101, 102, 103]})
        before, _, _ = add_changes(raw)
        self.assertEqual(before.loc[4, "dx1"], 0)
        self.assertEqual(before.loc[4, "m5_x1"], 100)
        raw.loc[3, "x1"] = 999
        after, _, _ = add_changes(raw)
        pd.testing.assert_frame_equal(before.iloc[:3], after.iloc[:3])
        pd.testing.assert_frame_equal(before.iloc[4:], after.iloc[4:])

    def test_forecast_features_do_not_see_current_or_future_target(self):
        raw = pd.DataFrame({"month": pd.period_range("2015-01", periods=120, freq="M").astype(str),
                            "exports_mmcf": 100000 + np.arange(120) * 1200.0})
        before, columns = export_features(raw)
        raw.loc[85:, "exports_mmcf"] *= 3
        after, _ = export_features(raw)
        pd.testing.assert_frame_equal(before.loc[:85, columns + ["baseline"]],
                                      after.loc[:85, columns + ["baseline"]])
        a = predict_export(before, columns, 85, "xgboost", 40)
        b = predict_export(after, columns, 85, "xgboost", 40)
        self.assertEqual(a, b)

    def test_missing_month_rejected(self):
        raw = pd.DataFrame({"month": ["2020-01", "2020-03"], "exports_mmcf": [10, 20]})
        with self.assertRaises(ValueError):
            export_features(raw)


@unittest.skipUnless((GOLD / "evaluation.json").exists(), "Run model.py for artifact checks")
class EvaluationArtifacts(unittest.TestCase):
    def test_headline_model_chosen_from_validation(self):
        result = json.loads((GOLD / "evaluation.json").read_text())
        self.assertEqual(result["faults"]["selected"], max(result["faults"]["validation"],
            key=lambda k: result["faults"]["validation"][k]["balanced_accuracy"]))
        self.assertEqual(result["exports"]["selected"], min(result["exports"]["validation_mae_bcf"],
            key=result["exports"]["validation_mae_bcf"].get))
        self.assertLess(result["exports"]["validation_last"], result["exports"]["test_first"])

    def test_metrics_match_predictions(self):
        result = json.loads((GOLD / "evaluation.json").read_text())
        faults = pd.read_csv(GOLD / "fault_predictions.csv")
        exports = pd.read_csv(GOLD / "volume_forecast.csv")
        self.assertFalse(faults.duplicated(["fault_id", "step"]).any())
        self.assertEqual(len(exports), 24)
        for kind in ["baseline", "mlp", "xgboost"]:
            metric = fault_metrics(faults.is_fault, faults[f"{kind}_predicted"])
            for key in ["caught", "false_alarm", "balanced_accuracy"]:
                self.assertAlmostEqual(metric[key], result["faults"]["test"][kind][key])
            self.assertTrue(np.isfinite(exports[kind]).all())
            self.assertAlmostEqual((exports[kind] - exports.actual).abs().mean() / 1000,
                                   result["exports"]["test"][kind]["mae_bcf"])


if __name__ == "__main__":
    unittest.main()
