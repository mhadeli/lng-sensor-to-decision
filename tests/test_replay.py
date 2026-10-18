import unittest

import numpy as np
import pandas as pd

from replay import score_snapshot
from test_workflow import fixture_tables


class ConstantModel:
    def predict_proba(self, X):
        return np.tile([.2, .8], (len(X), 1))


class ReplayChecks(unittest.TestCase):
    def test_future_inputs_do_not_change_snapshot(self):
        tables = fixture_tables()
        bundle = {"model": ConstantModel(), "features": ["volt_mean"], "threshold": .5}
        timestamp = pd.Timestamp("2015-01-02")
        before, features = score_snapshot(tables, timestamp, bundle)
        tables["telemetry"].loc[tables["telemetry"].datetime > timestamp, "volt"] = -999
        after, changed = score_snapshot(tables, timestamp, bundle)
        pd.testing.assert_frame_equal(before, after)
        pd.testing.assert_frame_equal(features, changed)
        self.assertTrue(before.iloc[0].above_threshold)

    def test_stale_sensor_history_suppresses_model_output(self):
        tables = fixture_tables()
        tables["telemetry"] = tables["telemetry"].iloc[:24]
        result, _ = score_snapshot(tables, pd.Timestamp("2015-01-02"),
                                   {"model": ConstantModel(), "features": ["volt_mean"], "threshold": .5})
        self.assertTrue(pd.isna(result.iloc[0].score))
        self.assertFalse(result.iloc[0].above_threshold)
        self.assertIn("Suppressed", result.iloc[0].status)
