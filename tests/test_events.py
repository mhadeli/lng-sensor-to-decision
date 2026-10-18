"""Hand-built timelines verify warning lead time, future labels, and review workload."""
import unittest

import pandas as pd

from evaluation import evaluate, review_alerts, selection_key


class AlertEvaluation(unittest.TestCase):
    def frame(self, scores):
        return pd.DataFrame({"machineID": 1, "datetime": pd.date_range("2015-11-01", periods=len(scores), freq="6h"),
                             "failure_24h": [0] * len(scores), "score": scores})

    def test_cooldown_is_machine_local_and_never_reads_labels(self):
        frame = self.frame([1] * 9)
        alerts = review_alerts(frame, .5)
        self.assertEqual(list(alerts.datetime), list(frame.datetime.iloc[[0, 4, 8]]))
        frame.failure_24h = 1
        pd.testing.assert_frame_equal(alerts, review_alerts(frame, .5))
        combined = pd.concat([frame, frame.assign(machineID=2)])
        self.assertEqual(len(review_alerts(combined, .5)), 6)

    def test_future_warning_and_false_alert(self):
        frame = self.frame([1, 0, 0, 0, 1, 0, 0, 0])
        frame.loc[:3, "failure_24h"] = 1
        failures = pd.DataFrame({"machineID": [1], "datetime": [pd.Timestamp("2015-11-02")]})
        metric, alerts, events = evaluate(frame, failures, .5)
        self.assertEqual(metric["warned_events"], 1)
        self.assertEqual(metric["median_lead_hours"], 24)
        self.assertEqual(metric["false_alerts"], 1)  # Alert at the failure instant is not advance warning.
        self.assertEqual(metric["false_alerts_per_100_machine_days"], 50)
        self.assertTrue(events.iloc[0].warned)

    def test_incomplete_event_window_excluded_and_miss_has_no_lead_time(self):
        frame = self.frame([0] * 8)
        failures = pd.DataFrame({"machineID": [1, 1],
                                "datetime": pd.to_datetime(["2015-11-01 06:00", "2015-11-02 00:00"])})
        metric, _, events = evaluate(frame, failures, .5)
        self.assertEqual(metric["events"], 1)
        self.assertEqual(metric["missed_events"], 1)
        self.assertIsNone(metric["median_lead_hours"])

    def test_budget_takes_priority_over_recall(self):
        affordable = {"false_alerts_per_100_machine_days": 1.5, "event_recall": .7}
        noisy = {"false_alerts_per_100_machine_days": 5, "event_recall": 1}
        self.assertGreater(selection_key(affordable), selection_key(noisy))
