"""Small known timelines verify event boundaries, confirmation delay, and burden."""
import unittest

import pandas as pd

from evaluation import operational_metrics


class EventEvaluation(unittest.TestCase):
    def evaluate(self, actual, flags, run_id=1):
        frame = pd.DataFrame({"fault_id": run_id, "step": range(len(actual)),
                              "is_fault": actual, "baseline_predicted": flags})
        return operational_metrics(frame, models=("baseline",))

    def test_confirmation_delay_and_false_episode_rate(self):
        report, events, _ = self.evaluate([0] * 6 + [1] * 6, [1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0])
        metric = report["models"]["baseline"]
        self.assertEqual(events.iloc[0].delay_minutes, 9)
        self.assertEqual(metric["events_within_30_min"], 1)
        self.assertEqual(metric["false_alert_episodes"], 1)
        self.assertAlmostEqual(metric["false_alerts_per_24_normal_hours"], 80)

    def test_carryover_alert_and_brief_flags_do_not_detect_event(self):
        report, events, _ = self.evaluate([0, 0, 0, 1, 1, 1, 1, 1], [1, 1, 1, 1, 0, 1, 1, 0])
        metric = report["models"]["baseline"]
        self.assertEqual(metric["missed_events"], 1)
        self.assertIsNone(metric["median_delay_minutes_detected_only"])
        self.assertIsNone(events.iloc[0].delay_minutes)
        self.assertEqual(metric["false_alert_episodes"], 1)

    def test_confirmation_after_event_ends_is_not_detection(self):
        report, _, _ = self.evaluate([0, 1, 1, 0, 0], [0, 1, 1, 1, 0])
        self.assertEqual(report["models"]["baseline"]["detected_events"], 0)

    def test_detection_window_boundary_and_missed_events(self):
        # At step 10, confirmation is exactly 30 minutes after onset.
        report, events, _ = self.evaluate([1] * 12 + [0] + [1] * 3, [0] * 8 + [1] * 4 + [0] * 4)
        metric = report["models"]["baseline"]
        self.assertEqual(metric["events"], 2)
        self.assertEqual(metric["events_within_30_min"], 1)
        self.assertEqual(metric["missed_events"], 1)
        self.assertEqual(metric["median_delay_minutes_detected_only"], 30)
        self.assertTrue(pd.isna(events.iloc[1].delay_minutes))
        late, _, _ = self.evaluate([1] * 15, [0] * 9 + [1] * 6)
        self.assertEqual(late["models"]["baseline"]["events_within_30_min"], 0)
        self.assertIsNone(late["models"]["baseline"]["false_alerts_per_24_normal_hours"])

    def test_normal_only_run_and_episode_boundaries(self):
        frame = pd.DataFrame({"fault_id": [0] * 4 + [1] * 4, "step": list(range(4)) * 2,
                              "is_fault": [0] * 8, "baseline_predicted": [0, 0, 1, 1, 1, 1, 0, 0]})
        report, events, _ = operational_metrics(frame, models=("baseline",))
        self.assertTrue(events.empty)
        self.assertEqual(report["models"]["baseline"]["false_alert_episodes"], 0)
        self.assertEqual(report["models"]["baseline"]["events"], 0)

    def test_missing_steps_rejected(self):
        frame = pd.DataFrame({"fault_id": [1, 1], "step": [0, 2], "is_fault": [0, 1],
                              "baseline_predicted": [0, 1]})
        with self.assertRaises(ValueError):
            operational_metrics(frame, models=("baseline",))


if __name__ == "__main__":
    unittest.main()
