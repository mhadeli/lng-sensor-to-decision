"""Evaluate fixed alert policies on saved predictions, without retraining or tuning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ("baseline", "mlp", "xgboost")
STEP_MINUTES = 3
PERSISTENCE = 3
DETECTION_WINDOW_MINUTES = 30


def operational_metrics(predictions, models=MODELS):
    """A new alert needs three consecutive flags; confirm at the third reading.

    An event is a contiguous fault-labelled interval in one run. Only alerts
    starting within that event count as detections. Existing pre-fault alerts
    cannot claim a detection. All runs must have consecutive integer steps.
    """
    required = ["fault_id", "step", "is_fault"] + [f"{m}_predicted" for m in models]
    if predictions.empty or predictions[required].isna().any().any():
        raise ValueError("Complete, nonempty predictions are required")
    if predictions.duplicated(["fault_id", "step"]).any():
        raise ValueError("Duplicate run steps")
    if not predictions[["is_fault"] + required[3:]].isin([0, 1]).all().all():
        raise ValueError("Labels and predictions must be binary")
    event_rows, run_rows = [], []
    for run_id, run in predictions.groupby("fault_id", sort=True):
        run = run.sort_values("step").reset_index(drop=True)
        steps = run.step.to_numpy()
        if not np.equal(steps, steps.astype(int)).all() or not np.equal(np.diff(steps), 1).all():
            raise ValueError("Run steps must be consecutive integers")
        actual = run.is_fault.to_numpy().astype(bool)
        starts = np.flatnonzero(actual & ~np.r_[False, actual[:-1]])
        ends = np.flatnonzero(actual & ~np.r_[actual[1:], False])
        normal_hours = float((~actual).sum() * STEP_MINUTES / 60)
        for model in models:
            flags = run[f"{model}_predicted"].to_numpy().astype(bool)
            alarm_starts = np.flatnonzero(flags & ~np.r_[False, flags[:-1]])
            alarm_ends = np.flatnonzero(flags & ~np.r_[flags[1:], False])
            episodes = [(int(a), int(a + PERSISTENCE - 1))
                        for a, b in zip(alarm_starts, alarm_ends) if b - a + 1 >= PERSISTENCE]
            false_episodes = sum(not actual[a:c + 1].any() for a, c in episodes)
            run_rows.append({"model": model, "fault_id": int(run_id),
                             "normal_hours": normal_hours, "false_alert_episodes": int(false_episodes)})
            for number, (start, end) in enumerate(zip(starts, ends), start=1):
                confirmations = [c for a, c in episodes if a >= start and c <= end]
                delay = int((steps[min(confirmations)] - steps[start]) * STEP_MINUTES) if confirmations else None
                event_rows.append({"model": model, "fault_id": int(run_id), "event": number,
                                   "start_step": int(steps[start]), "end_step": int(steps[end]),
                                   "detected": delay is not None,
                                   "detected_within_30_min": delay is not None and delay <= DETECTION_WINDOW_MINUTES,
                                   "delay_minutes": delay})
    events = pd.DataFrame(event_rows, columns=["model", "fault_id", "event", "start_step", "end_step",
                                              "detected", "detected_within_30_min", "delay_minutes"])
    runs = pd.DataFrame(run_rows)
    summary = {}
    for model in models:
        subset, run_subset = events[events.model == model], runs[runs.model == model]
        detected = subset[subset.detected.astype(bool)]
        normal_hours = float(run_subset.normal_hours.sum())
        count = int(run_subset.false_alert_episodes.sum())
        summary[model] = {"events": len(subset), "detected_events": len(detected),
                          "events_within_30_min": int(subset.detected_within_30_min.sum()),
                          "missed_events": len(subset) - len(detected),
                          "median_delay_minutes_detected_only": float(detected.delay_minutes.median()) if len(detected) else None,
                          "normal_hours": normal_hours, "false_alert_episodes": count,
                          "false_alerts_per_24_normal_hours": count * 24 / normal_hours if normal_hours else None}
    return {"step_minutes": STEP_MINUTES, "consecutive_flags": PERSISTENCE,
            "detection_window_minutes": DETECTION_WINDOW_MINUTES, "models": summary}, events, runs


def write_operational_evaluation(predictions, destination):
    summary, events, runs = operational_metrics(predictions)
    destination.mkdir(parents=True, exist_ok=True)
    events.to_csv(destination / "fault_events.csv", index=False)
    runs.to_csv(destination / "fault_run_alerts.csv", index=False)
    (destination / "fault_operations.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


if __name__ == "__main__":
    gold = ROOT / "data" / "gold"
    print(json.dumps(write_operational_evaluation(pd.read_csv(gold / "fault_predictions.csv"), gold), indent=2))
