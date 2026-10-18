"""Translate six-hour failure scores into a bounded maintenance review workload."""
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score

HORIZON_HOURS = 24
REVIEW_COOLDOWN_HOURS = 24
FALSE_ALERT_BUDGET = 2.0  # Illustrative target per 100 observed machine-days, set before testing.


def review_alerts(predictions, threshold):
    """At most one emitted review per machine per 24 hours, using scores only."""
    emitted = []
    for _, group in predictions.groupby("machineID"):
        last = None
        for row in group.sort_values("datetime").itertuples():
            if row.score >= threshold and (last is None or row.datetime - last >= pd.Timedelta(hours=24)):
                emitted.append({"machineID": row.machineID, "datetime": row.datetime, "score": row.score})
                last = row.datetime
    return pd.DataFrame(emitted, columns=["machineID", "datetime", "score"])


def evaluate(predictions, failures, threshold):
    alerts = review_alerts(predictions, threshold)
    incidents = failures[["machineID", "datetime"]].drop_duplicates()
    alerts["followed_by_failure"] = False
    events = []
    for machine, group in predictions.groupby("machineID"):
        times = group.datetime.sort_values()
        fail_times = incidents.loc[incidents.machineID == machine, "datetime"].sort_values()
        machine_alerts = alerts[alerts.machineID == machine]
        for row in machine_alerts.itertuples():
            hit = ((fail_times > row.datetime) & (fail_times <= row.datetime + pd.Timedelta(hours=24))).any()
            alerts.loc[row.Index, "followed_by_failure"] = bool(hit)
        for failure in fail_times:
            opportunities = times[(times >= failure - pd.Timedelta(hours=24)) & (times < failure)]
            # Four six-hour decisions ensure a complete lookback; boundary incidents are excluded.
            if len(opportunities) != 4:
                continue
            warning = machine_alerts[(machine_alerts.datetime >= failure - pd.Timedelta(hours=24)) &
                                     (machine_alerts.datetime < failure)]
            lead = float((failure - warning.datetime.min()).total_seconds() / 3600) if len(warning) else None
            events.append({"machineID": int(machine), "failure_time": failure,
                           "warned": bool(len(warning)), "lead_hours": lead})
    events = pd.DataFrame(events, columns=["machineID", "failure_time", "warned", "lead_hours"])
    warned = int(events.warned.sum())
    false_count = int((~alerts.followed_by_failure).sum())
    machine_days = len(predictions) * 6 / 24
    actual = predictions.failure_24h.astype(int)
    metrics = {"rows": len(predictions), "machines": int(predictions.machineID.nunique()),
               "events": len(events), "warned_events": warned, "missed_events": len(events) - warned,
               "event_recall": warned / len(events) if len(events) else 0.0,
               "median_lead_hours": float(events.lead_hours.dropna().median()) if warned else None,
               "alerts": len(alerts), "false_alerts": false_count,
               "machine_days": machine_days,
               "false_alerts_per_100_machine_days": false_count * 100 / machine_days if machine_days else 0.0,
               "average_precision": float(average_precision_score(actual, predictions.score)) if actual.sum() else 0.0,
               "reading_recall": float(recall_score(actual, predictions.score >= threshold, zero_division=0)),
               "reading_precision": float(precision_score(actual, predictions.score >= threshold, zero_division=0))}
    return metrics, alerts, events


def selection_key(metrics):
    within_budget = metrics["false_alerts_per_100_machine_days"] <= FALSE_ALERT_BUDGET
    if within_budget:
        return (1, metrics["event_recall"], -metrics["false_alerts_per_100_machine_days"])
    return (0, -metrics["false_alerts_per_100_machine_days"], metrics["event_recall"])
