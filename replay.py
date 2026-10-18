"""Replay a week of historical telemetry through a saved model, without failure labels."""
import argparse
import json

import duckdb
import joblib
import numpy as np
import pandas as pd

from common import ARTIFACTS, DATABASE, GOLD, SILVER, file_hash, read_parquet, write_json
from model import score_model, split_data
from prepare import build_features


def score_snapshot(tables, timestamp, bundle):
    """Filter to the prediction cutoff before SQL feature construction; missing history suppresses scores."""
    timestamp = pd.Timestamp(timestamp)
    with duckdb.connect() as con:
        con.execute("SET threads=2")
        for name in ("machines", "telemetry", "errors", "maintenance"):
            source = tables[name]
            if "datetime" in source:
                # A current telemetry row anchors the decision; its values are excluded by the SQL window.
                source = source[source.datetime <= timestamp] if name == "telemetry" else source[source.datetime < timestamp]
                if name != "maintenance":
                    source = source[source.datetime >= timestamp - pd.Timedelta(hours=24)]
            con.register("batch_source", source)
            con.execute(f"CREATE TABLE {name} AS SELECT * FROM batch_source")
        features = build_features(con)
    features = features[features.datetime == timestamp]
    result = tables["machines"][["machineID"]].copy()
    result["datetime"] = timestamp
    values = features[["machineID"]].copy()
    values["score"] = score_model(bundle["model"], features, bundle["features"]) if len(features) else []
    result = result.merge(values, on="machineID", how="left", validate="one_to_one")
    result["status"] = np.where(result.score.isna(), "Suppressed: incomplete or stale history", "Scored")
    result["above_threshold"] = result.score >= bundle["threshold"]
    return result, features


def main(start=None, days=7):
    if days < 1 or days > 31:
        raise ValueError("Replay days must be between 1 and 31")
    registry = json.loads((ARTIFACTS / "registry.json").read_text())
    kind = registry["selected"]
    path = ARTIFACTS / f"{kind}.joblib"
    if file_hash(path) != registry["model_hashes"][kind]:
        raise ValueError("Saved model checksum does not match the local registry")
    bundle = joblib.load(path)  # Only load artifacts generated locally by model.py.
    feature_history = read_parquet(SILVER / "features.parquet")
    _, _, reference, _ = split_data(feature_history)
    centers = reference[bundle["features"]].mean()
    scales = reference[bundle["features"]].std().replace(0, 1)
    last_labeled = feature_history.loc[feature_history.failure_24h.notna(), "datetime"].max()
    start = pd.Timestamp(start) if start else last_labeled.normalize() - pd.Timedelta(days=days)
    if start < pd.Timestamp(registry["test_start"]) or start + pd.Timedelta(days=days) > last_labeled:
        raise ValueError("Replay must stay inside the complete held-out observation period")
    all_scores, monitor = [], []
    last_review = {}
    with duckdb.connect(str(DATABASE), read_only=True) as warehouse:
        for timestamp in pd.date_range(start, periods=days * 4, freq="6h"):
            tables = {"machines": warehouse.execute("SELECT * FROM machines").df()}
            for name in ("telemetry", "errors", "maintenance"):
                if name == "maintenance":
                    tables[name] = warehouse.execute("SELECT * FROM maintenance WHERE datetime < ?", [timestamp]).df()
                else:
                    tables[name] = warehouse.execute(f"SELECT * FROM {name} WHERE datetime BETWEEN ? AND ?",
                        [timestamp - pd.Timedelta(hours=24), timestamp]).df()
            scores, features = score_snapshot(tables, timestamp, bundle)
            scores["review_requested"] = False
            for row in scores[scores.above_threshold].itertuples():
                last = last_review.get(row.machineID)
                if last is None or timestamp - last >= pd.Timedelta(hours=24):
                    scores.loc[row.Index, "review_requested"] = True
                    last_review[row.machineID] = timestamp
            shifted = ((features[bundle["features"]].mean() - centers) / scales).abs() > 3
            monitor.append({"datetime": str(timestamp), "scored": int(scores.score.notna().sum()),
                            "suppressed": int(scores.score.isna().sum()),
                            "reviews": int(scores.review_requested.sum()),
                            "features_shifted_over_3sd": int(shifted.sum())})
            all_scores.append(scores)
    output = pd.concat(all_scores, ignore_index=True)
    output.to_csv(GOLD / "replay_scores.csv", index=False)
    pd.DataFrame(monitor).to_csv(GOLD / "replay_monitoring.csv", index=False)
    write_json(GOLD / "replay_summary.json", {"model": kind, "threshold": bundle["threshold"],
        "first": str(output.datetime.min()), "last": str(output.datetime.max()), "batches": len(monitor),
        "scored": int(output.score.notna().sum()), "suppressed": int(output.score.isna().sum()),
        "reviews": int(output.review_requested.sum()), "mode": "historical replay, not live plant data",
        "model_sha256": registry["model_hashes"][kind]})
    print(f"Replayed {len(monitor)} batches; {output.review_requested.sum()} review requests")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", help="Historical first day, e.g. 2015-12-01")
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()
    main(args.start, args.days)
