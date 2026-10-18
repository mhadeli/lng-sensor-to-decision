"""Validate public inputs and build a local SQL warehouse and causal feature table."""
import json

import duckdb
import numpy as np
import pandas as pd

from common import DATABASE, GOLD, RAW, ROOT, SILVER, file_hash, write_json, write_parquet

SCHEMAS = {
    "machines": ["machineID", "model", "age"],
    "telemetry": ["datetime", "machineID", "volt", "rotate", "pressure", "vibration"],
    "errors": ["datetime", "machineID", "errorID"],
    "maintenance": ["datetime", "machineID", "comp"],
    "failures": ["datetime", "machineID", "failure"],
}
SENSORS = ["volt", "rotate", "pressure", "vibration"]


def validate_table(frame, name, machine_ids=None):
    required = SCHEMAS[name]
    if set(frame.columns) != set(required):
        raise ValueError(f"Unexpected {name} schema: {list(frame)}")
    frame = frame.copy()
    if "datetime" in frame:
        frame["datetime"] = pd.to_datetime(frame.datetime, errors="coerce")
    numeric = ["machineID"] + (SENSORS if name == "telemetry" else ["age"] if name == "machines" else [])
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame[required].isna().any(axis=1) | ~np.isfinite(frame[numeric]).all(axis=1)
    invalid |= (frame.machineID < 1) | (frame.machineID % 1 != 0)
    if "datetime" in frame:
        invalid |= frame.datetime != frame.datetime.dt.floor("h")
    if machine_ids is not None:
        invalid |= ~frame.machineID.isin(machine_ids)
    if name == "telemetry":
        invalid |= (frame[SENSORS] <= 0).any(axis=1)
    if name == "machines":
        invalid |= (frame.age < 0) | ~frame.model.isin([f"model{i}" for i in range(1, 5)])
    for column, prefix, size in [("errorID", "error", 5), ("comp", "comp", 4), ("failure", "comp", 4)]:
        if column in frame:
            invalid |= ~frame[column].isin([f"{prefix}{i}" for i in range(1, size + 1)])
    keys = ["machineID"] if name == "machines" else ["machineID", "datetime"] if name == "telemetry" else required
    invalid |= frame.duplicated(keys, keep="first")
    rejected = frame[invalid].copy()
    rejected["reason"] = "invalid value, duplicate key, or unknown machine"
    return frame[~invalid].copy(), rejected


def build_features(con):
    return con.execute((ROOT / "sql" / "features.sql").read_text()).df()


def add_labels(features, failures, telemetry):
    """Strictly future failure in (t, t+24h]; unknown observation tails stay unlabeled."""
    result = features.copy()
    result["failure_24h"] = np.nan
    for machine, group in result.groupby("machineID"):
        observed = telemetry[telemetry.machineID == machine].datetime.sort_values()
        times = group.datetime.to_numpy(dtype="datetime64[us]")
        failure_times = np.sort(failures.loc[failures.machineID == machine, "datetime"].unique().astype("datetime64[us]"))
        left = np.searchsorted(failure_times, times, side="right")
        right = np.searchsorted(failure_times, times + np.timedelta64(24, "h"), side="right")
        # A complete future interval and no failure at the decision instant are required.
        future = observed.to_numpy(dtype="datetime64[us]")
        count = (np.searchsorted(future, times + np.timedelta64(24, "h"), side="right") -
                 np.searchsorted(future, times, side="right"))
        eligible = (count == 24) & ~np.isin(times, failure_times)
        result.loc[group.index[eligible], "failure_24h"] = (right > left)[eligible].astype(int)
    return result


def prepare_exports(con):
    """Preserve transaction rows; a vessel can legitimately have several allocations."""
    raw = pd.read_excel(RAW / "doe_lng_exports_2026_07.xlsx", sheet_name="By Vessel and ISO Container")
    raw = raw.rename(columns={"Arrival/Departure Date": "date", "Point of Entry or Exit": "terminal",
                              "Volume (MMCF)": "volume_mmcf"})
    raw["source_row"] = np.arange(2, len(raw) + 2)
    raw["date"] = pd.to_datetime(raw.date, errors="coerce")
    raw["volume_mmcf"] = pd.to_numeric(raw.volume_mmcf, errors="coerce")
    invalid = raw[["date", "terminal", "volume_mmcf"]].isna().any(axis=1) | (raw.volume_mmcf < 0)
    rejected = raw[invalid].copy()
    rejected.to_csv(ROOT / "data" / "quarantine" / "exports.csv", index=False)
    rows = raw[~invalid & (raw.Activity == "Exports") & (raw["Mode of Transport"] == "Vessel")].copy()
    con.register("transactions", rows)
    con.execute("CREATE OR REPLACE TABLE export_transactions AS SELECT * FROM transactions")
    monthly = con.execute("""SELECT strftime(date, '%Y-%m') AS month, terminal,
                              sum(volume_mmcf) AS exports_mmcf
                              FROM export_transactions GROUP BY month, terminal ORDER BY terminal, month""").df()
    # Fixed before modeling: long operating history, not chosen by forecast performance.
    series = monthly[monthly.terminal == "Sabine Pass, LA"].set_index("month").exports_mmcf
    months = pd.period_range("2017-01", "2026-07", freq="M").astype(str)
    selected = series.reindex(months)
    if selected.isna().any() or (selected <= 0).any():
        raise ValueError("Sabine Pass monthly coverage is incomplete; review the source, do not fill with invented volumes")
    export_table = selected.rename_axis("month").reset_index()
    write_parquet(monthly, SILVER / "terminal_exports.parquet")
    export_table.to_csv(SILVER / "exports.csv", index=False)
    return {"source_rows": len(raw), "quarantined": int(invalid.sum()), "vessel_export_rows": len(rows),
            "terminal": "Sabine Pass, LA", "months": len(export_table), "first": months[0], "last": months[-1]}


def main():
    SILVER.mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "quarantine").mkdir(parents=True, exist_ok=True)
    tables, quality = {}, {}
    for name in SCHEMAS:
        filename = "maint" if name == "maintenance" else name
        raw = pd.read_csv(RAW / "pdm" / f"PdM_{filename}.csv")
        clean, rejected = validate_table(raw, name, tables["machines"].machineID if name != "machines" else None)
        if clean.empty:
            raise ValueError(f"No valid {name} rows")
        rejected.to_csv(ROOT / "data" / "quarantine" / f"{name}.csv", index=False)
        write_parquet(clean, SILVER / f"{name}.parquet")
        tables[name] = clean
        quality[name] = {"raw_rows": len(raw), "valid_rows": len(clean), "quarantined": len(rejected)}
    building = DATABASE.with_suffix(".building.duckdb")
    with duckdb.connect(str(building)) as con:
        con.execute("SET threads=2")
        for name, frame in tables.items():
            con.register("source_frame", frame)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM source_frame")
        features = add_labels(build_features(con), tables["failures"], tables["telemetry"])
        con.register("feature_frame", features)
        con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM feature_frame")
        write_parquet(features, SILVER / "features.parquet")
        quality["exports"] = prepare_exports(con)
    building.replace(DATABASE)
    quality["features"] = {"rows": len(features), "labeled": int(features.failure_24h.notna().sum()),
                           "features": len(features.columns) - 4, "decision_interval_hours": 6}
    quality["manifest_sha256"] = file_hash(RAW / "manifest.json")
    write_json(GOLD / "quality.json", quality)
    print(json.dumps(quality, indent=2))


if __name__ == "__main__":
    main()
