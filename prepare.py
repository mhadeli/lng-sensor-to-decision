"""Build cleaned tables, alerts, and the monthly planning table."""

import argparse
import os
import re
import subprocess
from pathlib import Path

import duckdb
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"

# In the test runs dXX_te.dat the fault is switched on at step 160.
# In the training runs dXX.dat it is already on when recording starts.
FAULT_START = 160


def load_tep():
    frames = []
    for path in sorted((RAW / "tep").glob("d*.dat")):
        frame = pd.read_csv(path, sep=r"\s+", header=None)
        # The normal training file is stored with one row per sensor.
        if frame.shape[1] > frame.shape[0]:
            frame = frame.T.reset_index(drop=True)
        frame.columns = [f"x{i}" for i in range(1, len(frame.columns) + 1)]
        fault_id = int(path.name[1:3])
        split = "test" if path.stem.endswith("_te") else "train"
        frame.insert(0, "split", split)
        frame.insert(1, "fault_id", fault_id)
        frame.insert(2, "step", range(len(frame)))
        fault_on = frame["step"] >= (FAULT_START if split == "test" else 0)
        frame.insert(3, "is_fault", (fault_on & (fault_id > 0)).astype(int))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def ensure_java():
    """PySpark needs Java. On a Mac, find it if JAVA_HOME is not set."""
    if os.environ.get("JAVA_HOME"):
        return
    try:
        home = subprocess.check_output(["/usr/libexec/java_home"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return
    os.environ["JAVA_HOME"] = home


def write_sensors_and_alerts_spark(sensors):
    """Spark stands in for Databricks. The normal band comes from the normal training run only."""
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    ensure_java()
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("lng-sensors")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    frame = spark.createDataFrame(sensors)
    measures = [column for column in frame.columns if re.fullmatch(r"x\d+", column)]
    normal = frame.filter((F.col("split") == "train") & (F.col("fault_id") == 0))
    stats = normal.agg(
        *[F.mean(column).alias(f"{column}_mean") for column in measures],
        *[F.stddev(column).alias(f"{column}_std") for column in measures],
    ).collect()[0]

    outside = F.lit(False)
    for column in measures:
        mean = stats[f"{column}_mean"]
        std = stats[f"{column}_std"] or 0.0
        outside = outside | (F.abs(F.col(column) - mean) > 3 * std)

    alerts = (
        frame.withColumn("outside_limits", outside.cast("int"))
        .select("split", "fault_id", "step", "is_fault", "outside_limits")
        .toPandas()
    )
    spark.stop()

    sensors.to_csv(SILVER / "sensors.csv", index=False)
    alerts.to_csv(GOLD / "alerts.csv", index=False)
    print(f"sensors: {len(sensors):,} rows, alerts: {alerts['outside_limits'].sum():,}")


def write_sensors_and_alerts(sensors):
    """Vectorized local preparation; Spark is optional for demonstrating scale-out."""
    if sensors.duplicated(["split", "fault_id", "step"]).any():
        raise ValueError("Duplicate sensor observations")
    measures = [c for c in sensors if re.fullmatch(r"x\d+", c)]
    if len(measures) != 52 or sensors[measures].isna().any().any():
        raise ValueError("Expected 52 complete sensor measurements")
    normal = sensors.loc[(sensors.split == "train") & (sensors.fault_id == 0), measures]
    alerts = sensors[["split", "fault_id", "step", "is_fault"]].copy()
    alerts["outside_limits"] = ((sensors[measures] - normal.mean()).abs() > 3 * normal.std()).any(axis=1).astype(int)
    sensors.to_csv(SILVER / "sensors.csv", index=False)
    alerts.to_csv(GOLD / "alerts.csv", index=False)
    print(f"Prepared {len(sensors):,} sensor observations")


def load_storms():
    """Louisiana storm events per month. Months with no storms count as zero."""
    # NOAA reissues each year with a newer creation date. Keep only the newest file per year.
    newest = {}
    for path in sorted((RAW / "noaa").glob("*.csv.gz")):
        newest[re.search(r"_d(\d{4})_", path.name).group(1)] = path
    storms = pd.concat(
        [pd.read_csv(path, usecols=["STATE", "BEGIN_YEARMONTH"]) for path in newest.values()],
        ignore_index=True,
    )
    storms = storms[storms["STATE"].str.strip().str.upper() == "LOUISIANA"]
    months = pd.to_datetime(storms["BEGIN_YEARMONTH"].astype(str), format="%Y%m")
    counts = months.dt.to_period("M").value_counts()
    every_month = pd.period_range(counts.index.min(), counts.index.max(), freq="M")
    counts = counts.reindex(every_month, fill_value=0).sort_index()
    return pd.DataFrame({"month": counts.index.astype(str), "storms": counts.to_numpy()})


def load_terminals():
    terminals = pd.read_csv(RAW / "gulf_lng_exports.csv")
    monthly = terminals.groupby("month", as_index=False)["exports_mmcf"].sum()
    return monthly.rename(columns={"exports_mmcf": "terminal_mmcf"})


def load_maintenance():
    work = pd.read_csv(RAW / "pdm" / "PdM_maint.csv", parse_dates=["datetime"])
    failures = pd.read_csv(RAW / "pdm" / "PdM_failures.csv", parse_dates=["datetime"])
    work["month"] = work["datetime"].dt.strftime("%Y-%m")
    failures["month"] = failures["datetime"].dt.strftime("%Y-%m")
    jobs = work.groupby("month").size().rename("maintenance_jobs")
    breaks = failures.groupby("month").size().rename("failures")
    return (
        pd.concat([jobs, breaks], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
        .sort_values("month")
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", action="store_true", help="Prepare the original supporting EDA datasets")
    parser.add_argument("--spark", action="store_true", help="Use optional PySpark for sensor preparation")
    args = parser.parse_args()
    SILVER.mkdir(parents=True, exist_ok=True)
    GOLD.mkdir(parents=True, exist_ok=True)
    prepare_sensors = write_sensors_and_alerts_spark if args.spark else write_sensors_and_alerts
    prepare_sensors(load_tep())

    exports = pd.read_csv(RAW / "lng_exports.csv")
    if exports.month.duplicated().any() or exports.exports_mmcf.isna().any() or (exports.exports_mmcf <= 0).any():
        raise ValueError("Expected unique months and positive export volumes")
    exports = duckdb.sql("SELECT month, exports_mmcf FROM exports ORDER BY month").df()
    exports.to_csv(SILVER / "exports.csv", index=False)
    if not args.extras:
        print(f"Prepared {len(exports):,} export months")
        return
    price = pd.read_csv(RAW / "henry_hub.csv")
    weather = pd.read_csv(RAW / "lake_charles_weather.csv").rename(
        columns={"DATE": "month", "TAVG": "temp_c"}
    )
    weather = weather.dropna(subset=["temp_c"])[["month", "temp_c"]]
    storms = load_storms()
    terminals = load_terminals()
    maintenance = load_maintenance()

    exports.to_csv(SILVER / "exports.csv", index=False)
    price.to_csv(SILVER / "henry_hub.csv", index=False)
    weather.to_csv(SILVER / "weather.csv", index=False)
    storms.to_csv(SILVER / "storms.csv", index=False)
    terminals.to_csv(SILVER / "terminals.csv", index=False)
    maintenance.to_csv(GOLD / "maintenance.csv", index=False)

    # DuckDB finds the pandas tables above by their variable names.
    monthly = duckdb.sql(
        """
        SELECT e.month, e.exports_mmcf, p.henry_hub, w.temp_c, s.storms, t.terminal_mmcf
        FROM exports e
        JOIN price p USING (month)
        JOIN weather w USING (month)
        JOIN storms s USING (month)
        JOIN terminals t USING (month)
        ORDER BY e.month
        """
    ).df()
    monthly.to_csv(GOLD / "monthly.csv", index=False)

    gap = duckdb.sql(
        """
        SELECT e.month, e.exports_mmcf, t.terminal_mmcf,
               e.exports_mmcf - t.terminal_mmcf AS gap_mmcf
        FROM exports e
        JOIN terminals t USING (month)
        ORDER BY e.month
        """
    ).df()
    gap.to_csv(GOLD / "volume_gap.csv", index=False)

    print(f"monthly planning rows: {len(monthly)} ({monthly['month'].min()} to {monthly['month'].max()})")
    print(f"volume gap rows: {len(gap)}")
    print(f"maintenance months: {len(maintenance)}")


if __name__ == "__main__":
    main()
