"""Gold layer: run each analysis, store marts in Parquet + DuckDB, and assemble dashboard payloads."""
import math
from datetime import datetime, timezone
from time import perf_counter

import duckdb
import numpy as np
import pandas as pd

from vg import bronze, commercial, fleet, forecast, price_model, production, reliability, shipping, silver
from vg.common import GOLD, SILVER, WAREHOUSE, write_parquet
from vg.contracts import report


def records(frame, columns=None, digits=4):
    """JSON-safe records: dates as ISO strings, NaN as null, floats rounded."""
    frame = frame[columns] if columns else frame
    out = []
    for row in frame.to_dict("records"):
        clean = {}
        for key, value in row.items():
            if isinstance(value, (pd.Timestamp, datetime)):
                value = value.date().isoformat() if value == value.normalize() else value.isoformat(timespec="minutes")
            elif isinstance(value, (float, np.floating)):
                value = None if not math.isfinite(value) else round(float(value), digits)
            elif isinstance(value, (np.integer,)):
                value = int(value)
            elif isinstance(value, (np.bool_,)):
                value = bool(value)
            elif value is pd.NaT or (value is not None and not isinstance(value, (str, bool, int)) and pd.isna(value)):
                value = None
            clean[key] = value
        out.append(clean)
    return out


def scalars(d):
    return {k: (None if isinstance(v, float) and not math.isfinite(v) else round(v, 4) if isinstance(v, float) else v)
            for k, v in d.items()}


def store(tables, schema, con):
    folder = (SILVER if schema == "silver" else GOLD)
    for name, frame in tables.items():
        write_parquet(frame, folder / f"{name}.parquet")
        con.register("frame", frame)
        con.execute(f"CREATE OR REPLACE TABLE {schema}.{name} AS SELECT * FROM frame")
        con.unregister("frame")


def build(timings):
    def step(name, fn, *args):
        start = perf_counter()
        print(f"  {name}...", flush=True)
        result = fn(*args)
        timings.append({"step": name, "seconds": round(perf_counter() - start, 2)})
        return result

    manifest = step("verify bronze", bronze.verify)
    s = step("build silver", silver.build)
    distances = step("sea-route distances", shipping.distances, s["terminals"], s["destinations"])

    cp = s["monthly_exports"][s["monthly_exports"].terminal == "Calcasieu Pass"][["month", "volume_bcf"]].reset_index(drop=True)
    outlook, backtest, cp_flags = step("export forecast", forecast.analyze, cp)

    util = production.utilization(s["monthly_exports"], s["terminals"])
    ramp = production.ramp(util)
    windows, pace, response, rel_summary, rel_stats = step(
        "reliability", reliability.analyze, s["cargoes"], s["hurricanes"], s["storm_events"], s["events"],
        s["weather_daily"], silver.CUTOFF)
    low_months = sorted({m for w in windows.itertuples() for m in pd.period_range(w.start, w.end, freq="M").astype(str)})
    heat, heat_points = production.heat_effect(util, s["weather_daily"], s["terminals"], low_months)

    nb = step("netbacks", commercial.netbacks, s["prices_monthly"], distances, s["assumptions"])
    mix = commercial.destination_mix(s["cargoes"])
    choice, choice_stats = commercial.choice_vs_spread(mix, nb)
    realized = commercial.realized_vs_modeled(s["cargoes"], nb, s["sec_quarterly"], distances, s["assumptions"])
    reconciliation = commercial.reconcile(s["cargoes"], s["sec_quarterly"])
    weekly, hh_predictions, hh_metrics, hh_outlook = step("Henry Hub models", price_model.analyze, s["henry_hub_daily"])

    share, ships = fleet.fleet_share(s["cargoes"])
    trips = step("round trips", fleet.round_trips, s["cargoes"], s["us_loadings"], distances, s["assumptions"])
    calls = fleet.port_calls(s["ais_positions"], s["cargoes"])
    speed_points, speed_effect, speed_bins = step("AIS speed model", fleet.speed_vs_weather, s["ais_positions"], calls, s["gulf_weather"])
    costs = fleet.voyage_costs(distances, s["assumptions"])

    gold = {"sea_distances": distances, "export_backtest": backtest, "utilization": util, "ramp": ramp,
            "heat_effect": heat, "low_output_windows": windows, "hazard_response": response, "netbacks": nb,
            "destination_mix": mix, "realized_vs_modeled": realized, "doe_sec_reconciliation": reconciliation,
            "henry_hub_forecasts": hh_predictions, "henry_hub_metrics": hh_metrics, "round_trips": trips,
            "port_calls": calls, "voyage_costs": costs, "ship_league": ships}
    with duckdb.connect(str(WAREHOUSE)) as con:
        con.execute("CREATE SCHEMA IF NOT EXISTS silver; CREATE SCHEMA IF NOT EXISTS gold")
        step("store silver", store, s, "silver", con)
        step("store gold", store, gold, "gold", con)
        lineage = con.execute("""SELECT schema_name AS layer, table_name AS name, estimated_size AS rows
                                 FROM duckdb_tables() WHERE schema_name IN ('silver', 'gold') ORDER BY 1, 2""").df()

    monthly = s["monthly_exports"]
    latest = monthly.month.max()
    last = monthly[monthly.month == latest]
    year_ago = monthly[monthly.month == str(pd.Period(latest, freq="M") - 12)]
    q_trips = trips[trips.route_inferred.isin(["Panama", "Suez or Cape", "Ambiguous"])]
    built = datetime.now(timezone.utc).isoformat(timespec="seconds")
    meta = {"built_at": built, "cutoff": silver.CUTOFF}

    payloads = {
        "index": {**meta, "latest_month": latest,
                  "terminals": records(last.merge(util[["terminal", "month", "utilization"]], on=["terminal", "month"])),
                  "total_bcf": float(last.volume_bcf.sum()), "total_cargoes": int(last.cargoes.sum()),
                  "year_ago_bcf": float(year_ago.volume_bcf.sum()),
                  "monthly": records(monthly.pivot_table(index="month", columns="terminal", values="volume_bcf").reset_index()),
                  "headlines": {"heat": scalars(heat.iloc[0].to_dict()), "reliability": scalars(rel_stats),
                                "choice": scalars(choice_stats),
                                "hh": records(hh_metrics[hh_metrics.horizon_weeks == 4]),
                                "reconciliation_max_pct": float(reconciliation.tbtu_difference_pct.abs().max()),
                                "asia_long_route_share": float((q_trips.route_inferred == "Suez or Cape").mean()),
                                "calls": len(calls), "calls_matched": int(calls.doe_departure.notna().sum()),
                                "sources": len(manifest),
                                "ramp": records(ramp)}},
        "exports": {**meta, "monthly": records(monthly), "cp_flags": records(cp_flags[["month", "prior_six_month_average", "review_flag"]]),
                    "destinations": records(s["allocations"].groupby(["terminal", "month", "destination", "region"]).volume_bcf.sum().reset_index()),
                    "forecast": scalars(outlook), "test": records(backtest)},
        "production": {**meta, "utilization": records(util), "ramp": records(ramp), "heat": records(heat),
                       "heat_points": records(heat_points), "terminals": records(s["terminals"])},
        "reliability": {**meta, "pace": records(pace.dropna(subset=["pace30_bcfd"])), "windows": records(windows),
                        "hazards": records(response), "summary": records(rel_summary), "stats": scalars(rel_stats),
                        "events": records(s["events"]),
                        "settings": {"gap_factor": reliability.GAP_FACTOR, "min_gap_days": reliability.MIN_GAP_DAYS,
                                     "dip_ratio": reliability.DIP_RATIO, "warmup_days": reliability.WARMUP_DAYS,
                                     "freeze_c": reliability.FREEZE_C, "gale_kmh": reliability.GALE_KMH}},
        "commercial": {**meta, "netbacks": records(nb[nb.month >= "2019-01"]), "mix": records(choice),
                       "choice": scalars(choice_stats), "realized": records(realized),
                       "hh_weekly": records(weekly[weekly.week >= "2019-01-01"][["week", "price"]]),
                       "hh_test": records(hh_predictions[hh_predictions.horizon_weeks == 4]),
                       "hh_metrics": records(hh_metrics), "hh_outlook": records(hh_outlook),
                       "assumptions": records(s["assumptions"])},
        "fleet": {**meta, "share": records(share), "ships": records(ships.head(25)), "trips": records(trips[trips.direct]),
                  "routes": records(q_trips.assign(quarter=pd.PeriodIndex(q_trips.date, freq="Q").astype(str))
                                    .groupby(["quarter", "route_inferred"]).size().rename("voyages").reset_index()),
                  "calls": records(calls.drop(columns=["mmsi"])), "speed": scalars(speed_effect), "speed_bins": records(speed_bins),
                  "costs": records(costs), "assumptions": records(s["assumptions"])},
        "data": {**meta, "sources": [{"id": k, **{f: v.get(f) for f in ("description", "license", "url", "sha256", "bytes", "retrieved_at", "builder")}}
                                     for k, v in manifest.items()],
                 "quality": records(report()), "reconciliation": records(reconciliation), "lineage": records(lineage),
                 "timings": timings, "events": records(s["events"])},
    }
    downloads = {"monthly_exports.csv": monthly, "cargoes.csv": s["cargoes"], "allocations.csv": s["allocations"],
                 "utilization.csv": util, "low_output_windows.csv": windows, "netbacks.csv": nb,
                 "realized_vs_modeled.csv": realized, "doe_sec_reconciliation.csv": reconciliation,
                 "henry_hub_forecasts.csv": hh_predictions, "round_trips.csv": trips, "port_calls.csv": calls,
                 "export_backtest.csv": backtest, "data_quality.csv": report()}
    return payloads, downloads
