"""Fleet analytics: own versus chartered ships, round trips and routes (DOE), and port calls and weather (AIS)."""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from vg import shipping
from vg.contracts import check
from vg.silver import haversine_km

BERTH = (29.7735, -93.3415)          # Calcasieu Pass berths, located from stationary AIS positions
BERTH_RADIUS_KM = 1.0
APPROACH_KM = 60
PLAUSIBLE_KNOTS = (13.0, 20.0)
MAX_DIRECT_DAYS = 150                # longer gaps mean the ship traded elsewhere before returning


def fleet_share(cargoes):
    share = cargoes.groupby(["month", "terminal"]).agg(cargoes=("own_fleet", "size"), own=("own_fleet", "sum")).reset_index()
    share["own_share"] = share.own / share.cargoes
    ships = (cargoes.groupby("vessel_key").agg(vessel=("vessel", "first"), own_fleet=("own_fleet", "first"),
             cargoes=("volume_bcf", "size"), volume_bcf=("volume_bcf", "sum"), first=("date", "min"), last=("date", "max"))
             .reset_index().sort_values("cargoes", ascending=False))
    return share, ships


def round_trips(cargoes, us_loadings, distances, assumptions):
    """Days until the same ship next loads anywhere in the U.S., compared with route-based expectations."""
    base = shipping.params(assumptions)
    loads = us_loadings.sort_values("date").groupby("vessel_key").date.apply(np.array).to_dict()
    d = distances.pivot_table(index=["terminal", "country"], columns="route", values="nautical_miles")
    rows = []
    for c in cargoes.itertuples():
        later = loads.get(c.vessel_key, np.array([], dtype="datetime64[ns]"))
        later = later[later > np.datetime64(c.date + pd.Timedelta(days=5))]
        if not len(later):
            continue
        days = (pd.Timestamp(later[0]) - c.date).days
        miles = d.loc[(c.terminal, c.destination)]
        row = {"terminal": c.terminal, "date": c.date, "vessel_key": c.vessel_key, "own_fleet": c.own_fleet,
               "destination": c.destination, "region": c.region, "round_trip_days": days, "direct": days <= MAX_DIRECT_DAYS}
        for route in shipping.ROUTES:
            sailing = days - base["port_days"]
            row[f"speed_{route}"] = 2 * miles[route] / (sailing * 24) if sailing > 0 and pd.notna(miles[route]) else np.nan
            row[f"expected_{route}"] = 2 * miles[route] / (base["speed"] * 24) + base["port_days"]
        rows.append(row)
    trips = pd.DataFrame(rows)
    low, high = PLAUSIBLE_KNOTS
    panama = trips.speed_shortest.between(low, high)
    longer = trips.speed_no_panama.between(low, high) | trips.speed_cape.between(low, high)
    asia = trips.region.eq("Asia") & trips.direct
    trips["route_inferred"] = np.select([asia & panama & ~longer, asia & longer & ~panama, asia & panama & longer],
                                        ["Panama", "Suez or Cape", "Ambiguous"], default="")
    return check(trips, "gold.round_trips", ["terminal", "date", "vessel_key", "region", "round_trip_days"],
                 ranges={"round_trip_days": (5, 2000)})


def port_calls(ais, cargoes):
    """Berth stays at Calcasieu Pass from AIS, with waiting time and a check against DOE departure dates."""
    ais = ais.copy()
    ais["berth_km"] = haversine_km(ais.lat, ais.lon, *BERTH)
    ais["at_berth"] = (ais.berth_km <= BERTH_RADIUS_KM) & (ais.sog_kn < 1)
    calls = []
    for mmsi, track in ais.groupby("mmsi"):
        berth = track[track.at_berth]
        if berth.empty:
            continue
        stay = (berth.time.diff() > pd.Timedelta(hours=3)).cumsum()
        for _, s in berth.groupby(stay):
            start, end = s.time.min(), s.time.max()
            if end - start < pd.Timedelta(hours=6):
                continue
            before = track[(track.time < start) & (track.time > start - pd.Timedelta(days=10))]
            approach = before[before.berth_km <= APPROACH_KM]
            # Approach begins at the last entry into the 60 km zone before berthing.
            outside = before[before.berth_km > APPROACH_KM]
            entered = approach[approach.time > outside.time.max()].time.min() if len(outside) else approach.time.min()
            waiting = approach[(approach.time >= entered) & (approach.sog_kn < 1) & ~approach.at_berth]
            calls.append({"mmsi": mmsi, "vessel_key": s.vessel_key.iloc[0], "vessel": s.vessel.iloc[0],
                          "approach_start": entered, "berth_start": start, "berth_end": end,
                          "berth_hours": (end - start).total_seconds() / 3600,
                          "waiting_hours": len(waiting) * 10 / 60 if pd.notna(entered) else np.nan,
                          "track_complete": bool(len(outside)) and pd.notna(entered)})
    calls = pd.DataFrame(calls)
    doe = cargoes[cargoes.terminal == "Calcasieu Pass"]
    near = lambda c: doe.date.between(c.berth_end.normalize() - pd.Timedelta(days=1), c.berth_end.normalize() + pd.Timedelta(days=1))
    dates, methods, doe_names = [], [], []
    for c in calls.itertuples():
        same = doe[near(c) & (doe.vessel_key == c.vessel_key)]
        dates.append(same.date.iloc[0] if len(same) else pd.NaT)
        methods.append("name and date" if len(same) else "")
        doe_names.append(same.vessel.iloc[0] if len(same) else "")
    # Ships are renamed and AIS names get abbreviated: fall back to the only unmatched DOE cargo that day.
    used = set(zip(calls.vessel_key, dates))
    for i, c in enumerate(calls.itertuples()):
        if methods[i]:
            continue
        candidates = doe[near(c) & ~doe.vessel_key.isin(calls.vessel_key)]
        if len(candidates) == 1:
            dates[i], methods[i], doe_names[i] = candidates.date.iloc[0], "date only (name differs)", candidates.vessel.iloc[0]
    calls["doe_departure"], calls["match_method"], calls["doe_vessel"] = dates, methods, doe_names
    return check(calls, "gold.port_calls", ["mmsi", "vessel_key", "berth_start", "berth_end", "berth_hours"],
                 unique=["mmsi", "berth_start"], ranges={"berth_hours": (6, 240)})


def speed_vs_weather(ais, calls, gulf):
    """Open-water speed against wave height and wind, laden versus ballast, with ship fixed effects."""
    ais = ais.copy()
    ais["time"] = ais.time.astype("datetime64[ns]")
    ais["berth_km"] = haversine_km(ais.lat, ais.lon, *BERTH)
    legs = []
    for c in calls.itertuples():
        track = ais[ais.mmsi == c.mmsi]
        legs.append(track[(track.time > c.berth_end) & (track.time < c.berth_end + pd.Timedelta(days=6))].assign(leg="laden"))
        legs.append(track[(track.time < c.berth_start) & (track.time > c.berth_start - pd.Timedelta(days=6))].assign(leg="ballast"))
    points = pd.concat(legs).drop_duplicates(["mmsi", "time"])
    points = points[(points.sog_kn >= 6) & (points.berth_km >= 80)].copy()
    points["grid_lat"], points["grid_lon"] = points.lat.round().astype(int), points.lon.round().astype(int)
    points["hour"] = points.time.dt.floor("h")
    weather = gulf.rename(columns={"lat": "grid_lat", "lon": "grid_lon", "time": "hour"})
    points = points.merge(weather, on=["hour", "grid_lat", "grid_lon"], how="inner")
    fit = smf.ols("sog_kn ~ wave_height_m + wind_kn + C(leg) + C(vessel_key)", data=points).fit(
        cov_type="cluster", cov_kwds={"groups": pd.factorize(points.vessel_key)[0]})
    effect = {"points": int(fit.nobs), "ships": int(points.vessel_key.nunique()),
              "knots_per_m_wave": float(fit.params.wave_height_m),
              "ci_low": float(fit.conf_int().loc["wave_height_m", 0]), "ci_high": float(fit.conf_int().loc["wave_height_m", 1]),
              "p_value_wave": float(fit.pvalues.wave_height_m), "knots_per_10kn_wind": float(10 * fit.params.wind_kn),
              "wind_ci_low": float(10 * fit.conf_int().loc["wind_kn", 0]), "wind_ci_high": float(10 * fit.conf_int().loc["wind_kn", 1]),
              "p_value_wind": float(fit.pvalues.wind_kn), "mean_speed_kn": float(points.sog_kn.mean()),
              "max_wave_m": float(points.wave_height_m.max())}
    bins = points.assign(wave_bin=pd.cut(points.wave_height_m, [0, 0.5, 1, 1.5, 2, 3, 6]))
    summary = bins.groupby(["wave_bin", "leg"], observed=True).sog_kn.agg(["mean", "size"]).reset_index()
    summary["wave_bin"] = summary.wave_bin.astype(str)
    return points, effect, summary


def voyage_costs(distances, assumptions):
    """Shipping cost per MMBtu to benchmark destinations under low, base and high assumptions."""
    rows = []
    d = distances[distances.terminal == "Calcasieu Pass"].set_index(["country", "route"])
    cases = [("Netherlands", "shortest", "Rotterdam"), ("Japan", "shortest", "Tokyo"), ("Japan", "no_panama", "Tokyo"),
             ("Japan", "cape", "Tokyo"), ("India", "shortest", "Dahej"), ("India", "cape", "Dahej"),
             ("Brazil", "shortest", "Rio de Janeiro"), ("China", "shortest", "Shanghai")]
    # Low cost uses low charter/fees and high speed; high cost the reverse.
    scenario = {"low": {**shipping.params(assumptions, "low"), "speed": assumptions.set_index("parameter").high["speed"]},
                "base": shipping.params(assumptions),
                "high": {**shipping.params(assumptions, "high"), "speed": assumptions.set_index("parameter").low["speed"]}}
    for country, route, city in cases:
        r = d.loc[(country, route)]
        via = "Panama" if r.via_panama else "Suez" if r.via_suez else "Cape of Good Hope" if route == "cape" else ""
        row = {"route": f"{city} via {via}" if via else city, "nautical_miles": r.nautical_miles}
        for name, p in scenario.items():
            v = shipping.voyage(r.nautical_miles, r.via_panama, r.via_suez, p)
            row[f"cost_{name}"] = v["shipping_usd_mmbtu"]
            if name == "base":
                row["round_trip_days"] = v["round_trip_days"]
        rows.append(row)
    return pd.DataFrame(rows)
