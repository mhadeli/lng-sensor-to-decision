"""Silver layer: validated, typed tables keyed by terminal, vessel and date."""
import gzip
import json

import numpy as np
import pandas as pd

from vg import bronze, sec
from vg.common import REFERENCE, vessel_key
from vg.contracts import check, note

CUTOFF = "2026-07"
DOE_COLUMNS = {"Arrival/Departure Date": "date", "Point of Entry or Exit": "point", "Volume (MMCF)": "volume_mmcf",
               "Tanker": "vessel", "Country": "destination", "Companies": "company", "Docket Term": "docket_term"}
HAZARDS = {
    "tropical": ["Hurricane", "Tropical Storm", "Tropical Depression", "Marine Tropical Storm",
                 "Marine Hurricane/Typhoon", "Marine Tropical Depression", "Storm Surge/Tide"],
    "fog": ["Dense Fog", "Marine Dense Fog", "Freezing Fog"],
    "cold": ["Winter Storm", "Winter Weather", "Cold/Wind Chill", "Extreme Cold/Wind Chill", "Frost/Freeze",
             "Ice Storm", "Heavy Snow", "Sleet", "Blizzard"],
    "wind": ["High Wind", "Strong Wind", "Marine High Wind", "Marine Strong Wind", "Thunderstorm Wind",
             "Marine Thunderstorm Wind", "Tornado", "Waterspout"],
    "flood": ["Flash Flood", "Flood", "Coastal Flood", "Heavy Rain"],
    "water_level": ["Astronomical Low Tide"],
    "heat": ["Heat", "Excessive Heat"],
}
STORM_TERMINAL = {"CAMERON": "Calcasieu Pass", "CALCASIEU": "Calcasieu Pass", "PLAQUEMINES": "Plaquemines",
                  "SOUTHWEST PASS": "Plaquemines", "BOOTHVILLE": "Plaquemines", "STAKE ISLAND": "Plaquemines"}
HURRICANE_RADIUS_KM = 300


def reference():
    terminals = check(pd.read_csv(REFERENCE / "terminals.csv"), "reference.terminals",
                      ["terminal", "doe_point", "lat", "lon", "nameplate_mtpa", "authorized_mtpa", "first_production"],
                      unique=["terminal"], ranges={"nameplate_mtpa": (1, 50)})
    destinations = check(pd.read_csv(REFERENCE / "destinations.csv"), "reference.destinations",
                         ["country", "region", "port", "lat", "lon"], unique=["country"],
                         ranges={"lat": (-90, 90), "lon": (-180, 180)})
    events = check(pd.read_csv(REFERENCE / "events.csv", parse_dates=["date"]), "reference.events",
                   ["date", "precision", "terminal", "category", "title", "source_url"])
    assumptions = check(pd.read_csv(REFERENCE / "assumptions.csv"), "reference.assumptions",
                        ["parameter", "low", "base", "high", "unit"], unique=["parameter"])
    if ((assumptions.low > assumptions.base) | (assumptions.base > assumptions.high)).any():
        raise ValueError("Each assumption needs low <= base <= high")
    return terminals, destinations, events, assumptions


def doe(terminals, destinations, raw=None):
    if raw is None:
        raw = pd.read_excel(bronze.path("doe_exports"), sheet_name="By Vessel and ISO Container")
    required = set(DOE_COLUMNS) | {"Activity", "Mode of Transport", "Gas Type"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Missing DOE columns: {sorted(required - set(raw.columns))}")
    frame = raw.rename(columns=DOE_COLUMNS)
    frame["source_row"] = np.arange(2, len(frame) + 2)
    frame = frame[(frame.Activity == "Exports") & (frame["Mode of Transport"] == "Vessel") & (frame["Gas Type"] == "LNG")].copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    frame["volume_mmcf"] = pd.to_numeric(frame.volume_mmcf, errors="coerce")
    frame["vessel_key"] = frame.vessel.map(vessel_key)
    # Every U.S. LNG loading, used to measure round trips of ships that also load at Venture Global terminals.
    us = frame.dropna(subset=["date", "vessel"]).groupby(["point", "date", "vessel_key"], as_index=False).volume_mmcf.sum()
    us = check(us.rename(columns={"point": "doe_point"}), "silver.us_loadings", ["doe_point", "date", "vessel_key"],
               unique=["doe_point", "date", "vessel_key"])

    points = dict(zip(terminals.doe_point, terminals.terminal))
    vg = frame[frame.point.isin(points)].copy()
    vg["terminal"] = vg.point.map(points)
    invalid = (vg.date.isna() | ~np.isfinite(vg.volume_mmcf) | (vg.volume_mmcf <= 0) | vg.vessel.isna() | vg.destination.isna())
    if invalid.any():
        raise ValueError(f"{int(invalid.sum())} invalid Venture Global allocation rows (dates, volumes, vessels or destinations)")
    other = vg[~vg.company.str.contains("Venture Global", na=False)]
    if len(other):
        note("silver.allocations", "warning", f"{len(other)} allocation row(s) at Venture Global terminals list another company "
             f"({', '.join(other.company.unique())}); kept because they are physical terminal exports", len(other))
    unmapped = sorted(set(vg.destination) - set(destinations.country))
    if unmapped:
        raise ValueError(f"Destinations missing from reference/destinations.csv: {unmapped}")
    vg = vg.merge(destinations[["country", "region"]], left_on="destination", right_on="country").drop(columns="country")
    vg["month"] = vg.date.dt.strftime("%Y-%m")
    vg["volume_bcf"] = vg.volume_mmcf / 1000
    vg["own_fleet"] = vg.vessel_key.str.startswith("VENTURE ")
    allocations = vg[["source_row", "terminal", "date", "month", "vessel", "vessel_key", "own_fleet", "destination",
                      "region", "volume_mmcf", "volume_bcf", "company", "docket_term"]].sort_values(["date", "source_row"])
    allocations = check(allocations.reset_index(drop=True), "silver.allocations", list(allocations.columns),
                        unique=["source_row"], ranges={"volume_bcf": (0.001, 5)})

    # Split allocations are legitimate: one cargo (vessel + date + terminal) can serve several buyers.
    def main_destination(group):
        top = group.sort_values("volume_bcf", ascending=False).iloc[0]
        return pd.Series({"volume_bcf": group.volume_bcf.sum(), "allocations": len(group),
                          "destinations": "; ".join(sorted(group.destination.unique())),
                          "destination": top.destination, "region": top.region, "vessel": top.vessel,
                          "own_fleet": bool(top.own_fleet)})
    cargoes = (allocations.groupby(["terminal", "date", "vessel_key"]).apply(main_destination, include_groups=False)
               .reset_index().sort_values(["date", "terminal"]))
    cargoes["month"] = cargoes.date.dt.strftime("%Y-%m")
    cargoes = check(cargoes.reset_index(drop=True), "silver.cargoes", list(cargoes.columns),
                    unique=["terminal", "date", "vessel_key"], ranges={"volume_bcf": (0.5, 5)})
    return allocations, cargoes, us


def monthly(allocations, cargoes, terminals):
    rows = []
    for terminal, group in allocations.groupby("terminal"):
        months = pd.period_range(group.month.min(), CUTOFF, freq="M").astype(str)
        observed = set(group.month)
        if missing := [m for m in months if m not in observed]:
            raise ValueError(f"{terminal}: no reported exports in {missing}. Do not substitute zero exports.")
        totals = group.groupby("month").agg(volume_bcf=("volume_bcf", "sum"), destinations=("destination", "nunique"))
        totals["cargoes"] = cargoes[cargoes.terminal == terminal].groupby("month").size()
        totals = totals.reindex(months).reset_index().rename(columns={"index": "month"})
        totals.insert(0, "terminal", terminal)
        rows.append(totals)
    frame = pd.concat(rows, ignore_index=True)
    frame["days"] = pd.to_datetime(frame.month).dt.days_in_month
    frame["bcfd"] = frame.volume_bcf / frame.days
    return check(frame, "silver.monthly_exports", list(frame.columns), unique=["terminal", "month"],
                 ranges={"bcfd": (0, 10), "cargoes": (1, 80)})


def weather(terminals):
    frames = []
    for terminal, key in [("Calcasieu Pass", "weather_calcasieu"), ("Plaquemines", "weather_plaquemines")]:
        daily = json.loads(bronze.path(key).read_text())["daily"]
        frame = pd.DataFrame(daily).rename(columns={"time": "date", "temperature_2m_mean": "temp_mean_c",
            "temperature_2m_max": "temp_max_c", "temperature_2m_min": "temp_min_c", "wind_speed_10m_max": "wind_max_kmh",
            "wind_gusts_10m_max": "gust_max_kmh", "precipitation_sum": "precip_mm"})
        frame.insert(0, "terminal", terminal)
        frames.append(frame)
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame.date)
    return check(frame, "silver.weather_daily", list(frame.columns), unique=["terminal", "date"],
                 ranges={"temp_mean_c": (-15, 40), "gust_max_kmh": (0, 300)})


def prices():
    hh = pd.read_excel(bronze.path("eia_henry_hub"), sheet_name="Data 1", skiprows=2)
    hh.columns = ["date", "henry_hub"]
    hh["date"] = pd.to_datetime(hh.date)
    hh = check(hh.dropna().reset_index(drop=True), "silver.henry_hub_daily", ["date", "henry_hub"], unique=["date"],
               ranges={"henry_hub": (-5, 60)}, min_rows=5000)
    monthly = hh.groupby(hh.date.dt.strftime("%Y-%m")).henry_hub.mean().rename_axis("month").reset_index()
    for key, column in [("fred_europe_gas", "europe"), ("fred_asia_lng", "asia")]:
        series = pd.read_csv(bronze.path(key))
        series.columns = ["date", column]
        series["month"] = pd.to_datetime(series.date).dt.strftime("%Y-%m")
        series[column] = pd.to_numeric(series[column], errors="coerce")
        monthly = monthly.merge(series[["month", column]], on="month", how="left")
    monthly = monthly[monthly.month >= "2016-01"].reset_index(drop=True)
    return hh, check(monthly, "silver.prices_monthly", ["month", "henry_hub", "europe", "asia"], unique=["month"],
                     not_null=["month", "henry_hub"], ranges={"europe": (0, 100), "asia": (0, 100)})


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def hurricanes(terminals):
    """Storms whose best track passed within 300 km of a terminal, 2021 onward."""
    points, storm, name = [], None, None
    for line in gzip.open(bronze.path("hurdat2"), "rt"):
        fields = [f.strip() for f in line.split(",")]
        if fields[0].startswith("AL"):
            storm, name = fields[0], fields[1]
            continue
        if int(fields[0][:4]) < 2021:
            continue
        lat = float(fields[4][:-1]) * (1 if fields[4].endswith("N") else -1)
        lon = float(fields[5][:-1]) * (-1 if fields[5].endswith("W") else 1)
        points.append({"storm_id": storm, "name": name.title(), "time": pd.to_datetime(fields[0] + fields[1], format="%Y%m%d%H%M"),
                       "status": fields[3], "lat": lat, "lon": lon, "wind_kt": int(fields[6])})
    track = pd.DataFrame(points)
    rows = []
    for _, terminal in terminals.iterrows():
        track["distance_km"] = haversine_km(track.lat, track.lon, terminal.lat, terminal.lon)
        near = track[track.distance_km <= HURRICANE_RADIUS_KM]
        for storm_id, group in near.groupby("storm_id"):
            closest = group.loc[group.distance_km.idxmin()]
            rows.append({"terminal": terminal.terminal, "storm_id": storm_id, "name": closest["name"],
                         "start": group.time.min().normalize(), "end": group.time.max().normalize(),
                         "closest_km": round(closest.distance_km), "max_wind_kt": int(group.wind_kt.max()),
                         "peak_status": group.loc[group.wind_kt.idxmax(), "status"]})
    frame = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    return check(frame, "silver.hurricanes", list(frame.columns), unique=["terminal", "storm_id"],
                 ranges={"closest_km": (0, HURRICANE_RADIUS_KM)})


def storm_events():
    raw = pd.read_csv(bronze.path("ncei_storm_events"), low_memory=False)
    groups = {event: group for group, events in HAZARDS.items() for event in events}
    raw = raw[raw.EVENT_TYPE.isin(groups)].copy()
    zone = raw.CZ_NAME.str.upper()
    raw["terminal"] = None
    for pattern, terminal in STORM_TERMINAL.items():
        raw.loc[zone.str.contains(pattern) & raw.terminal.isna(), "terminal"] = terminal
    frame = pd.DataFrame({
        "event_id": raw.EVENT_ID, "terminal": raw.terminal, "hazard": raw.EVENT_TYPE.map(groups), "event_type": raw.EVENT_TYPE,
        "zone": raw.CZ_NAME.str.title(),
        "begin": pd.to_datetime(raw.BEGIN_DATE_TIME, format="%d-%b-%y %H:%M:%S"),
        "end": pd.to_datetime(raw.END_DATE_TIME, format="%d-%b-%y %H:%M:%S")})
    frame = frame.sort_values("begin").reset_index(drop=True)
    return check(frame, "silver.storm_events", list(frame.columns), unique=["event_id"])


def ais():
    raw = pd.read_csv(bronze.path("ais_2023_10"), parse_dates=["BaseDateTime"])
    frame = pd.DataFrame({"mmsi": raw.MMSI, "time": raw.BaseDateTime, "lat": raw.LAT, "lon": raw.LON, "sog_kn": raw.SOG,
                          "cog": raw.COG, "vessel": raw.VesselName, "vessel_key": raw.VesselName.map(vessel_key),
                          "vessel_type": raw.VesselType, "length_m": raw.Length, "draft_m": raw.Draft, "match": raw.match})
    frame = frame.sort_values(["mmsi", "time"]).reset_index(drop=True)
    return check(frame, "silver.ais_positions", list(frame.columns), unique=["mmsi", "time"],
                 not_null=["mmsi", "time", "lat", "lon", "sog_kn", "vessel_key"],
                 ranges={"sog_kn": (0, 30), "lat": (17, 31), "lon": (-98, -80)})


def gulf_weather():
    frames = []
    for key, variable, column in [("gulf_waves_2023_10", "wave_height", "wave_height_m"),
                                  ("gulf_wind_2023_10", "wind_speed_10m", "wind_kn")]:
        locations = json.loads(gzip.open(bronze.path(key), "rt").read())
        for (lat, lon), location in zip(bronze.GRID, locations):
            frames.append(pd.DataFrame({"time": pd.to_datetime(location["hourly"]["time"]), "lat": lat, "lon": lon,
                                        "variable": column,
                                        "value": pd.to_numeric(pd.Series(location["hourly"][variable]), errors="coerce")}))
    long = pd.concat(frames, ignore_index=True)
    long["time"] = long.time.astype("datetime64[ns]")
    frame = long.pivot_table(index=["time", "lat", "lon"], columns="variable", values="value").reset_index()
    frame.columns.name = None
    frame = frame.dropna(subset=["wave_height_m"])  # land cells have no waves
    return check(frame, "silver.gulf_weather", ["time", "lat", "lon", "wave_height_m", "wind_kn"],
                 unique=["time", "lat", "lon"], ranges={"wave_height_m": (0, 20), "wind_kn": (0, 150)})


def sec_quarterly():
    frame = sec.quarterly()
    return check(frame, "silver.sec_quarterly", ["terminal", "quarter", "tbtu_sold", "price_per_mmbtu", "filing"],
                 unique=["terminal", "quarter"], ranges={"tbtu_sold": (0, 1000), "price_per_mmbtu": (0, 40)})


def build():
    terminals, destinations, events, assumptions = reference()
    allocations, cargoes, us = doe(terminals, destinations)
    hh, prices_monthly = prices()
    return {"terminals": terminals, "destinations": destinations, "events": events, "assumptions": assumptions,
            "allocations": allocations, "cargoes": cargoes, "us_loadings": us,
            "monthly_exports": monthly(allocations, cargoes, terminals), "weather_daily": weather(terminals),
            "henry_hub_daily": hh, "prices_monthly": prices_monthly, "hurricanes": hurricanes(terminals),
            "storm_events": storm_events(), "ais_positions": ais(), "gulf_weather": gulf_weather(),
            "sec_quarterly": sec_quarterly()}
