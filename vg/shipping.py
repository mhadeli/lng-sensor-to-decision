"""Sea-route distances and voyage economics shared by the fleet and commercial analyses."""
import warnings

import numpy as np
import pandas as pd
import searoute

from vg.contracts import check

# Route options. "Shortest" may use Panama or Suez; "No Panama" models the 2023–24 drought restrictions;
# "Cape" also avoids Suez (e.g. Red Sea disruption).
ROUTES = {"shortest": ["northwest"], "no_panama": ["northwest", "panama"], "cape": ["northwest", "panama", "suez"]}
MMBTU_PER_BCF = 1.038e6


def distances(terminals, destinations):
    rows = []
    for _, terminal in terminals.iterrows():
        for _, port in destinations.iterrows():
            for route, restrictions in ROUTES.items():
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    path = searoute.searoute([terminal.lon, terminal.lat], [port.lon, port.lat], units="naut",
                                             restrictions=restrictions, return_passages=True)
                length = path.properties["length"]
                passages = path.properties.get("traversed_passages") or []
                rows.append({"terminal": terminal.terminal, "country": port.country, "region": port.region,
                             "port": port.port, "route": route, "nautical_miles": length if length > 0 else np.nan,
                             "via_panama": "panama" in passages, "via_suez": "suez" in passages})
    frame = pd.DataFrame(rows)
    return check(frame, "gold.sea_distances", ["terminal", "country", "route", "nautical_miles"],
                 unique=["terminal", "country", "route"], not_null=["terminal", "country", "route"],
                 ranges={"nautical_miles": (100, 25000)})


def params(assumptions, which="base"):
    return dict(zip(assumptions.parameter, assumptions[which]))


def draw(assumptions, n, rng):
    """Triangular draws for each assumption, as a dict of arrays."""
    return {row.parameter: rng.triangular(row.low, row.base, row.high, n) if row.high > row.low else np.full(n, row.base)
            for row in assumptions.itertuples()}


def voyage(nautical_miles, via_panama, via_suez, p, cargo_bcf=3.6):
    """Round-trip time and shipping cost in USD/MMBtu for one cargo. Works with scalars or arrays in `p`."""
    sailing_days = nautical_miles / (p["speed"] * 24)
    round_trip_days = 2 * sailing_days + p["port_days"]
    fees = p["panama_fee"] * via_panama + p["suez_fee"] * via_suez
    cost = (round_trip_days * p["charter_rate"] + fees) / (cargo_bcf * MMBTU_PER_BCF)
    boil_off = p["boil_off"] / 100 * sailing_days
    return {"laden_days": sailing_days, "round_trip_days": round_trip_days, "shipping_usd_mmbtu": cost,
            "boil_off_fraction": boil_off}


def netback(price, nautical_miles, via_panama, via_suez, p):
    """FOB netback at the Louisiana terminal: delivered price less boil-off, shipping and regasification."""
    v = voyage(nautical_miles, via_panama, via_suez, p)
    return price * (1 - v["boil_off_fraction"]) - v["shipping_usd_mmbtu"] - p["regas_fee"]
