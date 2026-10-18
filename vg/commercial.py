"""Destination economics: Europe versus Asia netbacks, actual cargo choices, and realized SEC prices."""
import numpy as np
import pandas as pd
from scipy import stats

from vg import shipping
from vg.common import TBTU_PER_BCF
from vg.contracts import check

BENCHMARKS = {"Europe": "Netherlands", "Asia": "Japan"}   # TTF-like and JKM-like delivery points
DRAWS = 4000


def benchmark_routes(distances):
    d = distances[distances.terminal == "Calcasieu Pass"].set_index(["country", "route"])
    return {"Europe": d.loc[(BENCHMARKS["Europe"], "shortest")],
            "Asia (via Panama)": d.loc[(BENCHMARKS["Asia"], "shortest")],
            "Asia (no Panama)": d.loc[(BENCHMARKS["Asia"], "no_panama")]}


def netbacks(prices, distances, assumptions, seed=11):
    """Monthly FOB netbacks at the base case and Monte Carlo probabilities that Europe pays more."""
    routes = benchmark_routes(distances)
    rng = np.random.default_rng(seed)
    draws = shipping.draw(assumptions, DRAWS, rng)
    base = shipping.params(assumptions)
    rows = []
    for row in prices.dropna(subset=["europe", "asia"]).itertuples():
        out = {"month": row.month, "henry_hub": row.henry_hub, "europe_price": row.europe, "asia_price": row.asia,
               "feed_gas_cost": base["feed_gas_multiplier"] * row.henry_hub}
        simulated = {}
        for name, route in routes.items():
            price = row.europe if name == "Europe" else row.asia
            args = (route.nautical_miles, route.via_panama, route.via_suez)
            out[f"netback_{name}"] = shipping.netback(price, *args, base)
            out[f"shipping_{name}"] = shipping.voyage(*args, base)["shipping_usd_mmbtu"]
            simulated[name] = shipping.netback(price, *args, draws)
        out["p_europe_beats_asia_panama"] = float((simulated["Europe"] > simulated["Asia (via Panama)"]).mean())
        out["p_europe_beats_asia_no_panama"] = float((simulated["Europe"] > simulated["Asia (no Panama)"]).mean())
        out["europe_minus_asia"] = out["netback_Europe"] - out["netback_Asia (via Panama)"]
        rows.append(out)
    frame = pd.DataFrame(rows)
    return check(frame, "gold.netbacks", ["month", "netback_Europe", "netback_Asia (via Panama)", "p_europe_beats_asia_panama"],
                 unique=["month"], ranges={"p_europe_beats_asia_panama": (0, 1)})


def destination_mix(cargoes):
    mix = cargoes.pivot_table(index="month", columns="region", values="volume_bcf", aggfunc="sum", fill_value=0)
    share = mix.div(mix.sum(axis=1), axis=0).add_prefix("share_")
    europe_asia = mix.get("Europe", 0) + mix.get("Asia", 0)
    share["europe_share_of_europe_asia"] = mix.get("Europe", 0) / europe_asia.replace(0, np.nan)
    return share.reset_index()


def choice_vs_spread(mix, nb):
    """Does a higher modeled Europe advantage coincide with more cargoes to Europe (Europe vs Asia only)?"""
    data = mix.merge(nb, on="month").dropna(subset=["europe_share_of_europe_asia"])
    rho, p = stats.spearmanr(data.europe_minus_asia, data.europe_share_of_europe_asia)
    agree = ((data.europe_minus_asia > 0) == (data.europe_share_of_europe_asia > 0.5)).mean()
    return data, {"months": len(data), "spearman_rho": float(rho), "p_value": float(p), "direction_agreement": float(agree)}


def realized_vs_modeled(cargoes, nb, sec_quarterly, distances, assumptions):
    """Compare SEC realized prices with modeled FOB netbacks for the actual destination mix each quarter."""
    base = shipping.params(assumptions)
    d = distances[distances.route == "shortest"].set_index(["terminal", "country"])
    price = nb.set_index("month")
    rows = []
    for c in cargoes[cargoes.region.isin(["Europe", "Asia"]) & cargoes.month.isin(price.index)].itertuples():
        route = d.loc[(c.terminal, c.destination)]
        market = price.loc[c.month, "europe_price" if c.region == "Europe" else "asia_price"]
        rows.append({"terminal": c.terminal, "quarter": str(pd.Period(c.month, freq="Q")), "volume_bcf": c.volume_bcf,
                     "netback": shipping.netback(market, route.nautical_miles, route.via_panama, route.via_suez, base),
                     "feed_gas": price.loc[c.month, "feed_gas_cost"]})
    frame = pd.DataFrame(rows)
    weighted = frame.groupby(["terminal", "quarter"]).apply(lambda g: pd.Series({
        "modeled_netback": np.average(g.netback, weights=g.volume_bcf),
        "feed_gas_cost": np.average(g.feed_gas, weights=g.volume_bcf),
        "doe_bcf": g.volume_bcf.sum()}), include_groups=False).reset_index()
    out = sec_quarterly.merge(weighted, on=["terminal", "quarter"], how="left")
    out["realized_minus_modeled"] = out.price_per_mmbtu - out.modeled_netback
    return out[["terminal", "quarter", "price_per_mmbtu", "modeled_netback", "feed_gas_cost", "realized_minus_modeled", "tbtu_sold"]]


def reconcile(cargoes, sec_quarterly):
    """DOE physical exports versus volumes and cargo counts reported to the SEC, by quarter."""
    doe = cargoes.assign(quarter=pd.PeriodIndex(cargoes.date, freq="Q").astype(str))
    doe = doe.groupby(["terminal", "quarter"]).agg(doe_bcf=("volume_bcf", "sum"), doe_cargoes=("volume_bcf", "size")).reset_index()
    doe["doe_tbtu"] = doe.doe_bcf * TBTU_PER_BCF
    out = sec_quarterly.merge(doe, on=["terminal", "quarter"], how="left")
    reported = out.tbtu_exported.fillna(out.tbtu_sold)
    out["sec_tbtu"] = reported
    out["tbtu_difference_pct"] = 100 * (out.doe_tbtu - reported) / reported
    out["cargo_difference"] = out.doe_cargoes - out.cargoes_exported
    return out[["terminal", "quarter", "sec_tbtu", "doe_tbtu", "tbtu_difference_pct", "cargoes_exported", "doe_cargoes", "cargo_difference", "filing"]]
