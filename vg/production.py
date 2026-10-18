"""Utilization against nameplate, ramp-up speed, and the effect of hot weather on output."""
import numpy as np
import pandas as pd
import statsmodels.api as sm

from vg.common import BCF_PER_MTPA_YEAR
from vg.contracts import check

STEADY_FROM = {"Calcasieu Pass": "2022-10"}  # ramp-up complete (near-full output from October 2022)


def utilization(monthly, terminals):
    frame = monthly.merge(terminals[["terminal", "nameplate_mtpa", "authorized_mtpa"]], on="terminal")
    frame["nameplate_bcfd"] = frame.nameplate_mtpa * BCF_PER_MTPA_YEAR / 365.25
    frame["authorized_bcfd"] = frame.authorized_mtpa * BCF_PER_MTPA_YEAR / 365.25
    frame["utilization"] = frame.bcfd / frame.nameplate_bcfd
    frame["months_since_first_cargo"] = frame.groupby("terminal").cumcount()
    frame["utilization_3m"] = frame.groupby("terminal").utilization.transform(lambda s: s.rolling(3).mean())
    return check(frame.drop(columns=["nameplate_mtpa", "authorized_mtpa"]), "gold.utilization",
                 ["terminal", "month", "bcfd", "utilization", "months_since_first_cargo"],
                 not_null=["terminal", "month", "bcfd", "utilization"], ranges={"utilization": (0, 2)})


def ramp(util):
    rows = []
    for terminal, group in util.groupby("terminal"):
        reached = group[group.utilization_3m >= 0.9]
        rows.append({"terminal": terminal, "first_month": group.month.iloc[0],
                     "months_to_90pct": int(reached.months_since_first_cargo.iloc[0]) if len(reached) else None,
                     "latest_utilization": float(group.utilization.iloc[-1]),
                     "peak_3m_utilization": float(group.utilization_3m.max())})
    return pd.DataFrame(rows)


def heat_effect(util, weather, terminals, exclude_months=()):
    """OLS of monthly export pace on mean temperature with a trend and post-COD shift; HAC standard errors."""
    results, points = [], []
    for terminal, start in STEADY_FROM.items():
        temps = weather[weather.terminal == terminal].assign(month=lambda d: d.date.dt.strftime("%Y-%m"))
        temps = temps.groupby("month").agg(temp_mean_c=("temp_mean_c", "mean"), temp_max_c=("temp_max_c", "mean"))
        data = util[(util.terminal == terminal) & (util.month >= start)].merge(temps, on="month")
        cod = terminals.set_index("terminal").cod[terminal]
        data["after_cod"] = (data.month >= str(cod)[:7]).astype(float)
        data["trend_years"] = np.arange(len(data)) / 12
        data["excluded"] = data.month.isin(exclude_months)
        for label, sample in [("All steady-state months", data), ("Excluding low-output months", data[~data.excluded])]:
            x = sm.add_constant(sample[["temp_mean_c", "trend_years", "after_cod"]])
            fit = sm.OLS(sample.bcfd, x).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
            low, high = fit.conf_int().loc["temp_mean_c"]
            nameplate = float(sample.nameplate_bcfd.iloc[0])
            results.append({"terminal": terminal, "sample": label, "months": int(fit.nobs),
                            "bcfd_per_c": fit.params.temp_mean_c, "ci_low": low, "ci_high": high,
                            "pct_nameplate_per_c": 100 * fit.params.temp_mean_c / nameplate,
                            "p_value": fit.pvalues.temp_mean_c, "r_squared": fit.rsquared})
        points.append(data[["terminal", "month", "bcfd", "utilization", "temp_mean_c", "temp_max_c", "after_cod", "excluded"]])
    return pd.DataFrame(results), pd.concat(points, ignore_index=True)
