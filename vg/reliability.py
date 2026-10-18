"""Detect low-output windows from cargo cadence and check them against public hazard and event records.

No public outage log exists for either terminal, so detections are validated against weather hazards
(hurricane tracks, NOAA Storm Events) and curated milestones. Unexplained windows are listed for review,
not counted as false alarms.
"""
import numpy as np
import pandas as pd

from vg.contracts import check

GAP_FACTOR = 3.0          # a gap is unusual when it exceeds 3x the recent typical gap...
MIN_GAP_DAYS = 6.0        # ...and at least six days
DIP_RATIO = 0.75          # or when 30-day volume falls below 75% of the prior 90-day pace
WARMUP_DAYS = 120         # skip each terminal's first months of start-up
FREEZE_C = -2.0           # ERA5 daily minimum at or below this is a hard freeze
GALE_KMH = 75.0           # ERA5 daily maximum gust at or above this is gale-force
SEVERE = {"tropical": None, "cold": None, "fog": None,
          "wind": ["High Wind", "Marine High Wind", "Strong Wind", "Marine Strong Wind", "Tornado"]}


def cadence(cargoes):
    rows = []
    for terminal, group in cargoes.groupby("terminal"):
        dates = group.groupby("date").volume_bcf.sum().sort_index()
        gaps = pd.Series(dates.index, index=dates.index).diff().dt.total_seconds() / 86400
        typical = gaps.shift(1).rolling(20, min_periods=8).median()
        frame = pd.DataFrame({"terminal": terminal, "date": dates.index, "volume_bcf": dates.values,
                              "gap_days": gaps.values, "typical_gap_days": typical.values})
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


def daily(cargoes, cutoff):
    frames = []
    for terminal, group in cargoes.groupby("terminal"):
        days = pd.date_range(group.date.min(), pd.Period(cutoff).end_time.normalize(), freq="D")
        volume = group.groupby("date").volume_bcf.sum().reindex(days, fill_value=0.0)
        pace30 = volume.rolling(30).sum() / 30
        prior90 = volume.shift(30).rolling(90).sum() / 90   # strictly before the current 30 days
        frames.append(pd.DataFrame({"terminal": terminal, "date": days, "volume_bcf": volume.values,
                                    "pace30_bcfd": pace30.values, "prior90_bcfd": prior90.values}))
    return pd.concat(frames, ignore_index=True)


def detect(cargoes, cutoff):
    """Union of gap-based and volume-dip windows, merged when they overlap."""
    windows = []
    gaps = cadence(cargoes)
    for row in gaps.itertuples():
        start = cargoes[cargoes.terminal == row.terminal].date.min()
        if (row.date - start).days < WARMUP_DAYS or pd.isna(row.typical_gap_days):
            continue
        if row.gap_days >= max(MIN_GAP_DAYS, GAP_FACTOR * row.typical_gap_days):
            windows.append({"terminal": row.terminal, "start": row.date - pd.Timedelta(days=row.gap_days),
                            "end": row.date, "detector": "cargo gap",
                            "expected_cargoes": row.gap_days / row.typical_gap_days})
    pace = daily(cargoes, cutoff)
    for terminal, group in pace.groupby("terminal"):
        group = group[group.date >= group.date.min() + pd.Timedelta(days=WARMUP_DAYS)]
        low = (group.pace30_bcfd < DIP_RATIO * group.prior90_bcfd).to_numpy()
        runs = np.flatnonzero(np.diff(np.r_[0, low.astype(int), 0]))
        for begin, finish in zip(runs[::2], runs[1::2]):
            windows.append({"terminal": terminal, "start": group.date.iloc[begin] - pd.Timedelta(days=29),
                            "end": group.date.iloc[finish - 1], "detector": "volume dip", "expected_cargoes": np.nan})
    if not windows:
        return pd.DataFrame(columns=["terminal", "start", "end", "detector"]), pace
    frame = pd.DataFrame(windows).sort_values(["terminal", "start"])
    merged = []
    for terminal, group in frame.groupby("terminal"):
        current = None
        for row in group.itertuples():
            if current and row.start <= current["end"]:
                current["end"] = max(current["end"], row.end)
                current["detector"] = " + ".join(sorted(set(current["detector"].split(" + ")) | {row.detector}))
            else:
                if current:
                    merged.append(current)
                current = {"terminal": terminal, "start": row.start, "end": row.end, "detector": row.detector}
        merged.append(current)
    windows = pd.DataFrame(merged)
    windows["days"] = (windows.end - windows.start).dt.days
    return windows, pace


def lost_volume(windows, pace):
    """Shortfall versus the prior 90-day pace over each window (Bcf)."""
    lost = []
    for row in windows.itertuples():
        days = pace[(pace.terminal == row.terminal) & pace.date.between(row.start, row.end)]
        baseline = pace[(pace.terminal == row.terminal) & (pace.date < row.start)].tail(90).volume_bcf.mean()
        lost.append(max(0.0, baseline * len(days) - days.volume_bcf.sum()))
    return lost


def weather_hazards(weather):
    """Freeze and gale spells from daily reanalysis, which also covers storms missing from NOAA Storm Events."""
    rows = []
    for (terminal, hazard), flag in [((t, "cold"), g.temp_min_c <= FREEZE_C) for t, g in weather.groupby("terminal")] + \
                                     [((t, "wind"), g.gust_max_kmh >= GALE_KMH) for t, g in weather.groupby("terminal")]:
        days = weather.loc[flag[flag].index].sort_values("date")
        spell = (days.date.diff().dt.days != 1).cumsum()
        for _, group in days.groupby(spell):
            extreme = group.temp_min_c.min() if hazard == "cold" else group.gust_max_kmh.max()
            label = (f"Hard freeze, low {extreme:.0f} °C" if hazard == "cold" else f"Gale gusts to {extreme:.0f} km/h")
            rows.append({"terminal": terminal, "start": group.date.min() - pd.Timedelta(days=1),
                         "end": group.date.max() + pd.Timedelta(days=4), "kind": "weather", "hazard": hazard,
                         "label": f"{label} (ERA5)"})
    return rows


def hazards(hurricanes, storm_events, events, weather):
    """One table of dated hazards and context events, each with a matching interval."""
    rows = weather_hazards(weather) + [{"terminal": h.terminal, "start": h.start - pd.Timedelta(days=2), "end": h.end + pd.Timedelta(days=7),
             "kind": "weather", "hazard": "tropical", "label": f"{h.name} ({h.peak_status}, closest {h.closest_km} km)"}
            for h in hurricanes.itertuples()]
    for e in storm_events.itertuples():
        allowed = SEVERE.get(e.hazard, False)
        if allowed is False or (allowed and e.event_type not in allowed):
            continue
        rows.append({"terminal": e.terminal, "start": e.begin.normalize() - pd.Timedelta(days=1),
                     "end": e.end.normalize() + pd.Timedelta(days=4), "kind": "weather", "hazard": e.hazard,
                     "label": f"{e.event_type} ({e.zone})"})
    for e in events.itertuples():
        pad = pd.Timedelta(days=31 if e.precision == "month" else 15)
        rows.append({"terminal": e.terminal, "start": e.date - pad, "end": e.date + pad, "kind": "context",
                     "hazard": e.category, "label": e.title})
    frame = pd.DataFrame(rows)
    # Collapse multi-zone reports of the same weather on the same day.
    return frame.drop_duplicates(["terminal", "start", "hazard", "kind"]).sort_values("start").reset_index(drop=True)


ONSET_BEFORE, ONSET_AFTER = pd.Timedelta(days=3), pd.Timedelta(days=10)


def onset_hits(terminal, start, hazard_table):
    """Hazards active around the onset of a window; long windows do not get credit for later coincidences."""
    return hazard_table[(hazard_table.terminal == terminal) & (hazard_table.start <= start + ONSET_AFTER) &
                        (hazard_table.end >= start - ONSET_BEFORE)]


def explain(windows, hazard_table):
    labels, kinds = [], []
    for row in windows.itertuples():
        hits = onset_hits(row.terminal, row.start, hazard_table)
        weather, context = hits[hits.kind == "weather"], hits[hits.kind == "context"]
        kinds.append("weather" if len(weather) else "context" if len(context) else "unexplained")
        labels.append("; ".join(dict.fromkeys((weather if len(weather) else context).label)) if len(hits) else "")
    return kinds, labels


def chance_match_rate(windows, hazard_table, cargoes, draws=2000, seed=7):
    """Share of randomly placed windows that would match a weather hazard by coincidence."""
    rng = np.random.default_rng(seed)
    weather = hazard_table[hazard_table.kind == "weather"]
    rates = []
    for row in windows.itertuples():
        first = cargoes[cargoes.terminal == row.terminal].date.min() + pd.Timedelta(days=WARMUP_DAYS)
        last = cargoes[cargoes.terminal == row.terminal].date.max() - pd.Timedelta(days=row.days)
        span = max(1, (last - first).days)
        starts = first + pd.to_timedelta(rng.integers(0, span, draws), unit="D")
        mine = weather[weather.terminal == row.terminal]
        hit = [((mine.start <= s + ONSET_AFTER) & (mine.end >= s - ONSET_BEFORE)).any() for s in starts]
        rates.append(np.mean(hit))
    return float(np.mean(rates)) if rates else float("nan")


def hazard_response(windows, hazard_table):
    """For each weather hazard, did a low-output window overlap it?"""
    weather = hazard_table[hazard_table.kind == "weather"].copy()
    # Same onset rule as `explain`, so both tables agree.
    weather["followed_by_low_output"] = [
        bool(((windows.terminal == h.terminal) & (windows.start <= h.end + ONSET_BEFORE) &
              (windows.start >= h.start - ONSET_AFTER)).any())
        for h in weather.itertuples()]
    return weather


def analyze(cargoes, hurricanes, storm_events, events, weather, cutoff):
    windows, pace = detect(cargoes, cutoff)
    first = cargoes.groupby("terminal").date.min()
    hazard_table = hazards(hurricanes, storm_events, events, weather)
    hazard_table = hazard_table[hazard_table.apply(lambda h: h.end >= first.get(h.terminal, pd.Timestamp.max)
                                                   + pd.Timedelta(days=WARMUP_DAYS), axis=1)]
    windows["lost_bcf"] = lost_volume(windows, pace)
    windows["explanation"], windows["matched"] = explain(windows, hazard_table)
    windows = check(windows.reset_index(drop=True), "gold.low_output_windows",
                    ["terminal", "start", "end", "days", "detector", "lost_bcf", "explanation"],
                    ranges={"days": (1, 365), "lost_bcf": (0, 200)}, min_rows=0)
    response = hazard_response(windows, hazard_table)
    observed = float((windows.explanation == "weather").mean()) if len(windows) else float("nan")
    chance = chance_match_rate(windows, hazard_table, cargoes)
    summary = (windows.groupby(["terminal", "explanation"]).agg(windows=("days", "size"), days=("days", "sum"),
               lost_bcf=("lost_bcf", "sum")).reset_index())
    stats = {"windows": len(windows), "weather_match_rate": observed, "chance_match_rate": chance,
             "weather_hazards": int((hazard_table.kind == "weather").sum()),
             "hazards_followed_by_low_output": int(response.followed_by_low_output.sum())}
    return windows, pace, response, summary, stats
