"""Extract per-project LNG volumes and realized prices from Venture Global's 10-Q/10-K filings."""
import gzip
import html
import re

import pandas as pd

from vg import bronze

PROJECTS = {"Calcasieu Project": "Calcasieu Pass", "Plaquemines Project": "Plaquemines"}
NUMBER = r"([\d.,]+|—)"


def filing_text(key):
    raw = gzip.open(bronze.path(key), "rt", errors="ignore").read()
    raw = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", raw, flags=re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", raw)))


def value(pattern, block):
    match = re.search(pattern, block)
    if not match:
        return None, None
    # In SEC tables an em dash means zero.
    return tuple(0.0 if v == "—" else float(v.replace(",", "")) for v in match.groups())


def parse(key, text, period_end):
    """Return rows for the current period and the prior-year comparison column."""
    rows = []
    annual = key.startswith("sec_10k")
    for project, terminal in PROJECTS.items():
        heading = rf"{project} (Three months|Years) ended"
        match = re.search(heading + r".{0,900}?Weighted average price of LNG volumes sold (?:\(per MMBtu\) )?(?:\(\d\) )?\$ ?" + NUMBER + r" \$ ?" + NUMBER, text)
        if not match:
            raise ValueError(f"Could not find the {project} volume table in {key}")
        block = match.group(0)
        cargoes = value(r"Cargos " + NUMBER + " " + NUMBER, block)
        exported = value(r"LNG volumes exported Cargos [\d—]+ [\d—]+(?: [\d—]+ [\d—]+)? TBtu " + NUMBER + " " + NUMBER, block)
        sold = value(r"LNG volumes sold \(TBtu\)(?: \(\d\))? " + NUMBER + " " + NUMBER, block)
        price = (float(match.group(2).replace(",", "")) if match.group(2) != "—" else None,
                 float(match.group(3).replace(",", "")) if match.group(3) != "—" else None)
        for column, end in enumerate([pd.Timestamp(period_end), pd.Timestamp(period_end) - pd.DateOffset(years=1)]):
            rows.append({"terminal": terminal, "period": "year" if annual else "quarter",
                         "period_end": end.date().isoformat(), "cargoes_exported": cargoes[column],
                         "tbtu_exported": exported[column], "tbtu_sold": sold[column],
                         "price_per_mmbtu": price[column], "filing": key, "filed_for": period_end,
                         "column": "current" if column == 0 else "prior_year"})
    return rows


def quarterly():
    """Quarterly table; the latest filing wins for each restated field; Q4 = year minus Q1–Q3."""
    rows = []
    for key, spec in bronze.SOURCES.items():
        if key.startswith("sec_"):
            rows += parse(key, filing_text(key), spec["period"])
    frame = pd.DataFrame(rows).sort_values("filed_for")
    frame = frame.groupby(["terminal", "period", "period_end"], as_index=False).last()
    quarters = frame[frame.period == "quarter"].copy()
    for _, year in frame[frame.period == "year"].iterrows():
        end = pd.Timestamp(year.period_end)
        q = quarters[(quarters.terminal == year.terminal) & pd.to_datetime(quarters.period_end).dt.year.eq(end.year)]
        if len(q) != 3:
            continue
        row = {"terminal": year.terminal, "period": "quarter", "period_end": end.date().isoformat(),
               "filing": year.filing + " minus Q1–Q3", "column": "derived"}
        for field in ["cargoes_exported", "tbtu_exported", "tbtu_sold"]:
            row[field] = None if pd.isna(year[field]) or q[field].isna().any() else year[field] - q[field].sum()
        known = q.tbtu_sold.notna().all() and pd.notna(year.tbtu_sold) and row["tbtu_sold"]
        row["price_per_mmbtu"] = ((year.tbtu_sold * year.price_per_mmbtu - (q.tbtu_sold * q.price_per_mmbtu).sum())
                                  / row["tbtu_sold"]) if known else None
        quarters = pd.concat([quarters, pd.DataFrame([row])], ignore_index=True)
    quarters["quarter"] = pd.PeriodIndex(pd.to_datetime(quarters.period_end), freq="Q").astype(str)
    quarters = quarters[quarters.tbtu_sold.fillna(0) > 0]
    return quarters.sort_values(["terminal", "quarter"]).reset_index(drop=True).drop(columns=["period", "filed_for"])
