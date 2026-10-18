"""Reduce one month of NOAA MarineCadastre AIS to the ships that load at Venture Global terminals.

The raw national files are ~350 MB per day, so they are streamed, filtered and discarded.
Only the reduced extract and the upstream file hashes are kept in bronze/.

    python -m vg.ais --month 2023-10
"""
import argparse
import gzip
import hashlib
import io
import tempfile
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import pandas as pd
import requests

from vg.bronze import register_extracts
from vg.common import BRONZE, vessel_key, write_json

BASE = "https://coast.noaa.gov/htdata/CMSP/AISDataHandler/{year}/AIS_{day}.zip"
GULF = dict(lat=(17.0, 31.0), lon=(-98.0, -80.0))
# Large ships near the Calcasieu Pass terminal are kept even if their AIS name differs from DOE.
CALCASIEU_BOX = dict(lat=(29.3, 29.85), lon=(-93.6, -93.0))
COLUMNS = ["MMSI", "BaseDateTime", "LAT", "LON", "SOG", "COG", "Heading", "VesselName",
           "IMO", "VesselType", "Status", "Length", "Draft"]
RESAMPLE = "10min"


def doe_vessels():
    raw = pd.read_excel(BRONZE / "doe_lng_exports_2026_07.xlsx", sheet_name="By Vessel and ISO Container")
    rows = raw[raw.Companies.str.contains("Venture Global", na=False)]
    return set(rows.Tanker.dropna().map(vessel_key))


def reduce_day(day, names):
    url = BASE.format(year=day[:4], day=day.replace("-", "_"))
    digest, size, kept = hashlib.sha256(), 0, []
    with tempfile.TemporaryFile() as handle:
        with requests.get(url, stream=True, timeout=(30, 600), headers={"User-Agent": "Mozilla/5.0"}) as response:
            response.raise_for_status()
            for block in response.iter_content(1 << 20):
                digest.update(block)
                size += len(block)
                handle.write(block)
        handle.seek(0)
        with zipfile.ZipFile(handle) as archive, archive.open(archive.namelist()[0]) as csv:
            for chunk in pd.read_csv(io.TextIOWrapper(csv), usecols=COLUMNS, chunksize=1_000_000):
                gulf = chunk.LAT.between(*GULF["lat"]) & chunk.LON.between(*GULF["lon"])
                named = chunk.VesselName.map(vessel_key).isin(names)
                near = (chunk.LAT.between(*CALCASIEU_BOX["lat"]) & chunk.LON.between(*CALCASIEU_BOX["lon"])
                        & (chunk.Length >= 250))
                part = chunk.loc[gulf & (named | near)].copy()
                part["match"] = named[part.index].map({True: "doe_name", False: "large_ship_near_terminal"})
                kept.append(part)
    frame = pd.concat(kept, ignore_index=True)
    frame["BaseDateTime"] = pd.to_datetime(frame.BaseDateTime)
    frame["slot"] = frame.BaseDateTime.dt.floor(RESAMPLE)
    frame = frame.sort_values("BaseDateTime").drop_duplicates(["MMSI", "slot"]).drop(columns="slot")
    print(f"{day}: kept {len(frame):,} rows", flush=True)
    return frame, {"url": url, "sha256": digest.hexdigest(), "bytes": size}


def build(month, workers=3):
    days = pd.period_range(f"{month}-01", periods=pd.Period(month).days_in_month, freq="D").astype(str)
    names = doe_vessels()
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(lambda d: reduce_day(d, names), days))
    frame = pd.concat([r[0] for r in results]).sort_values(["MMSI", "BaseDateTime"])
    out = BRONZE / "ais" / f"ais_{month.replace('-', '_')}_vg.csv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out, "wt", compresslevel=9) as handle:
        frame.to_csv(handle, index=False)
    write_json(out.with_name(out.name.replace(".csv.gz", ".provenance.json")), {
        "built_at": datetime.now(timezone.utc).isoformat(), "resample": RESAMPLE,
        "filters": {"gulf_box": GULF, "calcasieu_box": CALCASIEU_BOX, "doe_vessel_names": len(names)},
        "rows": len(frame), "upstream": [r[1] for r in results]})
    register_extracts()
    print(f"Wrote {out} ({len(frame):,} rows) and updated bronze/manifest.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--month", default="2023-10")
    build(parser.parse_args().month)
