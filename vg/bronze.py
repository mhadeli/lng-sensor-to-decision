"""Bronze layer: pinned raw snapshots of every public source, verified by checksum.

Snapshots live in bronze/ (committed), so a normal build never touches the network and
always reproduces the same numbers. Accepting new upstream data is an explicit step:

    python -m vg.bronze --refresh            # re-download every source
    python -m vg.bronze --refresh fred_asia  # or just one

Review the changed hashes in bronze/manifest.json (git diff) before committing them.
The AIS extract is built separately with `python -m vg.ais` because it streams ~10 GB.
"""
import argparse
import gzip
import io
import time
from datetime import datetime, timezone

import pandas as pd
import requests

from vg.common import BRONZE, file_hash, write_json

MANIFEST = BRONZE / "manifest.json"
SEC_AGENT = "lng-sensor-to-decision research admin@example.com"
SEC_BASE = "https://www.sec.gov/Archives/edgar/data/2007855/"
METEO_DAILY = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               "&start_date=2022-01-01&end_date=2026-07-31&timezone=America/Chicago&daily="
               "temperature_2m_mean,temperature_2m_max,temperature_2m_min,wind_speed_10m_max,"
               "wind_gusts_10m_max,precipitation_sum")
GRID = [(lat, lon) for lat in range(20, 31) for lon in range(-97, -80)]
GRID_QUERY = "latitude={}&longitude={}".format(",".join(str(p[0]) for p in GRID), ",".join(str(p[1]) for p in GRID))
STORM_FILES = ["d2022_c20260625", "d2023_c20260323", "d2024_c20260728", "d2025_c20260819", "d2026_c20260918"]
STORM_BASE = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/StormEvents_details-ftp_v1.0_{}.csv.gz"
# NWS zones and counties that cover the two terminals and their ship channels.
STORM_AREAS = ["CAMERON", "CALCASIEU", "PLAQUEMINES", "SOUTHWEST PASS", "BOOTHVILLE", "STAKE ISLAND"]

SOURCES = {
    "doe_exports": dict(
        file="doe_lng_exports_2026_07.xlsx", license="U.S. government work (public domain)",
        url="https://www.energy.gov/sites/default/files/2026-09/3.%20U.S.%20LNG%20Exports%20and%20Re-Exports%20Details%20%28Jan%202016%20-%20Jul%202026%29.xlsx",
        description="DOE LNG export allocations by vessel, Jan 2016–Jul 2026 vintage"),
    "eia_henry_hub": dict(
        file="eia_henry_hub_daily.xls", url="https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDd.xls",
        license="U.S. government work (public domain)", description="Henry Hub natural gas spot price, daily"),
    "fred_europe_gas": dict(
        file="fred_PNGASEUUSDM.csv", url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=PNGASEUUSDM",
        license="IMF Primary Commodity Prices via FRED; attribution required",
        description="Global price of natural gas, EU (TTF-linked proxy), monthly USD/MMBtu"),
    "fred_asia_lng": dict(
        file="fred_PNGASJPUSDM.csv", url="https://fred.stlouisfed.org/graph/fredgraph.csv?id=PNGASJPUSDM",
        license="IMF Primary Commodity Prices via FRED; attribution required",
        description="Global price of LNG, Asia (Japan import price, mostly oil-linked), monthly USD/MMBtu"),
    "weather_calcasieu": dict(
        file="weather_calcasieu_daily.json", url=METEO_DAILY.format(lat=29.773, lon=-93.334),
        license="Open-Meteo (ERA5), CC BY 4.0", description="Daily weather at Calcasieu Pass, 2022–Jul 2026"),
    "weather_plaquemines": dict(
        file="weather_plaquemines_daily.json", url=METEO_DAILY.format(lat=29.600, lon=-89.886),
        license="Open-Meteo (ERA5), CC BY 4.0", description="Daily weather at Plaquemines, 2022–Jul 2026"),
    "gulf_waves_2023_10": dict(
        file="gulf_waves_2023_10.json.gz", gzip=True,
        url="https://marine-api.open-meteo.com/v1/marine?" + GRID_QUERY +
            "&start_date=2023-10-01&end_date=2023-10-31&hourly=wave_height&timezone=UTC",
        license="Open-Meteo marine, CC BY 4.0", description="Hourly wave height on a 1° Gulf grid, Oct 2023"),
    "gulf_wind_2023_10": dict(
        file="gulf_wind_2023_10.json.gz", gzip=True,
        url="https://archive-api.open-meteo.com/v1/archive?" + GRID_QUERY +
            "&start_date=2023-10-01&end_date=2023-10-31&hourly=wind_speed_10m&wind_speed_unit=kn&timezone=UTC",
        license="Open-Meteo (ERA5), CC BY 4.0", description="Hourly 10 m wind on a 1° Gulf grid, Oct 2023"),
    "hurdat2": dict(
        file="hurdat2_1851_2025.txt.gz", gzip=True,
        url="https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt",
        license="U.S. government work (public domain)", description="NHC Atlantic hurricane best tracks through 2025"),
    "ncei_storm_events": dict(
        file="ncei_storm_events_vg.csv.gz", gzip=True, builder="storm_events",
        url=[STORM_BASE.format(f) for f in STORM_FILES],
        license="U.S. government work (public domain)",
        description="NOAA Storm Events 2022–2026, filtered to the terminal parishes and adjacent zones"),
}
SEC_FILINGS = {  # accession folder, document, period
    "sec_10q_2025q1": ("000200785525000018", "vg-20250331.htm", "2025-03-31"),
    "sec_10q_2025q2": ("000200785525000042", "vg-20250630.htm", "2025-06-30"),
    "sec_10q_2025q3": ("000200785525000066", "vg-20250930.htm", "2025-09-30"),
    "sec_10k_2025": ("000200785526000013", "vg-20251231.htm", "2025-12-31"),
    "sec_10q_2026q1": ("000200785526000040", "vg-20260331.htm", "2026-03-31"),
    "sec_10q_2026q2": ("000200785526000062", "vg-20260630.htm", "2026-06-30"),
}
for key, (folder, document, period) in SEC_FILINGS.items():
    SOURCES[key] = dict(file=f"sec/{document}.gz", gzip=True, url=SEC_BASE + f"{folder}/{document}", period=period,
                        license="U.S. SEC public filing", description=f"Venture Global {key[4:7].upper()} for {period}")
EXTRACTS = {"ais_2023_10": dict(file="ais/ais_2023_10_vg.csv.gz", builder="python -m vg.ais --month 2023-10",
                                license="NOAA MarineCadastre AIS (public domain)",
                                description="AIS positions of ships that load at Venture Global terminals, Oct 2023")}


def fetch(url, attempts=3):
    headers = {"User-Agent": SEC_AGENT if "sec.gov" in url else "lng-sensor-to-decision/1.0 (research)"}
    for attempt in range(attempts):
        try:
            response = requests.get(url, headers=headers, timeout=(30, 120))
            response.raise_for_status()
            return response.content
        except requests.RequestException:
            if attempt == attempts - 1:
                raise
            time.sleep(5 * (attempt + 1))


def storm_events(urls):
    frames = []
    for url in urls:
        raw = pd.read_csv(io.BytesIO(fetch(url)), compression="gzip", low_memory=False)
        keep = (raw.STATE.isin(["LOUISIANA", "GULF OF MEXICO"]) &
                raw.CZ_NAME.str.upper().str.contains("|".join(STORM_AREAS), na=False))
        frames.append(raw.loc[keep])
    return pd.concat(frames).to_csv(index=False).encode()


def deterministic_gzip(content):
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0, compresslevel=9) as handle:
        handle.write(content)
    return buffer.getvalue()


def refresh(ids=None):
    manifest = load_manifest()
    for key in ids or SOURCES:
        if key not in SOURCES:
            raise SystemExit(f"Unknown source {key}. Choose from: {', '.join(SOURCES)}")
        spec = SOURCES[key]
        content = storm_events(spec["url"]) if spec.get("builder") == "storm_events" else fetch(spec["url"])
        if spec.get("gzip"):
            content = deterministic_gzip(content)
        path = BRONZE / spec["file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        digest = file_hash(path)
        old = manifest.get(key, {}).get("sha256")
        print(f"{key:22s} {'unchanged' if old == digest else 'UPDATED  ' if old else 'new      '} {digest[:12]}")
        manifest[key] = {**{k: v for k, v in spec.items() if k != "gzip"}, "sha256": digest,
                         "bytes": path.stat().st_size, "retrieved_at": datetime.now(timezone.utc).isoformat()}
        write_json(MANIFEST, dict(sorted(manifest.items())))
    register_extracts()


def register_extracts():
    """Record locally built extracts (e.g. AIS) in the manifest."""
    manifest = load_manifest()
    for key, spec in EXTRACTS.items():
        path = BRONZE / spec["file"]
        if path.exists():
            manifest[key] = {**spec, "sha256": file_hash(path), "bytes": path.stat().st_size,
                             "provenance": spec["file"].replace(".csv.gz", ".provenance.json")}
    write_json(MANIFEST, dict(sorted(manifest.items())))


def load_manifest():
    return __import__("json").loads(MANIFEST.read_text()) if MANIFEST.exists() else {}


def verify():
    """Fail loudly if any pinned snapshot is missing or differs from the reviewed manifest."""
    manifest = load_manifest()
    missing = [k for k in [*SOURCES, *EXTRACTS] if k not in manifest]
    if missing:
        raise ValueError(f"Sources not in bronze/manifest.json: {missing}. Run python -m vg.bronze --refresh.")
    for key, entry in manifest.items():
        path = BRONZE / entry["file"]
        if not path.exists() or file_hash(path) != entry["sha256"]:
            raise ValueError(f"Bronze snapshot {entry['file']} is missing or changed. Review and refresh it explicitly.")
    return manifest


def path(key):
    return BRONZE / (SOURCES.get(key) or EXTRACTS[key])["file"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", nargs="*", metavar="SOURCE", help="Download sources (all if none named)")
    args = parser.parse_args()
    if args.refresh is not None:
        refresh(args.refresh or None)
    verify()
    print("All bronze snapshots match bronze/manifest.json")
