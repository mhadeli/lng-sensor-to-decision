"""Download the public stand-ins for the plant into data/raw."""

import argparse
import json
import re
import zipfile
from pathlib import Path

import requests
import xlrd

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"

TEP_ZIP = (
    "https://github.com/camaramm/tennessee-eastman-profBraatz/archive/refs/heads/master.zip"
)
EIA_XLS = "https://www.eia.gov/dnav/ng/hist_xls/N9133US2m.xls"
HENRY_HUB_XLS = "https://www.eia.gov/dnav/ng/hist_xls/RNGWHHDm.xls"
NOAA_INDEX = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
WEATHER_URL = (
    "https://www.ncei.noaa.gov/access/services/data/v1"
    "?dataset=global-summary-of-the-month&stations=USW00003937"
    "&startDate=1997-01-01&endDate=2026-12-01&dataTypes=TAVG"
    "&format=csv&includeAttributes=false"
)
EIA_BULK = "https://www.eia.gov/opendata/bulk/NG.zip"
STORM_YEARS = ("2024", "2025", "2026")
PDM_BASE = (
    "https://raw.githubusercontent.com/microsoft/sqlworkshops/master/"
    "SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data/"
)
# Monthly liquefied-gas exports from Gulf terminals, million cubic feet.
GULF_SERIES = {
    "NG.NGM_EPG0_ENG_YSPL-Z00_MMCF.M": "Sabine Pass",
    "NG.NGM_EPG0_ENG_YCAM-Z00_MMCF.M": "Cameron",
    "NG.NGM_EPG0_ENG_YCCPL-Z00_MMCF.M": "Calcasieu Pass",
    "NG.NGM_EPG0_ENG_YPLAQ-Z00_MMCF.M": "Plaquemines",
    "NG.NGM_EPG0_ENG_YCRP-Z00_MMCF.M": "Corpus Christi",
    "NG.NGM_EPG0_ENG_YFPT-Z00_MMCF.M": "Freeport",
    "NG.NGM_EPG0_ENG_YGPT-Z00_MMCF.M": "Golden Pass",
}


def download(url, path, refresh=False):
    if not refresh and path.exists() and path.stat().st_size > 0:
        print(f"already have {path.relative_to(ROOT)}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {path.name}")
    with requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, stream=True, timeout=300) as response:
        response.raise_for_status()
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=256 * 1024):
                if chunk:
                    handle.write(chunk)
        temporary.replace(path)


def save_tep(zip_path):
    """Keep the training runs dXX.dat and the test runs dXX_te.dat."""
    out = RAW / "tep"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "d00_te.dat").exists():
        print("already have data/raw/tep")
        return

    with zipfile.ZipFile(zip_path) as archive:
        for name in archive.namelist():
            file_name = Path(name).name
            if re.fullmatch(r"d\d\d(_te)?\.dat", file_name):
                (out / file_name).write_bytes(archive.read(name))
    print(f"saved {len(list(out.glob('d*.dat')))} Tennessee Eastman files")


def save_monthly_xls(xls_path, csv_path, header, refresh=False):
    if not refresh and csv_path.exists() and csv_path.stat().st_size > 0:
        print(f"already have {csv_path.relative_to(ROOT)}")
        return
    book = xlrd.open_workbook(xls_path)
    sheet = book.sheet_by_name("Data 1")
    lines = [header]
    for row in range(3, sheet.nrows):
        serial = sheet.cell_value(row, 0)
        value = sheet.cell_value(row, 1)
        if serial == "" or value == "":
            continue
        month = xlrd.xldate_as_datetime(serial, book.datemode).strftime("%Y-%m")
        lines.append(f"{month},{value}")
    csv_path.write_text("\n".join(lines) + "\n")
    print(f"saved {csv_path.relative_to(ROOT)} ({len(lines) - 1} months)")


def save_gulf_terminals(zip_path):
    csv_path = RAW / "gulf_lng_exports.csv"
    if csv_path.exists() and csv_path.stat().st_size > 0:
        print("already have data/raw/gulf_lng_exports.csv")
        return
    lines = ["month,terminal,exports_mmcf"]
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("NG.txt") as handle:
            for raw_line in handle:
                row = json.loads(raw_line)
                terminal = GULF_SERIES.get(row.get("series_id"))
                if terminal is None:
                    continue
                for period, value in row.get("data", []):
                    if value is None or len(str(period)) < 6:
                        continue
                    month = f"{period[:4]}-{period[4:6]}"
                    lines.append(f"{month},{terminal},{value}")
    csv_path.write_text("\n".join(lines) + "\n")
    print(f"saved data/raw/gulf_lng_exports.csv ({len(lines) - 1} rows)")


def latest_storm_files():
    """Newest NOAA file for each year in STORM_YEARS."""
    page = requests.get(NOAA_INDEX, timeout=60)
    page.raise_for_status()
    names = sorted(set(re.findall(r"StormEvents_details-ftp_v1\.0_d\d{4}_c\d+\.csv\.gz", page.text)))
    by_year = {re.search(r"_d(\d{4})_", name).group(1): name for name in names}
    return [by_year[year] for year in STORM_YEARS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extras", action="store_true", help="Download the original supporting EDA sources")
    parser.add_argument("--refresh", action="store_true", help="Refresh the monthly EIA export source")
    args = parser.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    tep_zip = RAW / "tep.zip"
    download(TEP_ZIP, tep_zip)
    save_tep(tep_zip)

    xls_path = RAW / "lng_exports.xls"
    download(EIA_XLS, xls_path, refresh=args.refresh)
    save_monthly_xls(xls_path, RAW / "lng_exports.csv", "month,exports_mmcf", refresh=args.refresh)
    if not args.extras:
        return

    price_path = RAW / "henry_hub.xls"
    download(HENRY_HUB_XLS, price_path)
    save_monthly_xls(price_path, RAW / "henry_hub.csv", "month,henry_hub")

    download(WEATHER_URL, RAW / "lake_charles_weather.csv")

    bulk = RAW / "eia_ng.zip"
    download(EIA_BULK, bulk)
    save_gulf_terminals(bulk)

    for name in latest_storm_files():
        download(NOAA_INDEX + name, RAW / "noaa" / name)

    for name in ("PdM_maint.csv", "PdM_failures.csv"):
        download(PDM_BASE + name, RAW / "pdm" / name)


if __name__ == "__main__":
    main()
