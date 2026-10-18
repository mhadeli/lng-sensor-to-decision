"""Cache five public machine tables and a pinned DOE LNG export workbook."""
import argparse
import json
from datetime import datetime, timezone

import requests

from common import RAW, file_hash, write_json

PDM_BASE = ("https://raw.githubusercontent.com/microsoft/sqlworkshops/master/"
            "SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server/data/")
DOE_URL = ("https://www.energy.gov/sites/default/files/2026-09/"
           "3.%20U.S.%20LNG%20Exports%20and%20Re-Exports%20Details%20%28Jan%202016%20-%20Jul%202026%29.xlsx")
SOURCES = {f"pdm/PdM_{name}.csv": PDM_BASE + f"PdM_{name}.csv"
           for name in ("telemetry", "errors", "maint", "failures", "machines")}
SOURCES["doe_lng_exports_2026_07.xlsx"] = DOE_URL


def main(refresh=False):
    """Never replace a raw file: refresh verifies against the pinned snapshot."""
    manifest = RAW / "manifest.json"
    previous = {item["file"]: item for item in json.loads(manifest.read_text())} if manifest.exists() else {}
    records = []
    for relative, url in SOURCES.items():
        path = RAW / relative
        cached = path.exists() and path.stat().st_size > 0
        if cached and relative in previous and file_hash(path) != previous[relative]["sha256"]:
            raise ValueError(f"Cached source changed: {relative}. Review it before rebuilding.")
        if not cached or refresh:
            print(f"Downloading {relative}", flush=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".part")
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with temporary.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        handle.write(chunk)
            if cached and file_hash(temporary) != file_hash(path):
                raise ValueError(f"Source changed: {relative}. Review .part and version the source before rebuilding.")
            temporary.replace(path)
        records.append({"file": relative, "source_url": url, "sha256": file_hash(path),
                        "bytes": path.stat().st_size, "schema_version": 1,
                        "verified_at": datetime.now(timezone.utc).isoformat(),
                        "access": "cached" if cached and not refresh else "downloaded"})
        print(f"Ready: {relative}", flush=True)
    write_json(RAW / "manifest.json", records)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Verify pinned sources; reject upstream changes")
    main(parser.parse_args().refresh)
