"""Build the Venture Global LNG analytics site from pinned public data.

    python pipeline.py

Steps: unit tests → verify bronze checksums → silver tables → gold analyses → static site in site/.
To accept newer upstream data, run `python -m vg.bronze --refresh` first and review the manifest diff.
"""
import argparse
import subprocess
import sys
from datetime import datetime, timezone
from time import perf_counter

from vg import gold, site
from vg.common import GOLD, ROOT, SITE, file_hash, write_json


def main():
    timings, start = [], perf_counter()
    print("Running unit tests...", flush=True)
    subprocess.run([sys.executable, "-m", "unittest", "discover", "-s", "tests"], cwd=ROOT, check=True)
    print("Building bronze → silver → gold...", flush=True)
    payloads, downloads = gold.build(timings)
    print("Rendering site...", flush=True)
    site.render(payloads, downloads)
    write_json(GOLD / "run.json", {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "seconds": round(perf_counter() - start, 1),
        "steps": timings, "code": {str(p.relative_to(ROOT)): file_hash(p) for p in sorted((ROOT / "vg").glob("*.py"))}})
    print(f"Complete in {perf_counter() - start:.0f}s. Open {SITE / 'index.html'}")


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    main()
