"""Run the complete local demonstration and record a small execution audit."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from time import perf_counter

from common import GOLD, RAW, ROOT, file_hash, write_json


def main(offline=False):
    if offline:
        manifest = json.loads((RAW / "manifest.json").read_text())
        for source in manifest:
            if file_hash(RAW / source["file"]) != source["sha256"]:
                raise ValueError(f"Cached source checksum changed: {source['file']}")
    stages = [] if offline else [("download", ["download.py"])]
    stages += [(name, [f"{name}.py"]) for name in ("prepare", "model", "forecasting", "replay")]
    stages += [("tests", ["-m", "unittest", "discover", "-s", "tests", "-v"]), ("dashboard", ["dashboard.py"])]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_dir = ROOT / "data" / "runs" / run_id
    log_dir.mkdir(parents=True)
    audit = {"run_id": run_id, "status": "running", "offline": offline, "stages": [],
             "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in
                               sorted(ROOT.glob("*.py")) + sorted((ROOT / "sql").glob("*.sql"))}}
    for name, arguments in stages:
        print(f"Running {name}...", flush=True)
        start = perf_counter()
        with (log_dir / f"{name}.log").open("w") as log:
            result = subprocess.run([sys.executable, "-u", *arguments], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        audit["stages"].append({"name": name, "seconds": round(perf_counter() - start, 2), "exit_code": result.returncode})
        if result.returncode:
            audit["status"] = "failed"
            write_json(log_dir / "run.json", audit)
            print((log_dir / f"{name}.log").read_text())
            raise SystemExit(f"Stage {name} failed. Details: {log_dir}")
        write_json(log_dir / "run.json", audit)
    audit["status"] = "complete"
    audit["raw_inputs"] = json.loads((RAW / "manifest.json").read_text())
    write_json(log_dir / "run.json", audit)
    write_json(GOLD / "latest_run.json", audit)
    print(f"Complete. Open {ROOT / 'site' / 'index.html'}\nRun logs: {log_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true", help="Verify cached input hashes and run without downloading")
    main(parser.parse_args().offline)
