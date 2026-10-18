"""Shared paths and small storage helpers."""
import hashlib
import json
import re
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
BRONZE = ROOT / "bronze"
REFERENCE = ROOT / "reference"
DATA = ROOT / "data"
SILVER = DATA / "silver"
GOLD = DATA / "gold"
WAREHOUSE = DATA / "warehouse.duckdb"
SITE = ROOT / "site"

# Approximate conversions used throughout; methodology docs explain the ranges.
BCF_PER_MTPA_YEAR = 48.0      # 1 million tonnes of LNG ≈ 48 Bcf of gas
TBTU_PER_BCF = 1.038          # pipeline-quality gas ≈ 1,038 Btu per cubic foot


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def vessel_key(name):
    """Normalize vessel names so DOE and AIS spellings can be matched."""
    return re.sub(r"[^A-Z0-9]+", " ", str(name).upper()).strip()


def write_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.register("output", frame)
        con.execute("COPY output TO ? (FORMAT PARQUET)", [str(path)])


def read_parquet(path):
    with duckdb.connect() as con:
        return con.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()
