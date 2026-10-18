"""Shared paths and small storage helpers for the local workflow."""
import hashlib
import json
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
SILVER = ROOT / "data" / "silver"
GOLD = ROOT / "data" / "gold"
ARTIFACTS = ROOT / "data" / "models"
SITE = ROOT / "site"
DATABASE = ROOT / "data" / "operations.duckdb"
MODELS = ("baseline", "mlp", "xgboost")
NAMES = {"baseline": "Simple rule", "mlp": "Neural network (MLP)", "xgboost": "XGBoost"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_parquet(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with duckdb.connect() as con:
        con.register("output", frame)
        con.execute("COPY output TO ? (FORMAT PARQUET)", [str(path)])


def read_parquet(path):
    with duckdb.connect() as con:
        return con.execute("SELECT * FROM read_parquet(?)", [str(path)]).df()
