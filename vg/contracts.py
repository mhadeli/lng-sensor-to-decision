"""Small, explicit data contracts. Every silver and gold table passes through `check`."""
import pandas as pd

REPORT = []


class ContractError(ValueError):
    pass


def check(frame, table, columns, unique=None, not_null=None, ranges=None, min_rows=1):
    """Validate a table and record the result; raise on any failure so bad data never reaches gold."""
    failures = []
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        failures.append(f"missing columns {missing}")
    if len(frame) < min_rows:
        failures.append(f"{len(frame)} rows (< {min_rows})")
    if not missing:
        for column in not_null or columns:
            nulls = int(frame[column].isna().sum())
            if nulls:
                failures.append(f"{nulls} null values in {column}")
        if unique and frame.duplicated(unique).any():
            failures.append(f"duplicate keys on {unique}: {int(frame.duplicated(unique).sum())}")
        for column, (low, high) in (ranges or {}).items():
            values = frame[column].dropna()
            bad = int(((values < low) | (values > high)).sum())
            if bad:
                failures.append(f"{bad} values of {column} outside [{low}, {high}]")
    REPORT.append({"table": table, "rows": len(frame), "checks": 2 + len(not_null or columns) + bool(unique) + len(ranges or {}),
                   "status": "failed" if failures else "passed", "detail": "; ".join(failures)})
    if failures:
        raise ContractError(f"{table}: " + "; ".join(failures))
    return frame


def note(table, status, detail, rows=None):
    """Record a non-fatal finding (e.g. a reconciliation difference) in the quality report."""
    REPORT.append({"table": table, "rows": rows, "checks": 1, "status": status, "detail": detail})


def report():
    return pd.DataFrame(REPORT)
