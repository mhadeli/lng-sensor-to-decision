"""Read one pinned DOE source, validate it, and retain terminal allocations."""
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

from common import RAW, ROOT, file_hash, write_json

OUTPUT = ROOT / 'data' / 'lng'
SOURCE = RAW / 'doe_lng_exports_2026_07.xlsx'
DOE_URL = ('https://www.energy.gov/sites/default/files/2026-09/'
           '3.%20U.S.%20LNG%20Exports%20and%20Re-Exports%20Details%20%28Jan%202016%20-%20Jul%202026%29.xlsx')
TERMINAL = 'Cameron (Calcasieu Pass), LA'
CUTOFF = '2026-07'
EXPECTED_HASH = '970f7d583d6a051b6aa4730ddac1ca5bf1bb19a9c24e0a78906f3b1c54eb8889'
COLUMNS = {'Arrival/Departure Date': 'date', 'Point of Entry or Exit': 'terminal',
           'Volume (MMCF)': 'volume_mmcf', 'Tanker': 'vessel', 'Country': 'destination'}


def acquire(offline=False):
    if not SOURCE.exists():
        if offline:
            raise FileNotFoundError('DOE workbook missing. Run python pipeline.py once online.')
        SOURCE.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(DOE_URL, timeout=(30, 180))
        response.raise_for_status()
        temporary = SOURCE.with_suffix('.part')
        temporary.write_bytes(response.content)
        if file_hash(temporary) != EXPECTED_HASH:
            raise ValueError('DOE source changed. Review the new workbook before updating the pinned hash.')
        temporary.replace(SOURCE)
    if file_hash(SOURCE) != EXPECTED_HASH:
        raise ValueError('Cached DOE workbook checksum does not match the reviewed source.')
    return SOURCE


def clean(raw):
    required = set(COLUMNS) | {'Activity', 'Mode of Transport', 'Gas Type'}
    if not required.issubset(raw.columns):
        raise ValueError(f'Missing DOE columns: {sorted(required - set(raw.columns))}')
    frame = raw.rename(columns=COLUMNS).copy()
    frame['source_row'] = np.arange(2, len(frame) + 2)
    frame = frame.loc[(frame.terminal == TERMINAL) & (frame.Activity == 'Exports') &
                      (frame['Mode of Transport'] == 'Vessel') & (frame['Gas Type'] == 'LNG')].copy()
    if frame.empty:
        raise ValueError('No Calcasieu Pass vessel exports found.')
    frame['date'] = pd.to_datetime(frame.date, errors='coerce')
    frame['volume_mmcf'] = pd.to_numeric(frame.volume_mmcf, errors='coerce')
    invalid = (frame.date.isna() | ~np.isfinite(frame.volume_mmcf) | (frame.volume_mmcf <= 0) |
               frame.vessel.isna() | frame.destination.isna())
    if invalid.any():
        OUTPUT.mkdir(parents=True, exist_ok=True)
        frame.loc[invalid].to_csv(OUTPUT / 'invalid_records.csv', index=False)
        raise ValueError('Invalid terminal records; inspect data/lng/invalid_records.csv before proceeding.')
    # Split allocations are legitimate. Never deduplicate by vessel or date alone.
    frame['month'] = frame.date.dt.strftime('%Y-%m')
    frame['volume_bcf'] = frame.volume_mmcf / 1000
    frame['shipment_key'] = frame.date.dt.strftime('%Y-%m-%d') + '|' + frame.vessel.str.strip()
    return frame.sort_values(['date', 'source_row']).reset_index(drop=True)


def aggregate(frame):
    monthly = frame.groupby('month').agg(volume_bcf=('volume_bcf', 'sum'),
        shipments=('shipment_key', 'nunique'), destinations=('destination', 'nunique')).reset_index()
    expected = pd.period_range(monthly.month.iloc[0], CUTOFF, freq='M').astype(str).tolist()
    if monthly.month.tolist() != expected:
        raise ValueError('Missing or unexpected reporting months. Do not substitute zero exports.')
    return monthly


def prepare(offline=False):
    path = acquire(offline)
    raw = pd.read_excel(path, sheet_name='By Vessel and ISO Container')
    transactions = clean(raw)
    monthly = aggregate(transactions)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(OUTPUT / 'transactions.csv', index=False)
    monthly.to_csv(OUTPUT / 'monthly_exports.csv', index=False)
    import duckdb
    with duckdb.connect(str(OUTPUT / 'exports.duckdb')) as con:
        con.register('clean_transactions', transactions)
        con.register('monthly_summary', monthly)
        con.execute('CREATE OR REPLACE TABLE transactions AS SELECT * FROM clean_transactions')
        con.execute('CREATE OR REPLACE TABLE monthly_exports AS SELECT * FROM monthly_summary')
    source = {'url': DOE_URL, 'sha256': file_hash(path), 'coverage_end': CUTOFF,
              'terminal': TERMINAL, 'source_rows': len(raw), 'terminal_rows': len(transactions),
              'built_at': datetime.now(timezone.utc).isoformat()}
    write_json(OUTPUT / 'source.json', source)
    return transactions, monthly, source
