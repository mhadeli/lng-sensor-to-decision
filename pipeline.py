"""Build the Calcasieu Pass export dashboard from one public DOE workbook."""
import argparse
import subprocess
import sys

from common import ROOT, write_json
from lng.data import OUTPUT, prepare
from lng.forecast import analyze
from lng.dashboard import render


def main(offline=False):
    print('Reading and validating the DOE workbook...', flush=True)
    transactions, monthly, source = prepare(offline)
    print('Comparing simple forecasts using chronological holdouts...', flush=True)
    forecast, backtest, monthly = analyze(monthly)
    monthly.to_csv(OUTPUT / 'monthly_exports.csv', index=False)
    backtest.to_csv(OUTPUT / 'backtest.csv', index=False)
    write_json(OUTPUT / 'forecast.json', forecast)
    subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_lng.py', '-v'], cwd=ROOT, check=True)
    render(transactions, monthly, forecast, backtest, source)
    print(f"Complete. Open {ROOT / 'site' / 'index.html'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--offline', action='store_true', help='Require the cached, checksum-verified workbook')
    main(parser.parse_args().offline)
