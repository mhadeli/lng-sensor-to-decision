"""Build an offline dashboard with plain-language labels and downloadable evidence."""
import json
import shutil
from pathlib import Path

from plotly.offline import get_plotlyjs

from common import ROOT, SITE
from lng.data import OUTPUT


def render(transactions, monthly, forecast, test, source):
    SITE.mkdir(exist_ok=True)
    payload = {'monthly': monthly.fillna(0).to_dict('records'), 'forecast': forecast,
               'test': test.to_dict('records'), 'source': source,
               'destinations': transactions.groupby(['month', 'destination']).volume_bcf.sum().reset_index().to_dict('records')}
    template = (Path(__file__).parent / 'dashboard.html').read_text()
    (SITE / 'index.html').write_text(template.replace('__DATA__', json.dumps(payload, allow_nan=False).replace('<', '\\u003c')))
    shutil.copyfile(Path(__file__).parent / 'dashboard.css', SITE / 'lng.css')
    shutil.copyfile(Path(__file__).parent / 'dashboard.js', SITE / 'lng.js')
    (SITE / 'plotly.min.js').write_text(get_plotlyjs())
    for name in ['monthly_exports.csv', 'transactions.csv', 'backtest.csv', 'forecast.json', 'source.json']:
        shutil.copyfile(OUTPUT / name, SITE / name)
    shutil.copyfile(ROOT / 'docs' / 'lng_methodology.md', SITE / 'lng_methodology.md')
