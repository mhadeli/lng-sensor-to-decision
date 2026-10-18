"""Check allocation preservation, calendar handling, and temporal separation."""
import unittest

import numpy as np
import pandas as pd

from lng.data import aggregate, clean, TERMINAL
from lng.forecast import analyze, predict


class ExportTests(unittest.TestCase):
    def raw(self):
        return pd.DataFrame({'Arrival/Departure Date': ['2026-07-02'] * 2,
            'Point of Entry or Exit': [TERMINAL] * 2, 'Volume (MMCF)': [1000, 2000],
            'Tanker': ['Example LNG'] * 2, 'Country': ['France', 'Spain'],
            'Activity': ['Exports'] * 2, 'Mode of Transport': ['Vessel'] * 2, 'Gas Type': ['LNG'] * 2})

    def test_split_allocations_preserved(self):
        rows = clean(self.raw())
        result = aggregate(rows)
        self.assertEqual(len(rows), 2)
        self.assertEqual(result.iloc[0].shipments, 1)
        self.assertEqual(result.iloc[0].volume_bcf, 3)
        self.assertEqual(result.iloc[0].destinations, 2)

    def test_other_terminal_excluded(self):
        raw = self.raw()
        raw.loc[1, 'Point of Entry or Exit'] = 'Sabine Pass, LA'
        self.assertEqual(len(clean(raw)), 1)

    def test_missing_month_is_not_zero(self):
        raw = self.raw()
        raw['Arrival/Departure Date'] = ['2026-05-01', '2026-07-01']
        with self.assertRaisesRegex(ValueError, 'Missing'):
            aggregate(clean(raw))

    def test_schema_change_fails(self):
        with self.assertRaisesRegex(ValueError, 'Missing DOE columns'):
            clean(self.raw().drop(columns='Country'))

    def series(self):
        months = pd.period_range('2022-03', periods=53, freq='M')
        return pd.DataFrame({'month': months.astype(str), 'volume_bcf': months.days_in_month * 1.5})

    def test_leap_year_calendar(self):
        history = pd.DataFrame({'month': ['2024-01'], 'volume_bcf': [31.]})
        self.assertEqual(predict(history, '2024-02', 'last_month'), 29.)

    def test_constant_daily_rate(self):
        report, test, _ = analyze(self.series())
        self.assertAlmostEqual(report['mae_bcf'], 0.)
        self.assertEqual(len(test), 12)
        self.assertEqual(report['target_month'], '2026-08')
        self.assertGreaterEqual(report['lower_bcf'], 0)

    def test_future_values_cannot_change_selection_or_previous_prediction(self):
        series = self.series()
        first, test, _ = analyze(series)
        series.loc[len(series) - 1, 'volume_bcf'] = 9999.
        changed, changed_test, _ = analyze(series)
        self.assertEqual(first['method'], changed['method'])
        np.testing.assert_array_equal(test.prediction, changed_test.prediction)
        np.testing.assert_array_equal(test.lower, changed_test.lower)
        self.assertNotEqual(first['mae_bcf'], changed['mae_bcf'])


if __name__ == '__main__':
    unittest.main()
