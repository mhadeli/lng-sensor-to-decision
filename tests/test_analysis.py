"""Forecast leakage, reliability detection, voyage economics, fleet logic and price-model splits."""
import unittest

import numpy as np
import pandas as pd

from vg import fleet, forecast, price_model, reliability, shipping

BASE = {"speed": 17.0, "port_days": 4.0, "charter_rate": 60000.0, "panama_fee": 600000.0, "suez_fee": 700000.0,
        "boil_off": 0.1, "regas_fee": 0.5}


def schedule(start="2024-01-01", days=400, every=3, gap=None):
    dates = list(pd.date_range(start, periods=days // every, freq=f"{every}D"))
    if gap:
        dates = [d for d in dates if not gap[0] <= d <= gap[1]]
    return pd.DataFrame({"terminal": "Calcasieu Pass", "date": dates, "vessel_key": [f"SHIP {i % 9}" for i in range(len(dates))],
                         "volume_bcf": 3.6})


class ExportForecastTests(unittest.TestCase):
    def series(self):
        months = pd.period_range("2022-03", periods=53, freq="M")
        return pd.DataFrame({"month": months.astype(str), "volume_bcf": months.days_in_month * 1.5})

    def test_leap_year_calendar(self):
        self.assertEqual(forecast.predict(pd.DataFrame({"month": ["2024-01"], "volume_bcf": [31.0]}), "2024-02", "last_month"), 29.0)

    def test_constant_daily_rate_is_exact(self):
        report, test, _ = forecast.analyze(self.series())
        self.assertAlmostEqual(report["mae_bcf"], 0.0)
        self.assertEqual(len(test), 12)

    def test_future_values_cannot_change_earlier_predictions(self):
        series = self.series()
        first, test, _ = forecast.analyze(series)
        series.loc[len(series) - 1, "volume_bcf"] = 9999.0
        changed, changed_test, _ = forecast.analyze(series)
        self.assertEqual(first["method"], changed["method"])
        np.testing.assert_array_equal(test.prediction, changed_test.prediction)


class ReliabilityTests(unittest.TestCase):
    def test_regular_schedule_has_no_windows(self):
        windows, _ = reliability.detect(schedule(), "2025-01")
        self.assertEqual(len(windows), 0)

    def test_long_gap_is_detected(self):
        cargoes = schedule(gap=(pd.Timestamp("2024-08-01"), pd.Timestamp("2024-08-12")))
        windows, _ = reliability.detect(cargoes, "2025-01")
        self.assertEqual(len(windows), 1)
        self.assertIn("cargo gap", windows.detector.iloc[0])
        self.assertLessEqual(windows.start.iloc[0], pd.Timestamp("2024-08-01"))

    def test_hazard_must_coincide_with_onset(self):
        hazards = pd.DataFrame({"terminal": ["Calcasieu Pass"] * 2, "kind": ["weather"] * 2, "hazard": ["cold", "wind"],
                                "label": ["early", "late"], "start": pd.to_datetime(["2024-08-01", "2024-09-20"]),
                                "end": pd.to_datetime(["2024-08-04", "2024-09-22"])})
        windows = pd.DataFrame({"terminal": ["Calcasieu Pass"], "start": pd.to_datetime(["2024-08-02"]), "end": pd.to_datetime(["2024-09-30"])})
        kinds, labels = reliability.explain(windows, hazards)
        self.assertEqual((kinds[0], labels[0]), ("weather", "early"))


class ShippingTests(unittest.TestCase):
    def test_round_trip_and_canal_fee(self):
        v = shipping.voyage(4080, False, False, BASE)
        self.assertAlmostEqual(v["round_trip_days"], 2 * 4080 / (17 * 24) + 4)
        with_fee = shipping.voyage(4080, True, False, BASE)["shipping_usd_mmbtu"]
        self.assertAlmostEqual(with_fee - v["shipping_usd_mmbtu"], 600000 / (3.6 * shipping.MMBTU_PER_BCF))

    def test_netback_falls_with_distance(self):
        near = shipping.netback(10.0, 5000, False, False, BASE)
        far = shipping.netback(10.0, 15000, False, False, BASE)
        self.assertLess(far, near)
        self.assertLess(near, 10.0 - BASE["regas_fee"])


class FleetTests(unittest.TestCase):
    def test_port_call_detected_and_matched_by_date_when_name_differs(self):
        times = pd.date_range("2023-10-01", periods=60, freq="1h")
        lat = np.r_[np.linspace(29.0, fleet.BERTH[0], 10), np.full(30, fleet.BERTH[0]), np.linspace(fleet.BERTH[0], 29.0, 20)]
        ais = pd.DataFrame({"mmsi": 1, "time": times, "lat": lat, "lon": fleet.BERTH[1], "sog_kn": np.r_[np.full(10, 12), np.zeros(30), np.full(20, 12)],
                            "vessel": "OLD NAME", "vessel_key": "OLD NAME"})
        cargoes = pd.DataFrame({"terminal": ["Calcasieu Pass"], "date": [pd.Timestamp("2023-10-02")], "vessel_key": ["NEW NAME"], "vessel": ["New Name"]})
        calls = fleet.port_calls(ais, cargoes)
        self.assertEqual(len(calls), 1)
        self.assertAlmostEqual(calls.berth_hours.iloc[0], 29, delta=1)
        self.assertTrue(calls.match_method.iloc[0].startswith("date only"))


class PriceModelTests(unittest.TestCase):
    def test_training_targets_end_before_test_year(self):
        weeks = pd.Series(pd.date_range("2018-01-05", "2023-12-29", freq="W-FRI"))
        train, test = price_model.split_origins(weeks, 2022)
        self.assertTrue((weeks.iloc[train + max(price_model.HORIZONS)] < pd.Timestamp("2022-01-01")).all())
        self.assertTrue((weeks.iloc[test].dt.year == 2022).all())

    def test_diebold_mariano_identical_errors(self):
        errors = np.random.default_rng(0).normal(size=200)
        _, p = price_model.diebold_mariano(errors, errors + 1e-12, 4)
        self.assertGreater(p, 0.5)


if __name__ == "__main__":
    unittest.main()
