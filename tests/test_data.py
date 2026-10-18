"""Bronze integrity, DOE cleaning, contracts and SEC parsing."""
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from vg import bronze, sec, silver
from vg.contracts import ContractError, check
from vg.common import write_json

TERMINALS = pd.DataFrame({"terminal": ["Calcasieu Pass"], "doe_point": ["Cameron (Calcasieu Pass), LA"]})
DESTINATIONS = pd.DataFrame({"country": ["France", "Spain", "Japan"], "region": ["Europe", "Europe", "Asia"]})


def doe_rows(**overrides):
    rows = {"Arrival/Departure Date": ["2026-07-02", "2026-07-02"], "Point of Entry or Exit": ["Cameron (Calcasieu Pass), LA"] * 2,
            "Volume (MMCF)": [1000, 2000], "Tanker": ["Example LNG"] * 2, "Country": ["France", "Spain"],
            "Companies": ["Venture Global Calcasieu Pass, LLC"] * 2, "Docket Term": ["Long-Term"] * 2,
            "Activity": ["Exports"] * 2, "Mode of Transport": ["Vessel"] * 2, "Gas Type": ["LNG"] * 2}
    rows.update(overrides)
    return pd.DataFrame(rows)


class BronzeTests(unittest.TestCase):
    def test_changed_snapshot_fails_verification(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            (folder / "a.csv").write_text("x\n1\n")
            manifest = {key: {"file": "a.csv", "sha256": "0" * 64} for key in [*bronze.SOURCES, *bronze.EXTRACTS]}
            write_json(folder / "manifest.json", manifest)
            with mock.patch.object(bronze, "BRONZE", folder), mock.patch.object(bronze, "MANIFEST", folder / "manifest.json"):
                with self.assertRaisesRegex(ValueError, "missing or changed"):
                    bronze.verify()

    def test_gzip_snapshots_are_deterministic(self):
        self.assertEqual(bronze.deterministic_gzip(b"same bytes"), bronze.deterministic_gzip(b"same bytes"))


class DoeTests(unittest.TestCase):
    def test_split_allocations_form_one_cargo(self):
        allocations, cargoes, _ = silver.doe(TERMINALS, DESTINATIONS, doe_rows())
        self.assertEqual(len(allocations), 2)
        self.assertEqual(len(cargoes), 1)
        self.assertAlmostEqual(cargoes.volume_bcf.iloc[0], 3.0)
        self.assertEqual(cargoes.destination.iloc[0], "Spain")       # largest allocation
        self.assertEqual(cargoes.destinations.iloc[0], "France; Spain")

    def test_other_terminals_are_excluded(self):
        raw = doe_rows(**{"Point of Entry or Exit": ["Cameron (Calcasieu Pass), LA", "Sabine Pass, LA"]})
        allocations, _, us = silver.doe(TERMINALS, DESTINATIONS, raw)
        self.assertEqual(len(allocations), 1)
        self.assertEqual(len(us), 2)                                  # but kept for round-trip matching

    def test_unmapped_destination_fails(self):
        with self.assertRaisesRegex(ValueError, "destinations.csv"):
            silver.doe(TERMINALS, DESTINATIONS, doe_rows(Country=["France", "Atlantis"]))

    def test_schema_change_fails(self):
        with self.assertRaisesRegex(ValueError, "Missing DOE columns"):
            silver.doe(TERMINALS, DESTINATIONS, doe_rows().drop(columns="Country"))

    def test_missing_month_is_not_zero(self):
        raw = doe_rows(**{"Arrival/Departure Date": ["2026-05-01", "2026-07-01"]})
        allocations, cargoes, _ = silver.doe(TERMINALS, DESTINATIONS, raw)
        with self.assertRaisesRegex(ValueError, "no reported exports"):
            silver.monthly(allocations, cargoes, TERMINALS)


class ContractTests(unittest.TestCase):
    def test_duplicates_nulls_and_ranges_fail(self):
        frame = pd.DataFrame({"key": [1, 1], "value": [5.0, None]})
        with self.assertRaises(ContractError) as error:
            check(frame, "test.table", ["key", "value"], unique=["key"], ranges={"value": (0, 1)})
        message = str(error.exception)
        self.assertIn("duplicate", message)
        self.assertIn("null", message)
        self.assertIn("outside", message)


class SecTests(unittest.TestCase):
    TEXT_2025 = ("Calcasieu Project Three months ended June 30, Six months ended June 30, 2025 2024 2025 2024 LNG volumes exported "
                 "Cargos 38 36 72 76 TBtu 140.2 129.0 266.2 273.5 LNG volumes sold (TBtu) 140.2 132.1 265.4 273.0 Weighted average price "
                 "of LNG volumes sold (per MMBtu) Fixed liquefaction fee $ 2.66 $ 6.14 $ 5.56 $ 6.79 Commodity fee 3.96 2.20 4.08 2.40 "
                 "Weighted average price of LNG volumes sold $ 6.62 $ 8.34 $ 9.64 $ 9.19 Plaquemines Project Three months ended June 30, "
                 "2025 2024 LNG volumes exported Cargos 51 — 80 — TBtu 190.5 — 298.3 — LNG volumes sold (TBtu) 190.6 — 298.3 — Weighted "
                 "average price of LNG volumes sold (per MMBtu) Fixed liquefaction fee $ 7.09 $ — Weighted average price of LNG volumes sold $ 11.05 $ —")
    TEXT_2026 = ("Calcasieu Project Three months ended June 30, 2026 2025 LNG volumes sold (TBtu) (1) 137.6 140.2 Weighted average price of "
                 "LNG volumes sold (per MMBtu) (2) $ 5.17 $ 6.62 Plaquemines Project Three months ended June 30, 2026 2025 LNG volumes sold "
                 "(TBtu) (1) 342.1 190.5 Weighted average price of LNG volumes sold (per MMBtu) (2) $ 10.95 $ 11.05")

    def test_both_filing_layouts(self):
        rows = {(r["terminal"], r["period_end"]): r for r in sec.parse("sec_10q_x", self.TEXT_2025, "2025-06-30")}
        cp = rows[("Calcasieu Pass", "2025-06-30")]
        self.assertEqual((cp["cargoes_exported"], cp["tbtu_exported"], cp["tbtu_sold"], cp["price_per_mmbtu"]), (38, 140.2, 140.2, 6.62))
        self.assertEqual(rows[("Plaquemines", "2024-06-30")]["tbtu_sold"], 0.0)   # an em dash means zero
        rows = {(r["terminal"], r["period_end"]): r for r in sec.parse("sec_10q_y", self.TEXT_2026, "2026-06-30")}
        self.assertEqual(rows[("Plaquemines", "2026-06-30")]["price_per_mmbtu"], 10.95)
        self.assertIsNone(rows[("Plaquemines", "2026-06-30")]["cargoes_exported"])

    def test_missing_table_fails(self):
        with self.assertRaisesRegex(ValueError, "Could not find"):
            sec.parse("sec_10q_z", "no tables here", "2026-06-30")


if __name__ == "__main__":
    unittest.main()
