"""Unit tests for the parsers, the label definition and the regression gate.

Run from the project root:  python -m unittest discover -s tests -v
"""
import importlib
import math
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import descriptors as d  # noqa: E402
import common  # noqa: E402


def isnan(x):
    return isinstance(x, float) and math.isnan(x)


class ParserTests(unittest.TestCase):
    def test_first_number(self):
        self.assertEqual(d.extract_first_number("~100%"), 100.0)
        self.assertEqual(d.extract_first_number("99.5%"), 99.5)
        self.assertEqual(d.extract_first_number("5 mM"), 5.0)
        self.assertTrue(isnan(d.extract_first_number(np.nan)))
        self.assertTrue(isnan(d.extract_first_number("n.d.")))

    def test_potential_requires_rhe(self):
        self.assertEqual(d.extract_potential("1.47 V vs. RHE"), 1.47)
        self.assertTrue(isnan(d.extract_potential("0.6 V vs Ag/AgCl")))
        self.assertTrue(isnan(d.extract_potential("constant current 10 mA")))
        self.assertTrue(isnan(d.extract_potential("1.8 V cell voltage")))

    def test_cell_voltage_separate_from_potential(self):
        self.assertEqual(d.extract_cell_voltage("1.8 V cell voltage"), 1.8)
        self.assertTrue(isnan(d.extract_cell_voltage("1.47 V vs. RHE")))

    def test_temperature(self):
        self.assertEqual(d.extract_temperature("55 °C"), 55.0)
        self.assertEqual(d.extract_temperature("23 ± 2 °C"), 23.0)
        self.assertEqual(d.extract_temperature("Room temperature"), 25.0)
        self.assertTrue(isnan(d.extract_temperature(np.nan)))

    def test_electrolyte(self):
        self.assertEqual(d.extract_electrolyte_concentration("1 M KOH"), 1.0)
        self.assertEqual(d.extract_electrolyte_concentration("100 mM KOH"), 0.1)
        self.assertEqual(d.extract_electrolyte_cation("1 M KOH"), "K")
        self.assertEqual(d.extract_electrolyte_cation("0.5 M Na2SO4"), "Na")
        self.assertEqual(d.extract_electrolyte_anion("1 M KOH"), "OH")
        self.assertAlmostEqual(d.extract_electrolyte_ph("1 M KOH"), 14.0)
        self.assertAlmostEqual(d.extract_electrolyte_ph("0.1 M KOH"), 13.0)
        self.assertEqual(d.extract_electrolyte_ph("pH = 9.2 borate"), 9.2)

    def test_anode_text_drops_counter_electrode(self):
        text = "Anode: Ni3N@C on Ni foam. Cathode: Pt foil"
        anode = d.anode_text(text)
        self.assertIn("Ni3N", anode)
        self.assertNotIn("Pt", anode)
        self.assertIn("Pt", d.counter_electrode_text(text))


class LabelTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({
            "Major_Product_Clean": ["FDCA", "FDCA", "FDCA", "HMFCA", "FDCA", "FDCA"],
            "Yield_pct":      [95.0, 80.0, 92.0, 99.0, np.nan, 120.0],
            "Conversion_pct": [99.0, 99.0, np.nan, 99.0, 99.0, 99.0],
        })

    def test_rows_and_labels(self):
        rows, y = common.binary_high_performance_labels(self.frame())
        # rows 0,1,2 qualify; 3 is not FDCA, 4 has no yield, 5 is out of range
        self.assertEqual(rows.tolist(), [0, 1, 2])
        self.assertEqual(y.tolist(), [1, 0, 1])

    def test_conversion_limits_performance(self):
        f = pd.DataFrame({"Major_Product_Clean": ["FDCA"], "Yield_pct": [95.0],
                          "Conversion_pct": [85.0]})
        _, y = common.binary_high_performance_labels(f)
        self.assertEqual(y.tolist(), [0])


class FeatureContractTests(unittest.TestCase):
    def test_outcomes_never_become_features(self):
        frame = pd.DataFrame({
            "Conversion_pct": [1.0, 2.0, 3.0], "Yield_pct": [1.0, 2.0, 4.0],
            "Catalyst_Ni": [0, 1, 0], "Catalyst_Yield_Proxy": [0, 1, 0],
            "Temperature_C": [20.0, 30.0, 40.0], "Cluster": [0, 1, 2],
            "Current_mA": [1.0, 2.0, 3.0],
        })
        cols = common.feature_columns(frame)
        self.assertEqual(sorted(cols), ["Catalyst_Ni", "Temperature_C"])


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reg = importlib.import_module("07_regression")

    def scores(self, model, base, r2, failing=()):
        rows, reps = [], []
        for target in common.TARGETS:
            bad = target in failing
            for _ in range(6):
                rows.append({"target": target, "model_mae": base + 1 if bad else model, "baseline_mae": base,
                             "mean_baseline_mae": base + 2, "model_r2": r2})
            for _ in range(3):
                reps.append({"target": target, "pooled_r2": -0.5 if bad else r2})
        return pd.DataFrame(rows), pd.DataFrame(reps)

    def test_passes_when_clearly_better(self):
        passed, _ = self.reg.validation_gate(*self.scores(5.0, 8.0, 0.2))
        self.assertTrue(passed)

    def test_fails_when_worse_than_median_even_if_better_than_mean(self):
        # model 9 beats a mean baseline (10) but not the median baseline (8)
        s, r = self.scores(9.0, 8.0, 0.2)
        s["mean_baseline_mae"] = 10.0
        passed, checks = self.reg.validation_gate(s, r)
        self.assertFalse(passed)
        self.assertFalse(checks["Yield_pct"]["mae_beats_baseline"])

    def test_fails_on_negative_pooled_r2_for_required_target(self):
        s, r = self.scores(5.0, 8.0, 0.2)
        r.loc[r.target == "Yield_pct", "pooled_r2"] = -0.5
        passed, checks = self.reg.validation_gate(s, r)
        self.assertFalse(passed)
        self.assertFalse(checks["Yield_pct"]["r2_at_or_above_floor"])

    def test_informational_target_does_not_block(self):
        passed, checks = self.reg.validation_gate(*self.scores(5.0, 8.0, 0.2, failing=("Conversion_pct",)))
        self.assertTrue(passed)
        self.assertFalse(checks["Conversion_pct"]["passed"])
        self.assertFalse(checks["Conversion_pct"]["required"])


class RegressorTests(unittest.TestCase):
    def test_fits_with_missing_values_and_keeps_imputer_first(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(rng.normal(size=(60, 5)), columns=list("abcde"))
        y = 3 * X["a"].to_numpy() + rng.normal(size=60)
        X.iloc[::7, 1] = np.nan
        model = common.make_regressor_single(0).fit(X, y)
        self.assertEqual(model.steps[0][0], "imputer")
        self.assertEqual(len(model.named_steps["model"].estimators_), 2)
        self.assertEqual(model.predict(X).shape, (60,))


if __name__ == "__main__":
    unittest.main()
