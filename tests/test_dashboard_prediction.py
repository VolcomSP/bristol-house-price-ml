from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "dashboard"))

from prediction import (  # noqa: E402
    DashboardInputError,
    build_model_input,
    load_resources,
    normalize_postcode,
    predict_property,
)


class DashboardPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.resources = load_resources()
        cls.cases = pd.read_csv(PROJECT_ROOT / "dashboard/data/dashboard_validation_cases.csv")

    def test_postcode_normalisation(self) -> None:
        self.assertEqual(normalize_postcode("bs1  1ad"), "BS11AD")
        self.assertEqual(normalize_postcode(" BS16 1QG "), "BS161QG")

    def test_non_bristol_format_is_rejected(self) -> None:
        with self.assertRaises(DashboardInputError):
            normalize_postcode("SW1A 1AA")

    def test_feature_order_and_transformation(self) -> None:
        frame = build_model_input("T", "F", "N", "2024-01", 12, 4.5)
        self.assertEqual(frame.columns.tolist(), self.resources.model_bundle["model_features"])
        self.assertAlmostEqual(frame.loc[0, "month_index"], 24.0)
        self.assertAlmostEqual(frame.loc[0, "log_lagged_total_crime"], np.log1p(12))

    def test_period_outside_validated_scope_is_rejected(self) -> None:
        with self.assertRaises(DashboardInputError):
            build_model_input("T", "F", "N", "2025-01", 12, 4.5)

    def test_all_neighbourhood_rows_use_previous_month(self) -> None:
        features = self.resources.features
        expected = (pd.PeriodIndex(features["sale_month"], freq="M") - 1).astype(str)
        self.assertTrue((features["neighbourhood_source_month"].to_numpy() == expected.to_numpy()).all())
        self.assertEqual(len(features), 12 * 263)

    def test_saved_model_reproduces_validation_cases(self) -> None:
        for case in self.cases.itertuples(index=False):
            result = predict_property(
                self.resources,
                postcode=case.postcode,
                property_type=case.property_type,
                duration=case.duration,
                old_new=case.old_new,
                sale_month=case.sale_month,
            )
            self.assertEqual(result["analysis_lsoa_code"], case.analysis_lsoa_code)
            self.assertAlmostEqual(result["predicted_price"], case.expected_prediction_gbp, places=2)


if __name__ == "__main__":
    unittest.main()
