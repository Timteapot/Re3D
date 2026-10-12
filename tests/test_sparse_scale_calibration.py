from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_depths_to_colmap_sparse import select_scale_ratios  # noqa: E402


class SparseScaleSelectionTests(unittest.TestCase):
    def test_prefers_existing_bounded_path(self) -> None:
        ratios = np.concatenate(
            [np.linspace(0.95, 1.05, 80), np.asarray([3.2, 3.3, 3.4])]
        )

        selected, report = select_scale_ratios(ratios, minimum_points=30)

        self.assertEqual(report["selection_mode"], "bounded")
        self.assertEqual(report["positive_samples"], 83)
        self.assertEqual(len(selected), 80)
        self.assertLess(float(selected.max()), 2.0)

    def test_accepts_coherent_scale_outside_preferred_bounds(self) -> None:
        ratios = np.linspace(3.2, 3.6, 100)

        selected, report = select_scale_ratios(ratios, minimum_points=30)

        self.assertEqual(report["selection_mode"], "adaptive")
        self.assertEqual(report["bounded_samples"], 0)
        self.assertEqual(len(selected), 100)
        self.assertAlmostEqual(float(np.median(selected)), 3.4)
        self.assertLess(report["selected_p90_p10_ratio"], 2.0)

    def test_adaptive_path_selects_shifted_majority_and_rejects_outliers(self) -> None:
        ratios = np.concatenate(
            [
                np.linspace(0.9, 1.1, 15),
                np.linspace(3.2, 3.6, 100),
                np.asarray([1000.0]),
            ]
        )

        selected, report = select_scale_ratios(ratios, minimum_points=30)

        self.assertEqual(report["selection_mode"], "adaptive")
        self.assertEqual(report["bounded_samples"], 15)
        self.assertEqual(len(selected), 100)
        self.assertGreater(float(selected.min()), 3.0)
        self.assertLess(float(selected.max()), 4.0)

    def test_rejects_dispersed_adaptive_scale(self) -> None:
        ratios = np.geomspace(2.1, 20.0, 100)

        with self.assertRaisesRegex(ValueError, "too dispersed"):
            select_scale_ratios(ratios, minimum_points=30)

    def test_rejects_too_few_positive_samples(self) -> None:
        ratios = np.asarray([3.2] * 29 + [0.0, -1.0, np.nan, np.inf])

        with self.assertRaisesRegex(ValueError, "only 29 robust scale samples"):
            select_scale_ratios(ratios, minimum_points=30)

    def test_rejects_invalid_policy_parameters(self) -> None:
        with self.assertRaisesRegex(ValueError, "ratio bounds"):
            select_scale_ratios(np.ones(30), 30, ratio_minimum=2.0, ratio_maximum=1.0)
        with self.assertRaisesRegex(ValueError, "P90/P10"):
            select_scale_ratios(
                np.ones(30),
                30,
                fallback_maximum_p90_p10_ratio=0.9,
            )


if __name__ == "__main__":
    unittest.main()
