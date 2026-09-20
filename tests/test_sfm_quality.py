from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from sfm_quality import ImageQuality, select_deferred_images  # noqa: E402


class DeferredRegistrationTests(unittest.TestCase):
    def test_low_observation_and_low_ratio_image_is_deferred(self) -> None:
        images = [
            ImageQuality(index, index, f"r_{index}.png", 3000, observations)
            for index, observations in enumerate([1400, 1500, 1350, 1600, 47])
        ]
        deferred, report = select_deferred_images(
            images,
            relative_observation_floor=0.15,
            minimum_triangulated_ratio=0.05,
            max_images=2,
        )
        self.assertEqual([item.name for item in deferred], ["r_4.png"])
        self.assertEqual(report["registered_images"], 5)

    def test_low_observation_but_adequate_ratio_is_kept(self) -> None:
        images = [
            ImageQuality(1, 1, "dense.png", 3000, 1500),
            ImageQuality(2, 2, "sparse.png", 300, 120),
            ImageQuality(3, 3, "dense-2.png", 3000, 1600),
        ]
        deferred, _ = select_deferred_images(
            images,
            relative_observation_floor=0.5,
            minimum_triangulated_ratio=0.05,
            max_images=2,
        )
        self.assertEqual(deferred, [])

    def test_candidate_count_is_bounded(self) -> None:
        images = [ImageQuality(0, 0, "core.png", 3000, 1500)] * 5 + [
            ImageQuality(10, 10, "bad-a.png", 3000, 20),
            ImageQuality(11, 11, "bad-b.png", 3000, 30),
        ]
        deferred, _ = select_deferred_images(
            images,
            relative_observation_floor=0.15,
            minimum_triangulated_ratio=0.05,
            max_images=1,
        )
        self.assertEqual([item.name for item in deferred], ["bad-a.png"])


if __name__ == "__main__":
    unittest.main()
