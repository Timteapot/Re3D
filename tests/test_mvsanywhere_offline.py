from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VIT_MODULES = (
    ROOT
    / "vendor"
    / "mvsanywhere"
    / "src"
    / "mvsanywhere"
    / "modules"
    / "vit_modules.py"
)
MODEL_UTILS = (
    ROOT
    / "vendor"
    / "mvsanywhere"
    / "src"
    / "mvsanywhere"
    / "utils"
    / "model_utils.py"
)


class MVSAnywhereOfflineCompatibilityTests(unittest.TestCase):
    def test_dinov2_hub_reference_has_an_explicit_branch(self) -> None:
        source = VIT_MODULES.read_text(encoding="utf-8")
        self.assertIn(
            "torch.hub.load('facebookresearch/dinov2:main', model_name)",
            source,
        )
        self.assertNotIn(
            "torch.hub.load('facebookresearch/dinov2', model_name)",
            source,
        )

    def test_manual_fallback_removes_shape_dependent_buffers_before_loading(self) -> None:
        source = MODEL_UTILS.read_text(encoding="utf-8")
        first_pop = source.index(
            "state_dict.pop('compute_normals.backproject.pix_coords_13N', None)"
        )
        second_pop = source.index(
            "state_dict.pop('mv_depth_loss.backproject.pix_coords_13N', None)"
        )
        non_strict_load = source.index(
            "model.load_state_dict(state_dict, strict=False)"
        )
        self.assertLess(first_pop, non_strict_load)
        self.assertLess(second_pop, non_strict_load)
        self.assertNotIn("model.load_state_dict(torch.load(", source)


if __name__ == "__main__":
    unittest.main()
