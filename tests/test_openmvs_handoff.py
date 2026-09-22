from __future__ import annotations

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from export_model_depths_to_openmvs import export  # noqa: E402
from openmvs_dmap import read_dmap, sha256_file, write_dmap  # noqa: E402
from stage_openmvs_handoff import stage, verify  # noqa: E402
from validate_openmvs_handoff import validate_handoff  # noqa: E402


def camera_header(width: int = 8, height: int = 6) -> dict:
    return {
        "image_width": width,
        "image_height": height,
        "depth_width": width,
        "depth_height": height,
        "file_name": "../images/000000.png",
        "view_ids": np.asarray([0], dtype=np.uint32),
        "K": np.asarray(
            [[500.0, 0.0, width / 2], [0.0, 500.0, height / 2], [0, 0, 1]],
            dtype=np.float64,
        ),
        "R": np.eye(3, dtype=np.float64),
        "C": np.zeros(3, dtype=np.float64),
    }


class DmapTests(unittest.TestCase):
    def test_depth_confidence_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth0000.dmap"
            depth = np.arange(48, dtype=np.float32).reshape(6, 8)
            confidence = np.full((6, 8), 0.75, dtype=np.float32)
            write_dmap(path, camera_header(), depth, confidence)
            header, payload = read_dmap(path)
            self.assertEqual(header["content_type"], 5)
            self.assertEqual((header["depth_width"], header["depth_height"]), (8, 6))
            np.testing.assert_array_equal(payload["depth"], depth)
            self.assertEqual(payload["confidence"][0, 0], 0)
            np.testing.assert_allclose(payload["confidence"][depth > 0], 0.75)

    def test_writer_rejects_image_depth_resolution_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            header = camera_header()
            header["image_width"] = 9
            with self.assertRaisesRegex(ValueError, "resolutions to match"):
                write_dmap(
                    Path(temporary) / "bad.dmap",
                    header,
                    np.ones((6, 8), dtype=np.float32),
                    np.ones((6, 8), dtype=np.float32),
                )


class HandoffTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.templates = self.root / "templates"
        self.depth = self.root / "model-depth"
        (self.root / "images").mkdir()
        (self.root / "images" / "000000.png").write_bytes(b"synthetic-image")
        self.templates.mkdir()
        self.depth.mkdir()
        template_depth = np.full((6, 8), 2.0, dtype=np.float32)
        template_normal = np.zeros((6, 8, 3), dtype=np.float32)
        template_normal[..., 2] = 1
        write_dmap(
            self.templates / "depth0000.dmap",
            camera_header(),
            template_depth,
            normal=template_normal,
        )
        (self.templates / "scene.mvs").write_bytes(b"synthetic-scene")
        np.save(self.depth / "000000_depth.npy", np.full((3, 4), 3.0, np.float32))
        np.save(self.depth / "000000_mask.npy", np.ones((3, 4), bool))
        np.save(
            self.depth / "000000_confidence.npy",
            np.full((3, 4), 0.8, np.float32),
        )
        self.canonical = self.root / "canonical"
        self.contract = {
            "resolution_level": 1,
            "min_resolution": 640,
            "max_resolution": 1024,
        }
        args = argparse.Namespace(
            template_dmaps=self.templates,
            scene=self.templates / "scene.mvs",
            depth=self.depth,
            output=self.canonical,
            depth_convention="camera_z",
            confidence_semantics="cross_view_support_ratio",
            **self.contract,
        )
        export(args)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_export_and_validation_pass(self) -> None:
        report = validate_handoff(
            self.canonical,
            self.templates,
            self.canonical / "dmap-export.json",
            self.contract,
        )
        self.assertTrue(report["all_passed"], report["errors"])
        self.assertEqual(report["dmap_count"], 1)

    def test_resolution_contract_mismatch_fails(self) -> None:
        mismatched = dict(self.contract)
        mismatched["resolution_level"] = 0
        report = validate_handoff(
            self.canonical,
            self.templates,
            self.canonical / "dmap-export.json",
            mismatched,
        )
        self.assertFalse(report["all_passed"])
        self.assertTrue(
            any("resolution contract mismatch" in item for item in report["errors"])
        )

    def test_staging_preserves_canonical_hashes(self) -> None:
        runtime = self.root / "runtime"
        stage_manifest = stage(self.canonical, runtime)
        canonical_dmap = self.canonical / "depth0000.dmap"
        before = sha256_file(canonical_dmap)
        with (runtime / "depth0000.dmap").open("ab") as handle:
            handle.write(b"runtime mutation")
        report = verify(runtime / "staging-manifest.json")
        self.assertTrue(report["all_passed"], report["errors"])
        self.assertEqual(before, sha256_file(canonical_dmap))
        self.assertEqual(stage_manifest["file_count"], 3)

    def test_canonical_tamper_is_detected(self) -> None:
        runtime = self.root / "runtime"
        stage(self.canonical, runtime)
        with (self.canonical / "depth0000.dmap").open("ab") as handle:
            handle.write(b"canonical mutation")
        report = verify(runtime / "staging-manifest.json")
        self.assertFalse(report["all_passed"])
        self.assertTrue(any("hash changed" in item for item in report["errors"]))

    def test_staging_rejects_broken_relative_image_reference(self) -> None:
        (self.root / "images" / "000000.png").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "relative image references"):
            stage(self.canonical, self.root / "runtime")

    def test_validator_detects_changed_source_array(self) -> None:
        np.save(self.depth / "000000_depth.npy", np.zeros((3, 4), np.float32))
        report = validate_handoff(
            self.canonical,
            self.templates,
            self.canonical / "dmap-export.json",
            self.contract,
        )
        self.assertFalse(report["all_passed"])
        self.assertTrue(any("source depth hash changed" in item for item in report["errors"]))


class PipelineConfigTests(unittest.TestCase):
    def make_pipeline(self):
        from run_pipeline import Pipeline

        return Pipeline(
            scene="contract-test",
            images=None,
            branches={"a"},
            config_path=ROOT / "configs" / "pipeline.json",
            paths_path=ROOT / "configs" / "paths.local.json",
            dry_run=True,
        )

    def test_c_and_external_commands_share_resolution_contract(self) -> None:
        pipeline = self.make_pipeline()
        commands: list[list[str]] = []

        def collect(_name, command, _marker=None):
            commands.append([str(value) for value in command])

        pipeline.step = collect  # type: ignore[method-assign]
        pipeline.c_geometry()
        pipeline.external_fusion("test", pipeline.a_openmvs, 1)
        densify_commands = [
            command for command in commands if command[0].endswith("DensifyPointCloud.exe")
        ]
        self.assertEqual(len(densify_commands), 2)

        def resolution(command: list[str]) -> tuple[str, str, str]:
            return (
                command[command.index("--resolution-level") + 1],
                command[command.index("--min-resolution") + 1],
                command[command.index("--max-resolution") + 1],
            )

        self.assertEqual(resolution(densify_commands[0]), resolution(densify_commands[1]))

    def test_texturemesh_executable_can_be_overridden(self) -> None:
        fixed = ROOT / ".cache" / "openmvs-fixed" / "TextureMesh.exe"
        with mock.patch.dict(
            "os.environ", {"RE3D_TEXTUREMESH_EXE": str(fixed)}, clear=False
        ):
            pipeline = self.make_pipeline()
        commands: list[list[str]] = []

        def collect(_name, command, _marker=None):
            commands.append([str(value) for value in command])

        pipeline.step = collect  # type: ignore[method-assign]
        pipeline.texture("test", pipeline.c_openmvs, "test-output")
        self.assertEqual(Path(commands[0][0]), fixed)

    def test_default_texturemesh_uses_fixed_vendor_bundle(self) -> None:
        from run_pipeline import DEFAULT_TEXTUREMESH, resolve_texturemesh_path

        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertEqual(resolve_texturemesh_path({}), DEFAULT_TEXTUREMESH)

    def test_bundled_texturemesh_is_rejected_when_seams_are_enabled(self) -> None:
        from run_pipeline import Pipeline

        with tempfile.TemporaryDirectory() as temporary:
            config = json.loads((ROOT / "configs" / "pipeline.json").read_text())
            config["texture"]["global_seam_leveling"] = 1
            config_path = Path(temporary) / "pipeline.json"
            config_path.write_text(json.dumps(config), encoding="utf-8")
            bundled = (
                ROOT
                / "vendor"
                / "openmvs-2.4.0-windows"
                / "vc17"
                / "x64"
                / "Release"
                / "TextureMesh.exe"
            )
            with mock.patch.dict(
                "os.environ", {"RE3D_TEXTUREMESH_EXE": str(bundled)}, clear=False
            ):
                with self.assertRaisesRegex(ValueError, "eeedab7"):
                    Pipeline(
                        scene="contract-test",
                        images=None,
                        branches={"c"},
                        config_path=config_path,
                        paths_path=ROOT / "configs" / "paths.local.json",
                        dry_run=True,
                    )

    def test_c_densify_explicitly_uses_alpha_masks(self) -> None:
        pipeline = self.make_pipeline()
        commands: list[list[str]] = []

        def collect(_name, command, _marker=None):
            commands.append([str(value) for value in command])

        pipeline.step = collect  # type: ignore[method-assign]
        pipeline.c_geometry()
        densify = next(
            command for command in commands if command[0].endswith("DensifyPointCloud.exe")
        )
        self.assertEqual(
            Path(densify[densify.index("--mask-path") + 1]),
            pipeline.shared / "openmvs_input" / "images",
        )
        self.assertEqual(
            densify[densify.index("--ignore-mask-label") + 1],
            str(pipeline.config["c"]["ignore_mask_label"]),
        )

    def test_colmap_enables_deferred_registration_from_config(self) -> None:
        pipeline = self.make_pipeline()
        commands: list[list[str]] = []

        def collect(_name, command, _marker=None):
            commands.append([str(value) for value in command])

        pipeline.step = collect  # type: ignore[method-assign]
        pipeline.cameras()
        colmap = next(
            command for command in commands if command[1].endswith("run_colmap_shared.py")
        )
        deferred = pipeline.config["camera_frontend"]["deferred_registration"]
        self.assertIn("--deferred-registration", colmap)
        self.assertEqual(
            colmap[colmap.index("--defer-relative-observation-floor") + 1],
            str(deferred["relative_observation_floor"]),
        )
        self.assertEqual(
            colmap[colmap.index("--retry-abs-pose-min-num-inliers") + 1],
            str(deferred["retry_abs_pose_min_num_inliers"]),
        )
        self.assertEqual(
            colmap[colmap.index("--random-seed") + 1],
            str(pipeline.config["camera_frontend"]["random_seed"]),
        )


if __name__ == "__main__":
    unittest.main()
