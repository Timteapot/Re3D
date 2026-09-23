from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_pipeline import Pipeline  # noqa: E402


def make_pipeline(**directories: Path | None) -> Pipeline:
    return Pipeline(
        scene="11111111-1111-4111-8111-111111111111",
        images=None,
        branches={"a", "b", "c"},
        config_path=ROOT / "configs" / "pipeline.json",
        paths_path=ROOT / "configs" / "paths.example.json",
        dry_run=True,
        work_dir=directories.get("work_dir"),
        output_dir=directories.get("output_dir"),
        log_dir=directories.get("log_dir"),
    )


class RunDirectoryTests(unittest.TestCase):
    def test_defaults_preserve_legacy_scene_directories(self) -> None:
        with mock.patch.dict(
            "os.environ",
            {
                "RE3D_WORK_DIR": "",
                "RE3D_OUTPUT_DIR": "",
                "RE3D_LOG_DIR": "",
            },
            clear=False,
        ):
            pipeline = make_pipeline()
        scene = "11111111-1111-4111-8111-111111111111"
        self.assertEqual(pipeline.work, (ROOT / "work" / scene).resolve())
        self.assertEqual(pipeline.outputs, (ROOT / "outputs" / scene).resolve())
        self.assertEqual(pipeline.logs, (ROOT / "logs" / scene).resolve())

    def test_explicit_directories_are_exact_and_not_nested_by_scene(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            pipeline = make_pipeline(
                work_dir=task / "runtime" / "work",
                output_dir=task / "output",
                log_dir=task / "runtime" / "logs",
            )
            self.assertEqual(pipeline.work, (task / "runtime" / "work").resolve())
            self.assertEqual(pipeline.outputs, (task / "output").resolve())
            self.assertEqual(pipeline.logs, (task / "runtime" / "logs").resolve())
            self.assertEqual(pipeline.input, pipeline.work / "input")

    def test_environment_directories_are_used_when_arguments_are_absent(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            environment = {
                "RE3D_WORK_DIR": str(task / "env-work"),
                "RE3D_OUTPUT_DIR": str(task / "env-output"),
                "RE3D_LOG_DIR": str(task / "env-logs"),
            }
            with mock.patch.dict("os.environ", environment, clear=False):
                pipeline = make_pipeline()
            self.assertEqual(pipeline.work, (task / "env-work").resolve())
            self.assertEqual(pipeline.outputs, (task / "env-output").resolve())
            self.assertEqual(pipeline.logs, (task / "env-logs").resolve())

    def test_explicit_directories_override_environment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            task = Path(temporary)
            environment = {
                "RE3D_WORK_DIR": str(task / "env-work"),
                "RE3D_OUTPUT_DIR": str(task / "env-output"),
                "RE3D_LOG_DIR": str(task / "env-logs"),
            }
            with mock.patch.dict("os.environ", environment, clear=False):
                pipeline = make_pipeline(
                    work_dir=task / "cli-work",
                    output_dir=task / "cli-output",
                    log_dir=task / "cli-logs",
                )
            self.assertEqual(pipeline.work, (task / "cli-work").resolve())
            self.assertEqual(pipeline.outputs, (task / "cli-output").resolve())
            self.assertEqual(pipeline.logs, (task / "cli-logs").resolve())


if __name__ == "__main__":
    unittest.main()
