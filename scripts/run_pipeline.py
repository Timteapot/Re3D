from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENMVS = ROOT / "vendor/openmvs-2.4.0-windows/vc17/x64/Release"
DEFAULT_TEXTUREMESH = (
    ROOT
    / "vendor/openmvs-2.4.0-33d9484-windows/vc18/x64/Release/TextureMesh.exe"
)
LEGACY_TEXTUREMESH = DEFAULT_OPENMVS / "TextureMesh.exe"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def resolve_runtime_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def resolve_run_directory(
    configured: Path | str | None,
    environment_name: str,
    default: Path,
) -> Path:
    value = configured if configured is not None else os.environ.get(environment_name)
    if value is None or not str(value).strip():
        return default.resolve()
    return resolve_runtime_path(str(value)).resolve()


def resolve_texturemesh_path(local_paths: dict) -> Path:
    configured = os.environ.get("RE3D_TEXTUREMESH_EXE") or local_paths.get(
        "texturemesh_executable"
    )
    return (
        resolve_runtime_path(configured)
        if configured
        else DEFAULT_TEXTUREMESH
    )


class Pipeline:
    def __init__(
        self,
        scene: str,
        images: Path | None,
        branches: set[str],
        config_path: Path,
        paths_path: Path,
        dry_run: bool,
        work_dir: Path | str | None = None,
        output_dir: Path | str | None = None,
        log_dir: Path | str | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9._-]+", scene):
            raise ValueError("Scene name may contain only letters, numbers, '.', '_' and '-'")
        self.scene = scene
        self.images = images.resolve() if images else None
        self.branches = branches
        self.config = read_json(config_path)
        self.local_paths = read_json(paths_path)
        self.dry_run = dry_run
        self.map_python = resolve_runtime_path(
            os.environ.get("RE3D_MAP_PYTHON", self.local_paths["mapanything_python"])
        )
        self.mvs_python = resolve_runtime_path(
            os.environ.get("RE3D_MVS_PYTHON", self.local_paths["mvsanywhere_python"])
        )
        self.openmvs = DEFAULT_OPENMVS
        self.texturemesh = resolve_texturemesh_path(self.local_paths)
        texture = self.config["texture"]
        seam_leveling_enabled = bool(
            int(texture["global_seam_leveling"])
            or int(texture["local_seam_leveling"])
        )
        if (
            seam_leveling_enabled
            and self.texturemesh.resolve() == LEGACY_TEXTUREMESH.resolve()
        ):
            raise ValueError(
                "Seam leveling is unsafe with the bundled OpenMVS 2.4.0 TextureMesh; "
                "set RE3D_TEXTUREMESH_EXE or paths.local.json:texturemesh_executable "
                "to a build containing upstream fix eeedab7 before enabling it"
            )
        self.work = resolve_run_directory(
            work_dir,
            "RE3D_WORK_DIR",
            ROOT / "work" / scene,
        )
        self.input = self.work / "input"
        self.shared = self.work / "shared"
        self.branch_root = self.work / "branches"
        self.a_depth = self.branch_root / "a_mapanything"
        self.b_depth = self.branch_root / "b_mvsanywhere"
        self.a_openmvs_input = self.branch_root / "a_openmvs_input"
        self.b_openmvs_input = self.branch_root / "b_openmvs_input"
        self.a_openmvs = self.branch_root / "a_openmvs"
        self.b_openmvs = self.branch_root / "b_openmvs"
        self.c_openmvs = self.branch_root / "c_openmvs"
        self.outputs = resolve_run_directory(
            output_dir,
            "RE3D_OUTPUT_DIR",
            ROOT / "outputs" / scene,
        )
        self.logs = resolve_run_directory(
            log_dir,
            "RE3D_LOG_DIR",
            ROOT / "logs" / scene,
        )
        self.base_env = os.environ.copy()
        self.base_env["HF_HOME"] = str(ROOT / "models/huggingface-cache")
        self.base_env["TORCH_HOME"] = str(ROOT / "models/torch")
        python_path = [
            str(ROOT / "vendor/map-anything"),
            str(ROOT / "vendor/mvsanywhere"),
            str(ROOT / "vendor/mvsanywhere/src"),
        ]
        if self.base_env.get("PYTHONPATH"):
            python_path.append(self.base_env["PYTHONPATH"])
        self.base_env["PYTHONPATH"] = os.pathsep.join(python_path)
        dense = self.config["openmvs_dense_resolution"]
        if int(dense["resolution_level"]) < 0:
            raise ValueError("openmvs_dense_resolution.resolution_level must be >= 0")
        if int(dense["min_resolution"]) <= 0:
            raise ValueError("openmvs_dense_resolution.min_resolution must be > 0")
        if int(dense["max_resolution"]) < int(dense["min_resolution"]):
            raise ValueError(
                "openmvs_dense_resolution.max_resolution must be >= min_resolution"
            )
        c_ignore_mask_label = int(self.config["c"]["ignore_mask_label"])
        if not 0 <= c_ignore_mask_label <= 255:
            raise ValueError("c.ignore_mask_label must be between 0 and 255")
        deferred = self.config["camera_frontend"]["deferred_registration"]
        if not 0 < float(deferred["relative_observation_floor"]) < 1:
            raise ValueError(
                "camera_frontend.deferred_registration.relative_observation_floor "
                "must be between 0 and 1"
            )
        if not 0 < float(deferred["minimum_triangulated_ratio"]) < 1:
            raise ValueError(
                "camera_frontend.deferred_registration.minimum_triangulated_ratio "
                "must be between 0 and 1"
            )
        if int(deferred["max_images"]) < 1:
            raise ValueError(
                "camera_frontend.deferred_registration.max_images must be >= 1"
            )

    def step(self, name: str, command: list[Path | str], marker: Path | None = None) -> None:
        printable = subprocess.list2cmdline([str(value) for value in command])
        if marker is not None and marker.exists():
            print(f"[skip] {name}: {marker}", flush=True)
            return
        print(f"[run] {name}\n  {printable}", flush=True)
        if self.dry_run:
            return
        self.logs.mkdir(parents=True, exist_ok=True)
        log_path = self.logs / f"{name}.log"
        with log_path.open("w", encoding="utf-8") as log:
            log.write(printable + "\n\n")
            process = subprocess.Popen(
                [str(value) for value in command],
                cwd=ROOT,
                env=self.base_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
            return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"Stage {name} failed with exit code {return_code}; see {log_path}")
        if marker is not None and not marker.exists():
            raise RuntimeError(f"Stage {name} did not create expected artifact: {marker}")

    def script(self, name: str) -> Path:
        return ROOT / "scripts" / name

    def exe(self, name: str) -> Path:
        return self.openmvs / name

    def prepare(self) -> tuple[int, int]:
        marker = self.input / "input-manifest.json"
        if not marker.exists() and self.images is None:
            raise ValueError("--images is required for a new scene")
        if self.images is not None:
            self.step(
                "00-prepare-input",
                [
                    self.map_python,
                    self.script("prepare_dataset.py"),
                    "--source",
                    self.images,
                    "--output",
                    self.input,
                    "--alpha-threshold",
                    str(self.config["input"]["alpha_threshold"]),
                ],
                marker,
            )
        if marker.exists():
            width, height = read_json(marker)["uniform_size"]
            return int(width), int(height)
        # Dry-run fallback: dimensions are only needed to print later commands.
        from PIL import Image

        candidates = sorted(
            path
            for path in self.images.iterdir()
            if path.suffix.lower() in set(self.config["input"]["extensions"])
        )
        with Image.open(candidates[0]) as image:
            return image.width, image.height

    def cameras(self) -> Path:
        images = self.input / "images"
        masks = self.input / "masks"
        colmap = self.shared / "colmap"
        cameras = self.shared / "cameras_refined.npz"
        camera_cfg = self.config["camera_frontend"]
        deferred = camera_cfg["deferred_registration"]
        colmap_command: list[Path | str] = [
            self.map_python,
            self.script("run_colmap_shared.py"),
            "--images",
            images,
            "--masks",
            masks,
            "--output",
            colmap,
            "--camera-model",
            camera_cfg["camera_model"],
            "--random-seed",
            str(camera_cfg["random_seed"]),
        ]
        if deferred["enabled"]:
            colmap_command.extend(
                [
                    "--deferred-registration",
                    "--defer-relative-observation-floor",
                    str(deferred["relative_observation_floor"]),
                    "--defer-min-triangulated-ratio",
                    str(deferred["minimum_triangulated_ratio"]),
                    "--defer-max-images",
                    str(deferred["max_images"]),
                    "--retry-abs-pose-max-error",
                    str(deferred["retry_abs_pose_max_error"]),
                    "--retry-abs-pose-min-num-inliers",
                    str(deferred["retry_abs_pose_min_num_inliers"]),
                    "--retry-abs-pose-min-inlier-ratio",
                    str(deferred["retry_abs_pose_min_inlier_ratio"]),
                ]
            )
        self.step(
            "01-colmap",
            colmap_command,
            colmap / "reconstruction_metrics.json",
        )
        self.step(
            "02-extract-cameras",
            [
                self.map_python,
                self.script("extract_colmap_cameras.py"),
                "--model",
                colmap / "refined_text",
                "--output",
                cameras,
            ],
            cameras,
        )
        openmvs_input = self.shared / "openmvs_input"
        self.step(
            "03-prepare-openmvs-input",
            [
                self.map_python,
                self.script("prepare_openmvs_texture_input.py"),
                "--images",
                images,
                "--sparse",
                colmap / "refined_text",
                "--output",
                openmvs_input,
                "--alpha-threshold",
                str(self.config["input"]["alpha_threshold"]),
            ],
            openmvs_input / "manifest.json",
        )
        return cameras

    def c_geometry(self) -> None:
        cfg = self.config["c"]
        dense = self.config["openmvs_dense_resolution"]
        if not self.dry_run:
            self.c_openmvs.mkdir(parents=True, exist_ok=True)
        self.step(
            "10-c-interface",
            [
                self.exe("InterfaceCOLMAP.exe"),
                "-w",
                self.c_openmvs,
                "-i",
                self.shared / "openmvs_input",
                "-o",
                "scene.mvs",
                "--binary",
                "1",
            ],
            self.c_openmvs / "scene.mvs",
        )
        self.step(
            "11-c-densify",
            [
                self.exe("DensifyPointCloud.exe"),
                "-w",
                self.c_openmvs,
                "-i",
                "scene.mvs",
                "-o",
                "scene_dense.mvs",
                "--resolution-level",
                str(dense["resolution_level"]),
                "--min-resolution",
                str(dense["min_resolution"]),
                "--max-resolution",
                str(dense["max_resolution"]),
                "--number-views",
                str(cfg["number_views"]),
                "--mask-path",
                self.shared / "openmvs_input" / "images",
                "--ignore-mask-label",
                str(cfg["ignore_mask_label"]),
                "--max-threads",
                str(cfg["openmvs_threads"]),
            ],
            self.c_openmvs / "scene_dense.mvs",
        )

    def reconstruct(self, prefix: str, folder: Path, threads: int) -> None:
        cfg = self.config["mesh"]
        command: list[str | Path] = [
            self.exe("ReconstructMesh.exe"),
            "-w",
            folder,
            "-i",
            "scene_dense.mvs",
            "-o",
            "scene_mesh.mvs",
            "--remove-spurious",
            str(cfg["remove_spurious"]),
            "--close-holes",
            str(cfg["close_holes"]),
            "--smooth",
            str(cfg["smooth"]),
        ]
        command.extend(["--max-threads", str(threads)])
        self.step(
            f"{prefix}-reconstruct",
            command,
            folder / "scene_mesh.ply",
        )

    def texture(self, prefix: str, folder: Path, output_label: str) -> None:
        cfg = self.config["texture"]
        self.step(
            f"{prefix}-texture",
            [
                self.texturemesh,
                "-w",
                folder,
                "-i",
                "scene.mvs",
                "--mesh-file",
                "scene_mesh.ply",
                "-o",
                "scene_textured_final.mvs",
                "--export-type",
                "obj",
                "--max-texture-size",
                str(cfg["max_texture_size"]),
                "--global-seam-leveling",
                str(cfg["global_seam_leveling"]),
                "--local-seam-leveling",
                str(cfg["local_seam_leveling"]),
                "--sharpness-weight",
                str(cfg["sharpness_weight"]),
                "--outlier-threshold",
                str(cfg["outlier_threshold"]),
                "--ignore-mask-label",
                str(cfg["ignore_mask_label"]),
                "--empty-color",
                str(cfg["empty_color"]),
                "--max-threads",
                str(cfg["max_threads"]),
            ],
            folder / "scene_textured_final.obj",
        )
        self.step(
            f"{prefix}-normalize",
            [
                self.map_python,
                self.script("normalize_openmvs_output.py"),
                "--obj",
                folder / "scene_textured_final.obj",
                "--mtl",
                folder / "scene_textured_final.mtl",
                "--texture",
                folder / "scene_textured_final_material_00_map_Kd.jpg",
                "--output",
                self.outputs / output_label,
            ],
            self.outputs / output_label / "mesh.glb",
        )

    def c_final(self) -> None:
        threads = int(self.config["c"]["openmvs_threads"])
        self.reconstruct("12-c", self.c_openmvs, threads)
        self.texture("13-c", self.c_openmvs, "C")

    def external_fusion(self, prefix: str, folder: Path, threads: int) -> None:
        cfg = self.config["external_depth_fusion"]
        dense = self.config["openmvs_dense_resolution"]
        self.step(
            f"{prefix}-fuse",
            [
                self.exe("DensifyPointCloud.exe"),
                "-w",
                folder,
                "-i",
                "scene.mvs",
                "-o",
                "scene_dense.mvs",
                "--resolution-level",
                str(dense["resolution_level"]),
                "--min-resolution",
                str(dense["min_resolution"]),
                "--max-resolution",
                str(dense["max_resolution"]),
                "--number-views-fuse",
                str(cfg["number_views_fuse"]),
                "--fusion-filter",
                str(cfg["fusion_filter"]),
                "--geometric-iters",
                str(cfg["geometric_iters"]),
                "--postprocess-dmaps",
                str(cfg["postprocess_dmaps"]),
                "--estimate-colors",
                str(cfg["estimate_colors"]),
                "--estimate-normals",
                str(cfg["estimate_normals"]),
                "--tower-mode",
                str(cfg["tower_mode"]),
                "--max-threads",
                str(threads),
            ],
            folder / "scene_dense.mvs",
        )

    def prepare_external_handoff(
        self,
        branch: str,
        step_prefix: str,
        depth: Path,
        canonical: Path,
        staging: Path,
    ) -> None:
        dense = self.config["openmvs_dense_resolution"]
        self.step(
            f"{step_prefix}-{branch}-export-dmap",
            [
                self.map_python,
                self.script("export_model_depths_to_openmvs.py"),
                "--template-dmaps",
                self.c_openmvs,
                "--scene",
                self.c_openmvs / "scene.mvs",
                "--depth",
                depth,
                "--output",
                canonical,
                "--resolution-level",
                str(dense["resolution_level"]),
                "--min-resolution",
                str(dense["min_resolution"]),
                "--max-resolution",
                str(dense["max_resolution"]),
            ],
            canonical / "dmap-export.json",
        )
        validation_report = self.branch_root / f"{branch}_handoff_validation.json"
        self.step(
            f"{step_prefix}a-{branch}-validate-dmap",
            [
                self.map_python,
                self.script("validate_openmvs_handoff.py"),
                "--dmaps",
                canonical,
                "--template-dmaps",
                self.c_openmvs,
                "--manifest",
                canonical / "dmap-export.json",
                "--resolution-level",
                str(dense["resolution_level"]),
                "--min-resolution",
                str(dense["min_resolution"]),
                "--max-resolution",
                str(dense["max_resolution"]),
                "--report",
                validation_report,
            ],
            validation_report,
        )
        self.step(
            f"{step_prefix}b-{branch}-stage-openmvs",
            [
                self.map_python,
                self.script("stage_openmvs_handoff.py"),
                "stage",
                "--source",
                canonical,
                "--output",
                staging,
            ],
            staging / "staging-manifest.json",
        )

    def verify_external_handoff(
        self,
        branch: str,
        step_prefix: str,
        staging: Path,
    ) -> None:
        report = self.branch_root / f"{branch}_handoff_immutability.json"
        self.step(
            f"{step_prefix}-{branch}-verify-handoff",
            [
                self.map_python,
                self.script("stage_openmvs_handoff.py"),
                "verify",
                "--manifest",
                staging / "staging-manifest.json",
                "--report",
                report,
            ],
            report,
        )

    def branch_a(self, cameras: Path, source_width: int, source_height: int) -> None:
        cfg = self.config["a_v4"]
        raw = self.a_depth / "raw"
        calibrated = self.a_depth / "depth-sparse-calibrated"
        optimized = self.a_depth / "depth-v4"
        self.step(
            "20-a-mapanything",
            [
                self.map_python,
                self.script("run_mapanything_batched.py"),
                "--images",
                self.input / "images",
                "--cameras",
                cameras,
                "--output",
                raw,
                "--model",
                ROOT / cfg["model"],
                "--batch-size",
                str(cfg["batch_size"]),
                "--skip-tsdf",
            ],
            raw / "prediction_manifest.json",
        )
        self.step(
            "21-a-scale-calibration",
            [
                self.map_python,
                self.script("calibrate_depths_to_colmap_sparse.py"),
                "--depth",
                raw / "depth",
                "--reconstruction",
                self.shared / "colmap/mapper/0",
                "--output",
                calibrated,
                "--manifest",
                raw / "prediction_manifest.json",
                "--source-width",
                str(source_width),
                "--source-height",
                str(source_height),
                "--scale-mode",
                cfg["sparse_scale_mode"],
                "--minimum-points",
                str(cfg["minimum_sparse_points"]),
            ],
            calibrated / "sparse_calibration_manifest.json",
        )
        cross = cfg["cross_view"]
        self.step(
            "22-a-crossview-v4",
            [
                self.map_python,
                self.script("optimize_multiview_depths.py"),
                "--depth",
                calibrated,
                "--cameras",
                cameras,
                "--manifest",
                raw / "prediction_manifest.json",
                "--output",
                optimized,
                "--source-width",
                str(source_width),
                "--source-height",
                str(source_height),
                "--neighbors",
                str(cross["neighbors"]),
                "--min-support",
                str(cross["interior_minimum_support"]),
                "--edge-min-support",
                str(cross["edge_minimum_support"]),
                "--edge-relative-gradient",
                str(cross["edge_relative_gradient"]),
                "--relative-depth-threshold",
                str(cross["relative_depth_threshold"]),
                "--reprojection-threshold",
                str(cross["reprojection_threshold_pixels"]),
                "--normal-angle-threshold",
                str(cross["normal_angle_threshold_degrees"]),
                "--min-component-area",
                str(cross["minimum_component_area_pixels"]),
            ],
            optimized / "consistency_manifest.json",
        )
        self.prepare_external_handoff(
            "a", "23", optimized, self.a_openmvs_input, self.a_openmvs
        )
        threads = int(cfg["openmvs_threads"])
        self.external_fusion("24-a", self.a_openmvs, threads)
        self.verify_external_handoff("a", "24a", self.a_openmvs)
        self.reconstruct("25-a", self.a_openmvs, threads)
        self.texture("26-a", self.a_openmvs, "A-v4")

    def branch_b(self, cameras: Path, source_width: int, source_height: int) -> None:
        cfg = self.config["b_v2"]
        raw = self.b_depth / "raw"
        calibrated = self.b_depth / "depth-sparse-calibrated"
        optimized = self.b_depth / "depth-v2"
        self.step(
            "30-b-mvsanywhere",
            [
                self.mvs_python,
                self.script("run_mvsanywhere.py"),
                "--repo",
                ROOT / cfg["repo"],
                "--checkpoint",
                ROOT / cfg["checkpoint"],
                "--images",
                self.input / "images",
                "--cameras",
                cameras,
                "--output",
                raw,
                "--source-views",
                str(cfg["source_views"]),
                "--height",
                str(cfg["height"]),
                "--width",
                str(cfg["width"]),
                "--refinement-steps",
                str(cfg["refinement_steps"]),
                "--skip-tsdf",
            ],
            raw / "prediction_manifest.json",
        )
        self.step(
            "31-b-scale-calibration",
            [
                self.map_python,
                self.script("calibrate_depths_to_colmap_sparse.py"),
                "--depth",
                raw / "depth",
                "--reconstruction",
                self.shared / "colmap/mapper/0",
                "--output",
                calibrated,
                "--manifest",
                raw / "prediction_manifest.json",
                "--source-width",
                str(source_width),
                "--source-height",
                str(source_height),
                "--scale-mode",
                cfg["sparse_scale_mode"],
                "--minimum-points",
                str(cfg["minimum_sparse_points"]),
            ],
            calibrated / "sparse_calibration_manifest.json",
        )
        cross = cfg["cross_view"]
        self.step(
            "32-b-crossview-v2",
            [
                self.map_python,
                self.script("optimize_multiview_depths.py"),
                "--depth",
                calibrated,
                "--cameras",
                cameras,
                "--manifest",
                raw / "prediction_manifest.json",
                "--output",
                optimized,
                "--source-width",
                str(source_width),
                "--source-height",
                str(source_height),
                "--neighbors",
                str(cross["neighbors"]),
                "--min-support",
                str(cross["minimum_support"]),
                "--relative-depth-threshold",
                str(cross["relative_depth_threshold"]),
                "--reprojection-threshold",
                str(cross["reprojection_threshold_pixels"]),
                "--min-component-area",
                str(cross["minimum_component_area_pixels"]),
            ],
            optimized / "consistency_manifest.json",
        )
        self.prepare_external_handoff(
            "b", "33", optimized, self.b_openmvs_input, self.b_openmvs
        )
        threads = int(cfg["openmvs_threads"])
        self.external_fusion("34-b", self.b_openmvs, threads)
        self.verify_external_handoff("b", "34a", self.b_openmvs)
        self.reconstruct("35-b", self.b_openmvs, threads)
        self.texture("36-b", self.b_openmvs, "B-v2")

    def validate(self) -> None:
        labels = []
        if "a" in self.branches:
            labels.append("A-v4")
        if "b" in self.branches:
            labels.append("B-v2")
        if "c" in self.branches:
            labels.append("C")
        if not labels or self.dry_run:
            return
        self.step(
            "90-validate",
            [
                self.map_python,
                self.script("validate_outputs.py"),
                "--outputs",
                self.outputs,
                "--branches",
                *labels,
                "--report",
                self.outputs / "validation.json",
            ],
            self.outputs / "validation.json",
        )

    def run(self) -> None:
        width, height = self.prepare()
        cameras = self.cameras()
        # C depth maps are the camera-correct DMAP templates for A and B, so C
        # densification is a prerequisite even when only a learned branch is requested.
        self.c_geometry()
        if "c" in self.branches:
            self.c_final()
        if "a" in self.branches:
            self.branch_a(cameras, width, height)
        if "b" in self.branches:
            self.branch_b(cameras, width, height)
        self.validate()


def doctor(paths_path: Path) -> int:
    local = read_json(paths_path)
    checks = {
        "mapanything_python": resolve_runtime_path(
            os.environ.get("RE3D_MAP_PYTHON", local["mapanything_python"])
        ),
        "mvsanywhere_python": resolve_runtime_path(
            os.environ.get("RE3D_MVS_PYTHON", local["mvsanywhere_python"])
        ),
        "mapanything_model": ROOT / "models/mapanything/model.safetensors",
        "mvsanywhere_checkpoint": ROOT / "models/mvsanywhere/mvsanywhere_hero.ckpt",
        "dinov2_checkpoint": ROOT / "models/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth",
        "mapanything_source": ROOT / "vendor/map-anything/mapanything",
        "mvsanywhere_source": ROOT / "vendor/mvsanywhere/src/mvsanywhere",
        "InterfaceCOLMAP": DEFAULT_OPENMVS / "InterfaceCOLMAP.exe",
        "DensifyPointCloud": DEFAULT_OPENMVS / "DensifyPointCloud.exe",
        "ReconstructMesh": DEFAULT_OPENMVS / "ReconstructMesh.exe",
        "TextureMesh": resolve_texturemesh_path(local),
    }
    failed = False
    for name, path in checks.items():
        ok = path.exists() and (not path.is_file() or path.stat().st_size > 0)
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {path}")
        failed |= not ok
    if failed:
        return 1
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "vendor/map-anything"), str(ROOT / "vendor/mvsanywhere/src")]
    )
    env["TORCH_HOME"] = str(ROOT / "models/torch")
    import_checks = [
        (
            checks["mapanything_python"],
            "import cv2, numpy, pycolmap, scipy, torch, trimesh, mapanything; print('map environment imports: OK')",
        ),
        (
            checks["mvsanywhere_python"],
            "import cv2, numpy, open3d, torch, trimesh; print('mvs environment imports: OK')",
        ),
    ]
    for python, code in import_checks:
        completed = subprocess.run([str(python), "-c", code], env=env, text=True)
        failed |= completed.returncode != 0
    return 1 if failed else 0


def parse_branches(raw: str) -> set[str]:
    if raw.lower() == "all":
        return {"a", "b", "c"}
    branches = {value.strip().lower() for value in raw.split(",") if value.strip()}
    unknown = branches - {"a", "b", "c"}
    if unknown or not branches:
        raise ValueError(f"Invalid branches: {sorted(unknown or branches)}")
    return branches


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the selected Re3D A-v4, B-v2 and C pipelines.")
    parser.add_argument("--scene")
    parser.add_argument("--images", type=Path)
    parser.add_argument("--branches", default="all", help="all or comma-separated a,b,c")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/pipeline.json")
    parser.add_argument("--paths", type=Path, default=ROOT / "configs/paths.local.json")
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Exact per-scene work directory (overrides RE3D_WORK_DIR)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Exact per-scene output directory (overrides RE3D_OUTPUT_DIR)",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Exact per-scene log directory (overrides RE3D_LOG_DIR)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--doctor", action="store_true")
    args = parser.parse_args()
    if args.doctor:
        raise SystemExit(doctor(args.paths))
    if not args.scene:
        parser.error("--scene is required unless --doctor is used")
    pipeline = Pipeline(
        scene=args.scene,
        images=args.images,
        branches=parse_branches(args.branches),
        config_path=args.config,
        paths_path=args.paths,
        dry_run=args.dry_run,
        work_dir=args.work_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
    )
    pipeline.run()


if __name__ == "__main__":
    main()
