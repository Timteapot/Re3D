from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from openmvs_dmap import read_dmap, read_dmap_header, sha256_file


def _same_metadata(actual: dict, template: dict) -> list[str]:
    errors: list[str] = []
    scalar_keys = (
        "image_width",
        "image_height",
        "depth_width",
        "depth_height",
        "file_name",
    )
    for key in scalar_keys:
        if actual[key] != template[key]:
            errors.append(f"metadata {key}: {actual[key]!r} != {template[key]!r}")
    if not np.array_equal(actual["view_ids"], template["view_ids"]):
        errors.append("metadata view_ids differ from template")
    for key in ("K", "R", "C"):
        if not np.array_equal(actual[key], template[key]):
            errors.append(f"metadata {key} differs from template")
    return errors


def _roundtrip_reprojection_error(header: dict, depth: np.ndarray) -> float:
    intrinsics = np.asarray(header["K"], dtype=np.float64)
    if not np.isfinite(np.linalg.cond(intrinsics)) or np.linalg.cond(intrinsics) > 1e12:
        raise ValueError("K is singular or numerically unstable")
    coordinates = np.argwhere(depth > 0)
    if not len(coordinates):
        return float("inf")
    selection = np.linspace(0, len(coordinates) - 1, min(16, len(coordinates))).astype(int)
    samples = coordinates[selection]
    pixels = np.column_stack(
        (samples[:, 1], samples[:, 0], np.ones(len(samples), dtype=np.float64))
    )
    rays = (np.linalg.inv(intrinsics) @ pixels.T).T
    sample_depth = depth[samples[:, 0], samples[:, 1]].astype(np.float64)
    camera_points = rays * (sample_depth / rays[:, 2])[:, None]
    rotation = np.asarray(header["R"], dtype=np.float64)
    center = np.asarray(header["C"], dtype=np.float64)
    world_points = (rotation.T @ camera_points.T).T + center
    reconstructed_camera = (rotation @ (world_points - center).T).T
    projected = (intrinsics @ reconstructed_camera.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    return float(np.linalg.norm(projected - pixels[:, :2], axis=1).max())


def validate_handoff(
    dmaps: Path,
    template_dmaps: Path,
    manifest_path: Path,
    expected_resolution: dict[str, int],
) -> dict:
    errors: list[str] = []
    records: list[dict] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {
            "schema_version": 1,
            "all_passed": False,
            "errors": [f"Cannot read manifest {manifest_path}: {exc}"],
            "records": [],
        }

    if manifest.get("schema_version") != 2:
        errors.append(f"manifest schema_version must be 2, got {manifest.get('schema_version')!r}")
    if manifest.get("content_type") != 5:
        errors.append(f"manifest content_type must be depth+confidence (5)")
    if manifest.get("depth_convention") != "camera_z":
        errors.append("manifest depth_convention must be camera_z")
    if manifest.get("confidence_semantics") != "cross_view_support_ratio":
        errors.append(
            "manifest confidence_semantics must be cross_view_support_ratio"
        )
    if manifest.get("openmvs_dense_resolution") != expected_resolution:
        errors.append(
            "resolution contract mismatch: "
            f"manifest={manifest.get('openmvs_dense_resolution')}, "
            f"runtime={expected_resolution}"
        )
    for stage, provenance in manifest.get("depth_provenance", {}).items():
        provenance_path = Path(str(provenance.get("manifest", "")))
        if not provenance_path.is_file():
            errors.append(f"{stage} provenance manifest is missing: {provenance_path}")
        elif sha256_file(provenance_path) != provenance.get("sha256"):
            errors.append(f"{stage} provenance manifest hash changed")

    dmap_paths = sorted(dmaps.glob("depth*.dmap"))
    template_paths = sorted(template_dmaps.glob("depth*.dmap"))
    actual_names = [path.name for path in dmap_paths]
    template_names = [path.name for path in template_paths]
    manifest_records = manifest.get("records", [])
    manifest_names = [record.get("dmap") for record in manifest_records]
    if not dmap_paths:
        errors.append(f"no depth*.dmap files in {dmaps}")
    if actual_names != template_names:
        errors.append("exported/template DMAP filename sets differ")
    if actual_names != manifest_names:
        errors.append("exported/manifest DMAP order or filename sets differ")
    if manifest.get("count") != len(dmap_paths):
        errors.append(
            f"manifest count {manifest.get('count')} != actual {len(dmap_paths)}"
        )

    scene_path = dmaps / "scene.mvs"
    if not scene_path.is_file():
        errors.append(f"missing scene.mvs in {dmaps}")
    elif sha256_file(scene_path) != manifest.get("output_scene_sha256"):
        errors.append("scene.mvs hash differs from export manifest")

    template_by_name = {path.name: path for path in template_paths}
    template_headers = {
        path.name: read_dmap_header(path) for path in template_paths
    }
    scene_view_ids = {
        int(header["view_ids"][0])
        for header in template_headers.values()
        if len(header["view_ids"])
    }
    if len(scene_view_ids) != len(template_paths):
        errors.append(
            "template primary view IDs are missing or not unique across the scene"
        )
    record_by_name = {record.get("dmap"): record for record in manifest_records}
    total_valid = 0
    total_pixels = 0
    for dmap_path in dmap_paths:
        file_errors: list[str] = []
        roundtrip_error: float | None = None
        try:
            header, payload = read_dmap(dmap_path)
            template_path = template_by_name.get(dmap_path.name)
            record = record_by_name.get(dmap_path.name)
            if template_path is None:
                raise ValueError("matching template is absent")
            if record is None:
                raise ValueError("matching manifest record is absent")
            template = template_headers[dmap_path.name]
            file_errors.extend(_same_metadata(header, template))
            if int(header["content_type"]) != 5:
                file_errors.append(
                    f"content_type is {header['content_type']}, expected 5"
                )
            if int(header["compression"]) != 0:
                file_errors.append("compression must be 0")
            image_size = (int(header["image_width"]), int(header["image_height"]))
            depth_size = (int(header["depth_width"]), int(header["depth_height"]))
            if image_size != depth_size:
                file_errors.append(
                    f"image/depth resolution mismatch: {image_size} != {depth_size}"
                )
            view_ids = np.asarray(header["view_ids"], dtype=np.uint32)
            if not len(view_ids):
                file_errors.append("view_ids are empty")
            if len(np.unique(view_ids)) != len(view_ids):
                file_errors.append("view_ids contain duplicates")
            unknown_view_ids = sorted(
                int(view_id)
                for view_id in view_ids
                if int(view_id) not in scene_view_ids
            )
            if unknown_view_ids:
                file_errors.append(
                    f"view_ids are absent from the template scene: {unknown_view_ids}"
                )
            for key in ("K", "R", "C"):
                if not np.all(np.isfinite(header[key])):
                    file_errors.append(f"{key} contains non-finite values")
            rotation = np.asarray(header["R"], dtype=np.float64)
            if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
                file_errors.append("R is not orthonormal")
            if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4):
                file_errors.append("det(R) is not +1")

            depth = payload["depth"]
            confidence = payload.get("confidence")
            if confidence is None:
                file_errors.append("confidence payload is absent")
            else:
                if not np.all(np.isfinite(confidence)):
                    file_errors.append("confidence contains non-finite values")
                if np.any((confidence < 0) | (confidence > 1)):
                    file_errors.append("confidence is outside [0, 1]")
            if not np.all(np.isfinite(depth)):
                file_errors.append("depth contains non-finite values")
            if np.any(depth < 0):
                file_errors.append("depth contains negative values")
            valid = depth > 0
            total_valid += int(valid.sum())
            total_pixels += int(depth.size)
            if not np.any(valid):
                file_errors.append("depth contains no valid positive samples")
            else:
                actual_min = float(depth[valid].min())
                actual_max = float(depth[valid].max())
                if not np.isclose(actual_min, header["depth_min"], rtol=1e-5):
                    file_errors.append("header depth_min does not match payload")
                if not np.isclose(actual_max, header["depth_max"], rtol=1e-5):
                    file_errors.append("header depth_max does not match payload")
            if confidence is not None and np.any(confidence[~valid] != 0):
                file_errors.append("confidence must be zero where depth is invalid")
            roundtrip_error = _roundtrip_reprojection_error(header, depth)
            if not np.isfinite(roundtrip_error) or roundtrip_error > 1e-5:
                file_errors.append(
                    "camera unproject/project round-trip is unstable: "
                    f"{roundtrip_error:.6g} px"
                )
            if sha256_file(dmap_path) != record.get("output_sha256"):
                file_errors.append("DMAP hash differs from export manifest")
            if sha256_file(template_path) != record.get("template_sha256"):
                file_errors.append("template DMAP hash differs from export manifest")
            if record.get("output_shape") != list(depth.shape):
                file_errors.append("payload shape differs from export manifest")
            for source_name, raw_path in record.get("source_files", {}).items():
                source_path = Path(str(raw_path))
                if not source_path.is_file():
                    file_errors.append(f"source {source_name} is missing")
                elif sha256_file(source_path) != record.get("source_sha256", {}).get(
                    source_name
                ):
                    file_errors.append(f"source {source_name} hash changed")
            applied_scale = record.get("applied_calibration_scale")
            if applied_scale is not None and not np.isfinite(applied_scale):
                file_errors.append("applied calibration scale is not finite")
        except Exception as exc:
            file_errors.append(str(exc))

        if file_errors:
            errors.extend(f"{dmap_path.name}: {message}" for message in file_errors)
        records.append(
            {
                "dmap": dmap_path.name,
                "passed": not file_errors,
                "max_roundtrip_reprojection_error_pixels": roundtrip_error,
                "errors": file_errors,
            }
        )

    return {
        "schema_version": 1,
        "all_passed": not errors,
        "dmaps": str(dmaps.resolve()),
        "template_dmaps": str(template_dmaps.resolve()),
        "manifest": str(manifest_path.resolve()),
        "resolution_contract": expected_resolution,
        "dmap_count": len(dmap_paths),
        "valid_depth_samples": total_valid,
        "total_depth_samples": total_pixels,
        "valid_depth_ratio": total_valid / total_pixels if total_pixels else 0.0,
        "errors": errors,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate an external OpenMVS DMAP handoff before fusion."
    )
    parser.add_argument("--dmaps", type=Path, required=True)
    parser.add_argument("--template-dmaps", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--resolution-level", type=int, required=True)
    parser.add_argument("--min-resolution", type=int, required=True)
    parser.add_argument("--max-resolution", type=int, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    expected_resolution = {
        "resolution_level": args.resolution_level,
        "min_resolution": args.min_resolution,
        "max_resolution": args.max_resolution,
    }
    report = validate_handoff(
        args.dmaps,
        args.template_dmaps,
        args.manifest,
        expected_resolution,
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = {key: value for key, value in report.items() if key != "records"}
    print(json.dumps(summary, indent=2))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
