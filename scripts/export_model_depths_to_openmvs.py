from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from openmvs_dmap import json_header, read_dmap_header, sha256_file, write_dmap


ADAPTER_VERSION = "2.1"


def _resolve_manifest_path(raw: str, parent: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else (parent / path).resolve()


def load_depth_provenance(depth_directory: Path) -> tuple[dict, dict[str, float]]:
    provenance: dict = {}
    applied_scales: dict[str, float] = {}
    consistency_path = depth_directory / "consistency_manifest.json"
    if not consistency_path.is_file():
        return provenance, applied_scales
    consistency = json.loads(consistency_path.read_text(encoding="utf-8-sig"))
    provenance["consistency"] = {
        "manifest": str(consistency_path.resolve()),
        "sha256": sha256_file(consistency_path),
        "source_depth": consistency.get("source_depth"),
        "parameters": consistency.get("parameters"),
        "input_valid_pixels": consistency.get("input_valid_pixels"),
        "output_valid_pixels": consistency.get("output_valid_pixels"),
        "retained_valid_fraction": consistency.get("retained_valid_fraction"),
    }
    raw_source = consistency.get("source_depth")
    if not raw_source:
        return provenance, applied_scales
    calibrated_directory = _resolve_manifest_path(str(raw_source), consistency_path.parent)
    calibration_path = calibrated_directory / "sparse_calibration_manifest.json"
    if not calibration_path.is_file():
        return provenance, applied_scales
    calibration = json.loads(calibration_path.read_text(encoding="utf-8-sig"))
    provenance["sparse_calibration"] = {
        "manifest": str(calibration_path.resolve()),
        "sha256": sha256_file(calibration_path),
        "source_depth": calibration.get("source_depth"),
        "reconstruction": calibration.get("reconstruction"),
        "scale_mode": calibration.get("scale_mode"),
        "scale_statistics": calibration.get("scale_statistics"),
    }
    applied_scales = {
        str(record["name"]): float(record["applied_scale"])
        for record in calibration.get("records", [])
        if "name" in record and "applied_scale" in record
    }
    return provenance, applied_scales


def resize_masked_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    depth = np.nan_to_num(
        depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    valid = mask.astype(bool) & (depth > 0)
    weights = cv2.resize(
        valid.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    numerator = cv2.resize(
        np.where(valid, depth, 0),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = numerator / np.maximum(weights, 1e-6)
    resized_valid = weights >= 0.5
    resized[~resized_valid] = 0
    return resized.astype(np.float32), resized_valid


def resize_masked_confidence(
    confidence: np.ndarray,
    mask: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    confidence = np.nan_to_num(
        confidence.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    valid = mask.astype(bool)
    weights = cv2.resize(
        valid.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR
    )
    numerator = cv2.resize(
        np.where(valid, confidence, 0),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = numerator / np.maximum(weights, 1e-6)
    resized[weights < 0.5] = 0
    return np.clip(resized, 0, 1).astype(np.float32)


def _require_matching_source_shapes(
    image_name: str,
    depth: np.ndarray,
    mask: np.ndarray,
    confidence: np.ndarray,
) -> None:
    if depth.ndim != 2:
        raise ValueError(f"{image_name}: depth must be HxW, got {depth.shape}")
    if mask.shape != depth.shape:
        raise ValueError(
            f"{image_name}: mask/depth shape mismatch: {mask.shape} != {depth.shape}"
        )
    if confidence.shape != depth.shape:
        raise ValueError(
            f"{image_name}: confidence/depth shape mismatch: "
            f"{confidence.shape} != {depth.shape}"
        )


def export(args: argparse.Namespace) -> dict:
    templates = sorted(args.template_dmaps.glob("depth*.dmap"))
    if not templates:
        raise FileNotFoundError(f"No template DMAPs in {args.template_dmaps}")
    if not args.scene.is_file():
        raise FileNotFoundError(f"Missing OpenMVS scene: {args.scene}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(
            f"Canonical handoff must be created in an empty directory: {args.output}"
        )

    temporary = args.output.with_name(args.output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Remove stale temporary export first: {temporary}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        args.output.rmdir()
    temporary.mkdir()

    provenance, applied_scales = load_depth_provenance(args.depth)
    records: list[dict] = []
    try:
        output_scene = temporary / "scene.mvs"
        shutil.copy2(args.scene, output_scene)
        for template in templates:
            header = read_dmap_header(template)
            image_size = (int(header["image_width"]), int(header["image_height"]))
            depth_size = (int(header["depth_width"]), int(header["depth_height"]))
            if image_size != depth_size:
                raise ValueError(
                    f"Template {template.name} violates the OpenMVS handoff "
                    f"resolution contract: image={image_size}, depth={depth_size}"
                )
            if int(header["compression"]) != 0:
                raise ValueError(f"Compressed template DMAP is unsupported: {template}")

            image_name = Path(header["file_name"]).name
            stem = Path(image_name).stem
            depth_path = args.depth / f"{stem}_depth.npy"
            mask_path = args.depth / f"{stem}_mask.npy"
            confidence_path = args.depth / f"{stem}_confidence.npy"
            missing = [
                path
                for path in (depth_path, mask_path, confidence_path)
                if not path.is_file()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Missing model handoff inputs for {image_name}: {missing}"
                )

            depth = np.load(depth_path, allow_pickle=False)
            mask = np.load(mask_path, allow_pickle=False)
            confidence = np.load(confidence_path, allow_pickle=False)
            _require_matching_source_shapes(image_name, depth, mask, confidence)
            width, height = depth_size
            resized, valid = resize_masked_depth(depth, mask, width, height)
            resized_confidence = resize_masked_confidence(
                confidence, mask, width, height
            )
            resized_confidence[~valid] = 0
            output_path = temporary / template.name
            write_dmap(output_path, header, resized, resized_confidence)
            records.append(
                {
                    "dmap": output_path.name,
                    "image": image_name,
                    "template_sha256": sha256_file(template),
                    "output_sha256": sha256_file(output_path),
                    "source_files": {
                        "depth": str(depth_path.resolve()),
                        "mask": str(mask_path.resolve()),
                        "confidence": str(confidence_path.resolve()),
                    },
                    "source_sha256": {
                        "depth": sha256_file(depth_path),
                        "mask": sha256_file(mask_path),
                        "confidence": sha256_file(confidence_path),
                    },
                    "source_shape": list(depth.shape),
                    "output_shape": list(resized.shape),
                    "applied_calibration_scale": applied_scales.get(image_name),
                    "valid_ratio": float(valid.mean()),
                    "depth_min": (
                        float(resized[valid].min()) if np.any(valid) else 0.0
                    ),
                    "depth_max": (
                        float(resized[valid].max()) if np.any(valid) else 0.0
                    ),
                    "mean_confidence": (
                        float(resized_confidence[valid].mean())
                        if np.any(valid)
                        else 0.0
                    ),
                    "template_header": json_header(header),
                }
            )

        manifest = {
            "schema_version": 2,
            "adapter": "export_model_depths_to_openmvs",
            "adapter_version": ADAPTER_VERSION,
            "scene": str(args.scene.resolve()),
            "scene_sha256": sha256_file(args.scene),
            "output_scene_sha256": sha256_file(output_scene),
            "template_dmaps": str(args.template_dmaps.resolve()),
            "source_depth": str(args.depth.resolve()),
            "depth_provenance": provenance,
            "count": len(records),
            "content_type": 5,
            "depth_convention": args.depth_convention,
            "confidence_semantics": args.confidence_semantics,
            "openmvs_dense_resolution": {
                "resolution_level": args.resolution_level,
                "min_resolution": args.min_resolution,
                "max_resolution": args.max_resolution,
            },
            "mean_valid_ratio": float(
                np.mean([record["valid_ratio"] for record in records])
            ),
            "method": (
                "write depth and confidence payloads while preserving the exact "
                "OpenMVS template camera metadata"
            ),
            "records": records,
        }
        (temporary / "dmap-export.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, args.output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an immutable OpenMVS depth/confidence DMAP handoff."
    )
    parser.add_argument("--template-dmaps", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resolution-level", type=int, required=True)
    parser.add_argument("--min-resolution", type=int, required=True)
    parser.add_argument("--max-resolution", type=int, required=True)
    parser.add_argument("--depth-convention", default="camera_z")
    parser.add_argument(
        "--confidence-semantics", default="cross_view_support_ratio"
    )
    args = parser.parse_args()
    manifest = export(args)
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "records"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
