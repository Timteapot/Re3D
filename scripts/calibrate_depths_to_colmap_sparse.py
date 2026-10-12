from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pycolmap


def sample_masked(values: np.ndarray, mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    map_x = x.astype(np.float32).reshape(-1, 1)
    map_y = y.astype(np.float32).reshape(-1, 1)
    weights = cv2.remap(
        mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    ).reshape(-1)
    sampled = cv2.remap(
        np.where(mask, values, 0).astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    ).reshape(-1)
    sampled /= np.maximum(weights, 1e-6)
    return sampled, weights >= 0.75


def robust_log_inliers(ratios: np.ndarray) -> np.ndarray:
    logs = np.log(ratios)
    median = np.median(logs)
    mad = np.median(np.abs(logs - median))
    limit = max(3.5 * 1.4826 * mad, 0.015)
    return np.abs(logs - median) <= limit


def select_scale_ratios(
    ratios: np.ndarray,
    minimum_points: int,
    ratio_minimum: float = 0.5,
    ratio_maximum: float = 2.0,
    fallback_maximum_p90_p10_ratio: float = 2.0,
) -> tuple[np.ndarray, dict]:
    """Select a stable set of sparse/predicted depth ratios.

    The fixed ratio window remains the preferred path. If it contains too few
    robust inliers, retry against every finite positive ratio. The fallback is
    accepted only when the robust inlier distribution is sufficiently tight,
    so a globally shifted but coherent learned-depth scale can be recovered
    without accepting an arbitrary multimodal or unstable estimate.
    """
    if minimum_points < 1:
        raise ValueError("minimum_points must be at least 1")
    if not 0 < ratio_minimum < ratio_maximum:
        raise ValueError("ratio bounds must satisfy 0 < minimum < maximum")
    if fallback_maximum_p90_p10_ratio < 1:
        raise ValueError("fallback P90/P10 limit must be at least 1")

    values = np.asarray(ratios, dtype=np.float64).reshape(-1)
    positive = values[np.isfinite(values) & (values > 0)]
    bounded = positive[(positive > ratio_minimum) & (positive < ratio_maximum)]
    bounded_inliers = (
        bounded[robust_log_inliers(bounded)] if len(bounded) else bounded
    )

    mode = "bounded"
    selected = bounded_inliers
    if len(selected) < minimum_points:
        mode = "adaptive"
        selected = (
            positive[robust_log_inliers(positive)] if len(positive) else positive
        )

    if len(selected) < minimum_points:
        raise ValueError(
            f"only {len(selected)} robust scale samples "
            f"({len(bounded)} bounded, {len(positive)} positive)"
        )

    ratio_p10, ratio_p90 = np.percentile(selected, [10, 90])
    p90_p10_ratio = float(ratio_p90 / ratio_p10)
    if mode == "adaptive" and p90_p10_ratio > fallback_maximum_p90_p10_ratio:
        raise ValueError(
            "adaptive scale samples are too dispersed: "
            f"P90/P10={p90_p10_ratio:.6g} exceeds "
            f"{fallback_maximum_p90_p10_ratio:.6g}"
        )

    return selected, {
        "selection_mode": mode,
        "positive_samples": int(len(positive)),
        "bounded_samples": int(len(bounded)),
        "bounded_inliers": int(len(bounded_inliers)),
        "selected_samples": int(len(selected)),
        "selected_p90_p10_ratio": p90_p10_ratio,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate learned depth scale against registered COLMAP sparse points."
    )
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--reconstruction", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-height", type=int, default=800)
    parser.add_argument("--source-width", type=int, default=800)
    parser.add_argument("--scale-mode", choices=["image", "group"], default="image")
    parser.add_argument("--group-field", default="batch")
    parser.add_argument("--minimum-points", type=int, default=30)
    parser.add_argument("--ratio-minimum", type=float, default=0.5)
    parser.add_argument("--ratio-maximum", type=float, default=2.0)
    parser.add_argument("--fallback-maximum-p90-p10-ratio", type=float, default=2.0)
    args = parser.parse_args()

    reconstruction = pycolmap.Reconstruction(str(args.reconstruction))
    images_by_name = {image.name: image for image in reconstruction.images.values()}
    images_by_stem = {Path(name).stem: image for name, image in images_by_name.items()}
    names_by_stem = {Path(name).stem: name for name in images_by_name}
    if len(images_by_stem) != len(images_by_name):
        raise RuntimeError("Image stems must be unique across file extensions")
    metadata: dict[str, dict] = {}
    if args.manifest and args.manifest.exists():
        metadata = {
            str(record["name"]): record
            for record in json.loads(args.manifest.read_text(encoding="utf-8"))
        }

    estimates: dict[str, dict] = {}
    group_logs: dict[str, list[np.ndarray]] = defaultdict(list)
    depth_paths = sorted(args.depth.glob("*_depth.npy"))
    for depth_path in depth_paths:
        stem = depth_path.name.removesuffix("_depth.npy")
        if stem not in images_by_stem:
            raise KeyError(f"No registered COLMAP image for depth stem {stem}")
        name = names_by_stem[stem]
        image = images_by_stem[stem]
        depth = np.load(depth_path).astype(np.float32)
        mask = np.load(args.depth / f"{stem}_mask.npy").astype(bool)
        points = [point for point in image.points2D if point.has_point3D()]
        xy = np.asarray([point.xy for point in points], dtype=np.float64)
        xyz = np.asarray(
            [reconstruction.points3D[point.point3D_id].xyz for point in points], dtype=np.float64
        )
        cam_from_world = image.cam_from_world().matrix()
        xyz_h = np.column_stack([xyz, np.ones(len(xyz))])
        sparse_depth = (xyz_h @ cam_from_world.T)[:, 2]
        sample_x = xy[:, 0] * depth.shape[1] / args.source_width
        sample_y = xy[:, 1] * depth.shape[0] / args.source_height
        predicted, valid = sample_masked(depth, mask, sample_x, sample_y)
        valid &= np.isfinite(sparse_depth) & (sparse_depth > 0) & (predicted > 0)
        ratios = sparse_depth[valid] / predicted[valid]
        try:
            ratios, selection = select_scale_ratios(
                ratios,
                minimum_points=args.minimum_points,
                ratio_minimum=args.ratio_minimum,
                ratio_maximum=args.ratio_maximum,
                fallback_maximum_p90_p10_ratio=args.fallback_maximum_p90_p10_ratio,
            )
        except ValueError as error:
            raise RuntimeError(f"Cannot calibrate sparse depth scale for {name}: {error}") from error
        log_ratios = np.log(ratios)
        image_scale = float(np.exp(np.median(log_ratios)))
        group_value = metadata.get(name, {}).get(args.group_field, "all")
        group_key = str(group_value)
        group_logs[group_key].append(log_ratios)
        estimates[name] = {
            "samples": int(len(ratios)),
            "image_scale": image_scale,
            "ratio_p10": float(np.percentile(ratios, 10)),
            "ratio_p90": float(np.percentile(ratios, 90)),
            "group": group_key,
            **selection,
        }

    group_scales = {
        group: float(np.exp(np.median(np.concatenate(logs)))) for group, logs in group_logs.items()
    }
    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    for depth_path in depth_paths:
        stem = depth_path.name.removesuffix("_depth.npy")
        name = names_by_stem[stem]
        estimate = estimates[name]
        scale = (
            estimate["image_scale"]
            if args.scale_mode == "image"
            else group_scales[estimate["group"]]
        )
        depth = np.load(depth_path).astype(np.float32) * scale
        mask = np.load(args.depth / f"{stem}_mask.npy").astype(bool)
        depth[~mask] = 0
        np.save(args.output / f"{stem}_depth.npy", depth)
        np.save(args.output / f"{stem}_mask.npy", mask)
        records.append(
            {
                "name": name,
                **estimate,
                "applied_scale": float(scale),
            }
        )

    scales = np.asarray([record["applied_scale"] for record in records])
    summary = {
        "source_depth": str(args.depth.resolve()),
        "reconstruction": str(args.reconstruction.resolve()),
        "output": str(args.output.resolve()),
        "scale_mode": args.scale_mode,
        "group_field": args.group_field if args.scale_mode == "group" else None,
        "group_scales": group_scales,
        "selection_policy": {
            "preferred_ratio_bounds": [args.ratio_minimum, args.ratio_maximum],
            "minimum_points": args.minimum_points,
            "fallback": "finite positive ratios filtered by log-MAD",
            "fallback_maximum_p90_p10_ratio": args.fallback_maximum_p90_p10_ratio,
            "fallback_images": [
                record["name"]
                for record in records
                if record["selection_mode"] == "adaptive"
            ],
        },
        "scale_statistics": {
            "minimum": float(scales.min()),
            "median": float(np.median(scales)),
            "maximum": float(scales.max()),
        },
        "records": records,
    }
    (args.output / "sparse_calibration_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
