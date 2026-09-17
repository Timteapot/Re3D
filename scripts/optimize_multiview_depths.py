from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import cv2
import numpy as np


def scaled_intrinsics(k: np.ndarray, source_hw: tuple[int, int], depth_hw: tuple[int, int]) -> np.ndarray:
    source_h, source_w = source_hw
    depth_h, depth_w = depth_hw
    result = k.astype(np.float64).copy()
    result[0] *= depth_w / source_w
    result[1] *= depth_h / source_h
    return result


def nearest_neighbors(poses: np.ndarray, current: int, count: int) -> list[int]:
    centers = poses[:, :3, 3]
    distances = np.linalg.norm(centers - centers[current], axis=1)
    return [int(index) for index in np.argsort(distances) if int(index) != current][:count]


def sample_masked_depth(
    depth: np.ndarray, mask: np.ndarray, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    weights = cv2.remap(
        mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    numerator = cv2.remap(
        np.where(mask, depth, 0).astype(np.float32),
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    sampled = numerator / np.maximum(weights, 1e-6)
    return sampled.astype(np.float32), weights >= 0.75


def depth_edges(depth: np.ndarray, mask: np.ndarray, threshold: float) -> np.ndarray:
    safe = np.where(mask, depth, np.nan)
    dx = np.zeros_like(depth, dtype=np.float32)
    dy = np.zeros_like(depth, dtype=np.float32)
    dx[:, 1:-1] = np.abs(safe[:, 2:] - safe[:, :-2]) / np.maximum(
        2.0 * safe[:, 1:-1], 1e-6
    )
    dy[1:-1, :] = np.abs(safe[2:, :] - safe[:-2, :]) / np.maximum(
        2.0 * safe[1:-1, :], 1e-6
    )
    invalid_neighborhood = cv2.dilate((~mask).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    return mask & (invalid_neighborhood | (np.nan_to_num(np.maximum(dx, dy)) >= threshold))


def world_normals(
    depth: np.ndarray, mask: np.ndarray, intrinsics: np.ndarray, pose: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    x = (xx - intrinsics[0, 2]) / intrinsics[0, 0] * depth
    y = (yy - intrinsics[1, 2]) / intrinsics[1, 1] * depth
    points = np.stack([x, y, depth], axis=-1).astype(np.float32)
    normal = np.zeros_like(points)
    horizontal = points[:, 2:] - points[:, :-2]
    vertical = points[2:] - points[:-2]
    normal[1:-1, 1:-1] = np.cross(horizontal[1:-1], vertical[:, 1:-1])
    normal_valid = mask.copy()
    kernel = np.ones((3, 3), np.uint8)
    normal_valid &= cv2.erode(mask.astype(np.uint8), kernel) > 0
    length = np.linalg.norm(normal, axis=2)
    normal_valid &= np.isfinite(length) & (length > 1e-8)
    normal /= np.maximum(length[..., None], 1e-8)
    rotation = pose[:3, :3]
    normal_world = normal @ rotation.T
    normal_world[~normal_valid] = 0
    return normal_world.astype(np.float32), normal_valid


def sample_masked_vectors(
    values: np.ndarray, mask: np.ndarray, u: np.ndarray, v: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    map_x = u.astype(np.float32)
    map_y = v.astype(np.float32)
    weights = cv2.remap(
        mask.astype(np.float32), map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT
    )
    sampled_channels = [
        cv2.remap(
            np.where(mask, values[..., channel], 0).astype(np.float32),
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        / np.maximum(weights, 1e-6)
        for channel in range(values.shape[2])
    ]
    sampled = np.stack(sampled_channels, axis=-1)
    length = np.linalg.norm(sampled, axis=2)
    valid = (weights >= 0.75) & np.isfinite(length) & (length > 1e-6)
    sampled /= np.maximum(length[..., None], 1e-6)
    return sampled.astype(np.float32), valid


def project_target_to_neighbor(
    depth: np.ndarray,
    mask: np.ndarray,
    k_target: np.ndarray,
    pose_target: np.ndarray,
    neighbor_depth: np.ndarray,
    neighbor_mask: np.ndarray,
    k_neighbor: np.ndarray,
    pose_neighbor: np.ndarray,
    target_normals: np.ndarray | None = None,
    target_normal_mask: np.ndarray | None = None,
    neighbor_normals: np.ndarray | None = None,
    neighbor_normal_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = depth.shape
    yy, xx = np.mgrid[:height, :width]
    x = (xx - k_target[0, 2]) / k_target[0, 0] * depth
    y = (yy - k_target[1, 2]) / k_target[1, 1] * depth
    points_target = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)
    points_world = points_target @ pose_target.T
    world_to_neighbor = np.linalg.inv(pose_neighbor)
    points_neighbor = points_world @ world_to_neighbor.T
    projected_z = points_neighbor[..., 2]
    projected = points_neighbor[..., :3] @ k_neighbor.T
    u = projected[..., 0] / np.maximum(projected[..., 2], 1e-9)
    v = projected[..., 1] / np.maximum(projected[..., 2], 1e-9)

    sampled_depth, sampled_valid = sample_masked_depth(neighbor_depth, neighbor_mask, u, v)
    in_bounds = (
        (u >= 0)
        & (u <= neighbor_depth.shape[1] - 1)
        & (v >= 0)
        & (v <= neighbor_depth.shape[0] - 1)
    )
    observable = mask & (projected_z > 0) & in_bounds & sampled_valid & (sampled_depth > 0)

    # Back-project the neighbor observation and measure whether it returns to
    # the original target pixel. This rejects occlusion and silhouette mixing.
    nx = (u - k_neighbor[0, 2]) / k_neighbor[0, 0] * sampled_depth
    ny = (v - k_neighbor[1, 2]) / k_neighbor[1, 1] * sampled_depth
    neighbor_points = np.stack([nx, ny, sampled_depth, np.ones_like(sampled_depth)], axis=-1)
    neighbor_world = neighbor_points @ pose_neighbor.T
    back_target = neighbor_world @ np.linalg.inv(pose_target).T
    back_projected = back_target[..., :3] @ k_target.T
    back_u = back_projected[..., 0] / np.maximum(back_projected[..., 2], 1e-9)
    back_v = back_projected[..., 1] / np.maximum(back_projected[..., 2], 1e-9)
    reprojection_error = np.sqrt((back_u - xx) ** 2 + (back_v - yy) ** 2)

    relative_depth_error = np.abs(sampled_depth - projected_z) / np.maximum(
        np.maximum(sampled_depth, projected_z), 1e-6
    )
    candidate_target_depth = back_target[..., 2]
    normal_dot = np.ones_like(depth, dtype=np.float32)
    comparable_normals = np.zeros_like(mask, dtype=bool)
    if (
        target_normals is not None
        and target_normal_mask is not None
        and neighbor_normals is not None
        and neighbor_normal_mask is not None
    ):
        sampled_normals, sampled_normal_valid = sample_masked_vectors(
            neighbor_normals, neighbor_normal_mask, u, v
        )
        comparable_normals = observable & target_normal_mask & sampled_normal_valid
        normal_dot[comparable_normals] = np.abs(
            np.sum(target_normals[comparable_normals] * sampled_normals[comparable_normals], axis=1)
        )
    return (
        observable,
        relative_depth_error,
        reprojection_error,
        candidate_target_depth,
        normal_dot,
        comparable_normals,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter and consolidate learned depths using calibrated cross-view reprojection."
    )
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--source-height", type=int, default=800)
    parser.add_argument("--source-width", type=int, default=800)
    parser.add_argument("--neighbors", type=int, default=7)
    parser.add_argument("--min-support", type=int, default=2)
    parser.add_argument("--edge-min-support", type=int)
    parser.add_argument("--edge-relative-gradient", type=float, default=0.02)
    parser.add_argument("--relative-depth-threshold", type=float, default=0.04)
    parser.add_argument("--reprojection-threshold", type=float, default=2.0)
    parser.add_argument("--normal-angle-threshold", type=float, default=0.0)
    parser.add_argument("--min-component-area", type=int, default=32)
    args = parser.parse_args()

    camera_data = np.load(args.cameras, allow_pickle=False)
    names = [str(value) for value in camera_data["names"]]
    poses = camera_data["camera_poses"].astype(np.float64)
    base_intrinsics = camera_data["intrinsics"].astype(np.float64)
    depths: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    intrinsics: list[np.ndarray] = []
    for index, name in enumerate(names):
        stem = Path(name).stem
        depth = np.load(args.depth / f"{stem}_depth.npy").astype(np.float32)
        mask = np.load(args.depth / f"{stem}_mask.npy").astype(bool)
        mask &= np.isfinite(depth) & (depth > 0)
        depth = np.where(mask, depth, 0).astype(np.float32)
        depths.append(depth)
        masks.append(mask)
        intrinsics.append(
            scaled_intrinsics(
                base_intrinsics[index],
                (args.source_height, args.source_width),
                depth.shape,
            )
        )

    manifest_by_name: dict[str, dict] = {}
    if args.manifest and args.manifest.exists():
        manifest_by_name = {
            str(record["name"]): record
            for record in json.loads(args.manifest.read_text(encoding="utf-8"))
        }
    name_to_index = {name: index for index, name in enumerate(names)}
    use_normals = args.normal_angle_threshold > 0
    normals: list[np.ndarray] = []
    normal_masks: list[np.ndarray] = []
    if use_normals:
        for depth, mask, k, pose in zip(depths, masks, intrinsics, poses):
            normal, normal_mask = world_normals(depth, mask, k, pose)
            normals.append(normal)
            normal_masks.append(normal_mask)
    normal_cosine = float(np.cos(np.deg2rad(args.normal_angle_threshold)))
    edge_min_support = args.edge_min_support or args.min_support

    args.output.mkdir(parents=True, exist_ok=True)
    records = []
    all_observed_errors: list[np.ndarray] = []
    all_supported_errors: list[np.ndarray] = []
    total_input = 0
    total_output = 0
    total_observations = 0
    total_supports = 0

    for current, name in enumerate(names):
        record = manifest_by_name.get(name, {})
        source_names = record.get("source_views", [])
        if source_names:
            neighbors = [name_to_index[value] for value in source_names if value in name_to_index]
            neighbors = neighbors[: args.neighbors]
        else:
            neighbors = nearest_neighbors(poses, current, args.neighbors)

        depth = depths[current]
        input_mask = masks[current]
        edge_mask = depth_edges(depth, input_mask, args.edge_relative_gradient)
        observations = np.zeros(depth.shape, dtype=np.uint8)
        support = np.zeros(depth.shape, dtype=np.uint8)
        candidate_depths = [np.where(input_mask, depth, np.nan)]
        image_observed_errors: list[np.ndarray] = []
        image_supported_errors: list[np.ndarray] = []

        for neighbor in neighbors:
            (
                observable,
                relative_error,
                reprojection_error,
                candidate,
                normal_dot,
                comparable_normals,
            ) = project_target_to_neighbor(
                depth,
                input_mask,
                intrinsics[current],
                poses[current],
                depths[neighbor],
                masks[neighbor],
                intrinsics[neighbor],
                poses[neighbor],
                normals[current] if use_normals else None,
                normal_masks[current] if use_normals else None,
                normals[neighbor] if use_normals else None,
                normal_masks[neighbor] if use_normals else None,
            )
            normal_consistent = (~comparable_normals) | (normal_dot >= normal_cosine)
            consistent = (
                observable
                & (relative_error <= args.relative_depth_threshold)
                & (reprojection_error <= args.reprojection_threshold)
                & normal_consistent
                & np.isfinite(candidate)
                & (candidate > 0)
            )
            observations += observable.astype(np.uint8)
            support += consistent.astype(np.uint8)
            candidate_depths.append(np.where(consistent, candidate, np.nan).astype(np.float32))
            if np.any(observable):
                image_observed_errors.append(relative_error[observable].astype(np.float32))
            if np.any(consistent):
                image_supported_errors.append(relative_error[consistent].astype(np.float32))

        required_support = np.where(edge_mask, edge_min_support, args.min_support)
        output_mask = input_mask & (support >= required_support)
        if args.min_component_area > 1 and np.any(output_mask):
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
                output_mask.astype(np.uint8), connectivity=8
            )
            keep = np.zeros(component_count, dtype=bool)
            keep[0] = False
            keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= args.min_component_area
            output_mask &= keep[labels]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            optimized_depth = np.nanmedian(np.stack(candidate_depths, axis=0), axis=0)
        optimized_depth = np.where(output_mask, optimized_depth, 0).astype(np.float32)
        confidence = np.divide(
            support,
            np.maximum(observations, 1),
            dtype=np.float32,
        )
        confidence[~output_mask] = 0

        stem = Path(name).stem
        np.save(args.output / f"{stem}_depth.npy", optimized_depth)
        np.save(args.output / f"{stem}_mask.npy", output_mask)
        np.save(args.output / f"{stem}_confidence.npy", confidence)
        np.save(args.output / f"{stem}_support.npy", support)
        np.save(args.output / f"{stem}_observations.npy", observations)

        input_count = int(input_mask.sum())
        output_count = int(output_mask.sum())
        total_input += input_count
        total_output += output_count
        total_observations += int(observations[input_mask].sum())
        total_supports += int(support[input_mask].sum())
        if image_observed_errors:
            all_observed_errors.append(np.concatenate(image_observed_errors))
        if image_supported_errors:
            all_supported_errors.append(np.concatenate(image_supported_errors))
        records.append(
            {
                "name": name,
                "neighbors": [names[index] for index in neighbors],
                "input_valid_ratio": float(input_mask.mean()),
                "output_valid_ratio": float(output_mask.mean()),
                "retained_valid_fraction": float(output_count / max(input_count, 1)),
                "mean_observations_per_input_pixel": float(
                    observations[input_mask].mean() if input_count else 0
                ),
                "mean_support_per_input_pixel": float(support[input_mask].mean() if input_count else 0),
                "mean_confidence_on_output": float(confidence[output_mask].mean() if output_count else 0),
                "edge_pixel_fraction": float(edge_mask[input_mask].mean() if input_count else 0),
            }
        )
        print(
            f"[{current + 1}/{len(names)}] {name}: "
            f"valid {input_mask.mean():.4f} -> {output_mask.mean():.4f}, "
            f"retained={output_count / max(input_count, 1):.3f}",
            flush=True,
        )

    observed_errors = np.concatenate(all_observed_errors) if all_observed_errors else np.array([])
    supported_errors = np.concatenate(all_supported_errors) if all_supported_errors else np.array([])
    summary = {
        "source_depth": str(args.depth.resolve()),
        "output": str(args.output.resolve()),
        "images": len(names),
        "parameters": {
            "neighbors": args.neighbors,
            "min_support": args.min_support,
            "edge_min_support": edge_min_support,
            "edge_relative_gradient": args.edge_relative_gradient,
            "relative_depth_threshold": args.relative_depth_threshold,
            "reprojection_threshold_pixels": args.reprojection_threshold,
            "normal_angle_threshold_degrees": args.normal_angle_threshold,
            "min_component_area": args.min_component_area,
        },
        "input_valid_pixels": total_input,
        "output_valid_pixels": total_output,
        "retained_valid_fraction": float(total_output / max(total_input, 1)),
        "support_fraction_of_observations": float(total_supports / max(total_observations, 1)),
        "observed_relative_depth_error": {
            "median": float(np.median(observed_errors)) if len(observed_errors) else None,
            "p90": float(np.percentile(observed_errors, 90)) if len(observed_errors) else None,
        },
        "accepted_relative_depth_error": {
            "median": float(np.median(supported_errors)) if len(supported_errors) else None,
            "p90": float(np.percentile(supported_errors, 90)) if len(supported_errors) else None,
        },
        "records": records,
    }
    (args.output / "consistency_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
