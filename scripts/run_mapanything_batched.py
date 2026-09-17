from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import load_images

from run_mapanything import apply_source_alpha, build_tsdf_mesh, similarity_scale, to_numpy


def nearest_groups(poses: np.ndarray, batch_size: int) -> list[tuple[list[int], list[int]]]:
    centers = poses[:, :3, 3]
    remaining = set(range(len(poses)))
    result: list[tuple[list[int], list[int]]] = []
    while remaining:
        seed = min(remaining)
        distances = np.linalg.norm(centers - centers[seed], axis=1)
        core = sorted(remaining, key=lambda index: distances[index])[:batch_size]
        remaining.difference_update(core)
        context: list[int] = []
        if len(core) < batch_size:
            candidates = [index for index in np.argsort(distances) if int(index) not in core]
            context = [int(index) for index in candidates[: batch_size - len(core)]]
        result.append((core, context))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/map-anything")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument(
        "--skip-tsdf",
        action="store_true",
        help="Only write learned depths; Re3D fuses them with OpenMVS later.",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    depth_dir = args.output / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    camera_data = np.load(args.cameras, allow_pickle=False)
    names = [str(value) for value in camera_data["names"]]
    refined_intrinsics = camera_data["intrinsics"].astype(np.float64)
    refined_poses = camera_data["camera_poses"].astype(np.float64)
    groups = nearest_groups(refined_poses, args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MapAnything.from_pretrained(args.model).to(device).eval()
    records: list[dict] = []
    manifest: list[dict] = []
    for batch_index, (core, context) in enumerate(groups):
        indices = core + context
        paths = [args.images / names[index] for index in indices]
        views = load_images([str(path) for path in paths], verbose=False)
        with torch.inference_mode():
            outputs = model.infer(
                views,
                memory_efficient_inference=True,
                minibatch_size=1,
                use_amp=True,
                amp_dtype="bf16",
                apply_mask=True,
                mask_edges=True,
            )
        predicted_poses = np.stack(
            [to_numpy(prediction["camera_poses"])[0].astype(np.float64) for prediction in outputs]
        )
        scale = similarity_scale(predicted_poses[:, :3, 3], refined_poses[indices, :3, 3])
        for local_index, global_index in enumerate(core):
            prediction = outputs[local_index]
            path = paths[local_index]
            depth = to_numpy(prediction["depth_z"])[0].squeeze(-1).astype(np.float32) * scale
            mask = to_numpy(prediction["mask"])[0].squeeze(-1).astype(bool)
            mask = apply_source_alpha(path, mask)
            image = to_numpy(prediction["img_no_norm"])[0].astype(np.float32)
            h, w = depth.shape
            with Image.open(path) as source:
                source_w, source_h = source.size
            k = refined_intrinsics[global_index].copy()
            k[0] *= w / source_w
            k[1] *= h / source_h
            pose = refined_poses[global_index]
            yy, xx = np.mgrid[:h, :w]
            x = (xx - k[0, 2]) / k[0, 0] * depth
            y = (yy - k[1, 2]) / k[1, 1] * depth
            camera_points = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)
            pts3d = (camera_points @ pose.T)[..., :3].astype(np.float32)
            name = names[global_index]
            stem = Path(name).stem
            np.save(depth_dir / f"{stem}_depth.npy", depth)
            np.save(depth_dir / f"{stem}_mask.npy", mask)
            Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(
                depth_dir / f"{stem}_rgb.png"
            )
            if not args.skip_tsdf:
                records.append(
                    {
                        "name": name,
                        "depth": depth,
                        "intrinsics": k,
                        "camera_pose": pose,
                        "mask": mask,
                        "image": image,
                        "pts3d": pts3d,
                    }
                )
            manifest.append(
                {
                    "name": name,
                    "batch": batch_index,
                    "scale_to_colmap": scale,
                    "valid_depth_ratio": float((mask & np.isfinite(depth) & (depth > 0)).mean()),
                }
            )
        print(
            f"batch {batch_index + 1}/{len(groups)}: core={len(core)} context={len(context)} scale={scale:.6g}",
            flush=True,
        )
        del outputs, views
        torch.cuda.empty_cache()

    manifest.sort(key=lambda record: names.index(record["name"]))
    (args.output / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.skip_tsdf:
        print(
            json.dumps(
                {
                    "images": len(manifest),
                    "batches": len(groups),
                    "batch_size": args.batch_size,
                    "tsdf_skipped": True,
                },
                indent=2,
            )
        )
        return
    records.sort(key=lambda record: names.index(record["name"]))
    mesh_stats = build_tsdf_mesh(records, args.output)
    mesh_stats["images"] = len(records)
    mesh_stats["batches"] = len(groups)
    mesh_stats["batch_size"] = args.batch_size
    (args.output / "dense" / "mesh_generation.json").write_text(
        json.dumps(mesh_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(mesh_stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
