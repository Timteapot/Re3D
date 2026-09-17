from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image


def select_source_views(camera_poses: np.ndarray, current: int, count: int) -> list[int]:
    centers = camera_poses[:, :3, 3]
    distances = np.linalg.norm(centers - centers[current], axis=1)
    order = np.argsort(distances)
    return [int(i) for i in order if int(i) != current][:count]


def scaled_intrinsics(k: np.ndarray, old_hw: tuple[int, int], new_hw: tuple[int, int]) -> np.ndarray:
    old_h, old_w = old_hw
    new_h, new_w = new_hw
    k4 = np.eye(4, dtype=np.float32)
    k4[:3, :3] = k.astype(np.float32)
    k4[0] *= new_w / old_w
    k4[1] *= new_h / old_h
    return k4


def fuse_depths(records: list[dict], output_dir: Path) -> dict:
    valid_values = [r["depth"][r["mask"]] for r in records if np.any(r["mask"])]
    if not valid_values:
        raise RuntimeError("MVSAnywhere returned no valid depth values")
    depths = np.concatenate(valid_values)
    depth_trunc = float(np.percentile(depths, 99.0))

    median_depth = float(np.median(depths))
    point_samples = []
    for record in records:
        valid = np.flatnonzero(record["mask"])
        if not len(valid):
            continue
        valid = valid[:: max(1, len(valid) // 10000)]
        h, w = record["depth"].shape
        yy, xx = np.unravel_index(valid, (h, w))
        z = record["depth"][yy, xx]
        k = record["intrinsics"]
        x = (xx - k[0, 2]) / k[0, 0] * z
        y = (yy - k[1, 2]) / k[1, 1] * z
        camera_points = np.column_stack([x, y, z, np.ones_like(z)])
        point_samples.append((camera_points @ record["camera_pose"].T)[:, :3])
    sampled = np.concatenate(point_samples)
    q05, q95 = np.percentile(sampled, [5, 95], axis=0)
    scene_extent = float(np.max(q95 - q05))
    voxel_length = float(np.clip(scene_extent / 256.0, 0.001, 0.5))
    sdf_trunc = voxel_length * 5.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    for record in records:
        depth = np.ascontiguousarray(record["depth"].astype(np.float32).copy())
        depth[~record["mask"]] = 0.0
        color = np.ascontiguousarray(np.asarray(record["image"], dtype=np.uint8))
        h, w = depth.shape
        k = record["intrinsics"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(color),
            o3d.geometry.Image(depth),
            depth_scale=1.0,
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, np.linalg.inv(record["camera_pose"]))

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    dense_dir = output_dir / "dense"
    dense_dir.mkdir(parents=True, exist_ok=True)
    ply_path = dense_dir / "mesh_raw.ply"
    o3d.io.write_triangle_mesh(str(ply_path), mesh, write_ascii=False, compressed=False)
    tri = trimesh.Trimesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles),
        vertex_colors=np.asarray(mesh.vertex_colors),
        process=False,
    )
    glb_path = dense_dir / "mesh_raw.glb"
    tri.export(glb_path)
    return {
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "median_depth": median_depth,
        "scene_extent_q05_q95": scene_extent,
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.triangles)),
        "ply": str(ply_path),
        "glb": str(glb_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-views", type=int, default=7)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=288)
    parser.add_argument("--refinement-steps", type=int, default=1)
    parser.add_argument(
        "--skip-tsdf",
        action="store_true",
        help="Only write learned depths; Re3D fuses them with OpenMVS later.",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(args.repo))
    sys.path.insert(0, str(args.repo / "src"))
    from hubconf import MVSAnywhereInference

    args.output.mkdir(parents=True, exist_ok=True)
    depth_dir = args.output / "depth"
    depth_dir.mkdir(parents=True, exist_ok=True)
    camera_data = np.load(args.cameras, allow_pickle=False)
    names = [str(v) for v in camera_data["names"]]
    intrinsics = camera_data["intrinsics"].astype(np.float64)
    camera_poses = camera_data["camera_poses"].astype(np.float64)
    source_images = [Image.open(args.images / name) for name in names]
    images = [image.convert("RGB") for image in source_images]
    alpha_masks = [
        np.asarray(image.getchannel("A")) > 127 if "A" in image.getbands() else None
        for image in source_images
    ]
    target_hw = (args.height, args.width)
    base_k4 = np.stack(
        [scaled_intrinsics(k, (image.height, image.width), (image.height, image.width)) for k, image in zip(intrinsics, images)]
    )
    adjusted_k4 = np.stack(
        [scaled_intrinsics(k, (image.height, image.width), target_hw) for k, image in zip(intrinsics, images)]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MVSAnywhereInference(
        pretrained=False,
        dot_model=False,
        checkpoint_path=args.checkpoint,
    ).to(device).eval()

    records: list[dict] = []
    manifest: list[dict] = []
    for current, name in enumerate(names):
        sources = select_source_views(camera_poses, current, min(args.source_views, len(names) - 1))
        cur_k = base_k4[current].copy()
        src_k = base_k4[sources].copy()
        with torch.inference_mode():
            output = model.preprocess_and_run(
                cur_image=images[current],
                src_image=[images[i] for i in sources],
                cur_pose=camera_poses[current],
                src_pose=camera_poses[sources],
                cur_intrinsics=cur_k,
                src_intrinsics=src_k,
                depth_range=None,
                num_refinement_steps=args.refinement_steps,
                target_image_size=target_hw,
                device=device,
            )
        depth = output["depth_masked"].detach().float().cpu().numpy().astype(np.float32)
        if depth.ndim != 2:
            depth = np.squeeze(depth)
        mask = np.isfinite(depth) & (depth > 0)
        if alpha_masks[current] is not None:
            alpha = Image.fromarray(alpha_masks[current].astype(np.uint8) * 255).resize(
                (args.width, args.height), Image.Resampling.NEAREST
            )
            mask &= np.asarray(alpha) > 127
        color = np.asarray(images[current].resize((args.width, args.height), Image.Resampling.LANCZOS))
        np.save(depth_dir / f"{Path(name).stem}_depth.npy", depth)
        np.save(depth_dir / f"{Path(name).stem}_mask.npy", mask)
        Image.fromarray(color).save(depth_dir / name)
        k3 = adjusted_k4[current, :3, :3].copy()
        if not args.skip_tsdf:
            records.append(
                {
                    "name": name,
                    "depth": depth,
                    "mask": mask,
                    "image": color,
                    "intrinsics": k3,
                    "camera_pose": camera_poses[current],
                }
            )
        manifest.append(
            {
                "name": name,
                "source_views": [names[i] for i in sources],
                "valid_depth_ratio": float(mask.mean()),
                "depth_median": float(np.median(depth[mask])) if np.any(mask) else None,
            }
        )
        print(f"[{current + 1}/{len(names)}] {name}: valid={mask.mean():.4f}", flush=True)
        torch.cuda.empty_cache()

    (args.output / "prediction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    if args.skip_tsdf:
        print(json.dumps({"images": len(manifest), "tsdf_skipped": True}, indent=2), flush=True)
        return
    stats = fuse_depths(records, args.output)
    (args.output / "dense" / "mesh_generation.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
