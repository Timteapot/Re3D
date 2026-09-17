from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
import torch
import trimesh
from PIL import Image

from mapanything.models import MapAnything
from mapanything.utils.image import load_images


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def apply_source_alpha(path: Path, mask: np.ndarray) -> np.ndarray:
    with Image.open(path) as source:
        if "A" not in source.getbands():
            return mask
        alpha = source.getchannel("A").resize((mask.shape[1], mask.shape[0]), Image.Resampling.NEAREST)
    return mask & (np.asarray(alpha) > 127)


def rotation_matrix_to_qvec(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to COLMAP's (qw, qx, qy, qz)."""
    r = np.asarray(rotation, dtype=np.float64)
    k = np.array(
        [
            [r[0, 0] - r[1, 1] - r[2, 2], r[1, 0] + r[0, 1], r[2, 0] + r[0, 2], r[1, 2] - r[2, 1]],
            [r[1, 0] + r[0, 1], r[1, 1] - r[0, 0] - r[2, 2], r[2, 1] + r[1, 2], r[2, 0] - r[0, 2]],
            [r[2, 0] + r[0, 2], r[2, 1] + r[1, 2], r[2, 2] - r[0, 0] - r[1, 1], r[0, 1] - r[1, 0]],
            [r[1, 2] - r[2, 1], r[2, 0] - r[0, 2], r[0, 1] - r[1, 0], r[0, 0] + r[1, 1] + r[2, 2]],
        ]
    ) / 3.0
    eigenvalues, eigenvectors = np.linalg.eigh(k)
    q = eigenvectors[[3, 0, 1, 2], np.argmax(eigenvalues)]
    if q[0] < 0:
        q = -q
    return q


def write_colmap_text(records: list[dict], output_dir: Path) -> dict:
    """Write a COLMAP text model without depending on pycolmap's evolving API."""
    root = output_dir / "mapanything_colmap"
    sparse_dir = root / "sparse"
    images_dir = root / "images"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    camera_lines = ["# Camera list with one line of data per camera:", "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]"]
    image_lines = [
        "# Image list with two lines of data per image:",
        "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME",
        "# POINTS2D[] as (X, Y, POINT3D_ID)",
    ]
    point_samples = []
    point_colors = []
    for image_id, record in enumerate(records, start=1):
        h, w = record["depth"].shape
        k = record["intrinsics"]
        camera_lines.append(
            f"{image_id} PINHOLE {w} {h} {k[0, 0]:.17g} {k[1, 1]:.17g} {k[0, 2]:.17g} {k[1, 2]:.17g}"
        )
        world_to_camera = np.linalg.inv(record["camera_pose"])
        qvec = rotation_matrix_to_qvec(world_to_camera[:3, :3])
        tvec = world_to_camera[:3, 3]
        image_lines.append(
            f"{image_id} {' '.join(f'{x:.17g}' for x in qvec)} "
            f"{' '.join(f'{x:.17g}' for x in tvec)} {image_id} {record['name']}"
        )
        image_lines.append("")
        Image.fromarray(np.clip(record["image"] * 255.0, 0, 255).astype(np.uint8)).save(images_dir / record["name"])

        mask = record["mask"] & np.isfinite(record["depth"]) & (record["depth"] > 0)
        pts = record["pts3d"][mask]
        rgb = np.clip(record["image"][mask] * 255.0, 0, 255).astype(np.uint8)
        if len(pts):
            stride = max(1, len(pts) // 50000)
            point_samples.append(pts[::stride])
            point_colors.append(rgb[::stride])

    (sparse_dir / "cameras.txt").write_text("\n".join(camera_lines) + "\n", encoding="utf-8")
    (sparse_dir / "images.txt").write_text("\n".join(image_lines) + "\n", encoding="utf-8")
    (sparse_dir / "points3D.txt").write_text("# Empty: points are triangulated after feature matching.\n", encoding="utf-8")
    if point_samples:
        trimesh.PointCloud(np.concatenate(point_samples), colors=np.concatenate(point_colors)).export(sparse_dir / "points.ply")
    return {
        "format": "COLMAP text",
        "images": len(records),
        "sparse_dir": str(sparse_dir),
        "images_dir": str(images_dir),
    }


def similarity_scale(source: np.ndarray, target: np.ndarray) -> float:
    source_centered = source - source.mean(axis=0)
    target_centered = target - target.mean(axis=0)
    covariance = target_centered.T @ source_centered / len(source)
    u, singular, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1
    variance = float(np.sum(source_centered**2) / len(source))
    return float(np.sum(singular * np.diag(correction)) / variance)


def load_existing_records(
    branch_output: Path,
    shared_output: Path,
    input_images: Path,
    refined_cameras: Path | None = None,
) -> tuple[list[dict], dict]:
    camera_file = shared_output / "cameras_mapanything.npz"
    data = np.load(camera_file, allow_pickle=False)
    refined_by_name = None
    depth_scale = 1.0
    if refined_cameras is not None:
        refined = np.load(refined_cameras, allow_pickle=False)
        refined_by_name = {
            str(name): (k, pose)
            for name, k, pose in zip(refined["names"], refined["intrinsics"], refined["camera_poses"])
        }
        source_centers = np.stack([pose[:3, 3] for pose in data["camera_poses"]])
        target_centers = np.stack([refined_by_name[str(name)][1][:3, 3] for name in data["names"]])
        depth_scale = similarity_scale(source_centers, target_centers)
    records = []
    for name, intrinsics, camera_pose in zip(data["names"], data["intrinsics"], data["camera_poses"]):
        name = str(name)
        stem = Path(name).stem
        depth = np.load(branch_output / "depth" / f"{stem}_depth.npy").astype(np.float32) * depth_scale
        mask = np.load(branch_output / "depth" / f"{stem}_mask.npy").astype(bool)
        rgb_path = branch_output / "depth" / f"{stem}_rgb.png"
        image = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        h, w = depth.shape
        if refined_by_name is not None:
            refined_k, refined_pose = refined_by_name[name]
            with Image.open(input_images / name) as source_image:
                source_w, source_h = source_image.size
            intrinsics = refined_k.copy()
            intrinsics[0] *= w / source_w
            intrinsics[1] *= h / source_h
            camera_pose = refined_pose
        yy, xx = np.mgrid[:h, :w]
        x = (xx - intrinsics[0, 2]) / intrinsics[0, 0] * depth
        y = (yy - intrinsics[1, 2]) / intrinsics[1, 1] * depth
        camera_points = np.stack([x, y, depth, np.ones_like(depth)], axis=-1)
        pts3d = (camera_points @ camera_pose.T)[..., :3].astype(np.float32)
        records.append(
            {
                "name": name,
                "depth": depth,
                "intrinsics": intrinsics.astype(np.float64),
                "camera_pose": camera_pose.astype(np.float64),
                "mask": mask,
                "image": image,
                "pts3d": pts3d,
            }
        )
    return records, {
        "refined_cameras": str(refined_cameras) if refined_cameras else None,
        "depth_scale_to_refined_camera_baseline": depth_scale,
    }


def build_tsdf_mesh(records: list[dict], output_dir: Path) -> dict:
    point_samples = []
    valid_depths = []
    for record in records:
        mask = record["mask"] & np.isfinite(record["depth"]) & (record["depth"] > 0)
        valid_depths.append(record["depth"][mask])
        pts = record["pts3d"][mask]
        if len(pts):
            point_samples.append(pts[:: max(1, len(pts) // 50000)])
    all_depths = np.concatenate(valid_depths)
    sampled_points = np.concatenate(point_samples)
    q05, q95 = np.percentile(sampled_points, [5, 95], axis=0)
    scene_extent = float(np.max(q95 - q05))
    voxel_length = float(np.clip(scene_extent / 256.0, 0.001, 0.5))
    depth_trunc = float(np.percentile(all_depths, 99.0))
    sdf_trunc = voxel_length * 5.0
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for record in records:
        depth = record["depth"].astype(np.float32).copy()
        depth[~record["mask"]] = 0.0
        image = np.ascontiguousarray(np.clip(record["image"] * 255.0, 0, 255).astype(np.uint8))
        depth = np.ascontiguousarray(depth)
        h, w = depth.shape
        k = record["intrinsics"]
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            w, h, float(k[0, 0]), float(k[1, 1]), float(k[0, 2]), float(k[1, 2])
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(image),
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
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    tri = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=colors, process=False)
    glb_path = dense_dir / "mesh_raw.glb"
    tri.export(glb_path)
    return {
        "voxel_length": voxel_length,
        "sdf_trunc": sdf_trunc,
        "depth_trunc": depth_trunc,
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "ply": str(ply_path),
        "glb": str(glb_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--shared-output", type=Path, required=True)
    parser.add_argument("--branch-output", type=Path, required=True)
    parser.add_argument("--model", default="facebook/map-anything")
    parser.add_argument("--postprocess-existing", action="store_true")
    parser.add_argument("--refined-cameras", type=Path)
    args = parser.parse_args()

    args.shared_output.mkdir(parents=True, exist_ok=True)
    args.branch_output.mkdir(parents=True, exist_ok=True)
    raw_dir = args.branch_output / "depth"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if args.postprocess_existing:
        records, alignment = load_existing_records(
            args.branch_output,
            args.shared_output,
            args.images,
            args.refined_cameras,
        )
        (args.branch_output / "camera_alignment.json").write_text(
            json.dumps(alignment, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        colmap_stats = write_colmap_text(records, args.shared_output)
        (args.shared_output / "mapanything_colmap" / "export.json").write_text(
            json.dumps(colmap_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        mesh_stats = build_tsdf_mesh(records, args.branch_output)
        (args.branch_output / "dense" / "mesh_generation.json").write_text(
            json.dumps(mesh_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(json.dumps(mesh_stats, indent=2))
        return

    image_paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    print(f"Loading {len(image_paths)} images")
    views = load_images([str(p) for p in image_paths], verbose=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}")
    model = MapAnything.from_pretrained(args.model).to(device).eval()
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

    records = []
    key_shapes = []
    for index, (path, pred) in enumerate(zip(image_paths, outputs)):
        depth = to_numpy(pred["depth_z"])[0].squeeze(-1).astype(np.float32)
        intrinsics = to_numpy(pred["intrinsics"])[0].astype(np.float64)
        camera_pose = to_numpy(pred["camera_poses"])[0].astype(np.float64)
        mask = to_numpy(pred["mask"])[0].squeeze(-1).astype(bool)
        mask = apply_source_alpha(path, mask)
        image = to_numpy(pred["img_no_norm"])[0].astype(np.float32)
        pts3d = to_numpy(pred["pts3d"])[0].astype(np.float32)
        np.save(raw_dir / f"{path.stem}_depth.npy", depth)
        np.save(raw_dir / f"{path.stem}_mask.npy", mask)
        Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(raw_dir / f"{path.stem}_rgb.png")
        records.append(
            {
                "name": path.name,
                "depth": depth,
                "intrinsics": intrinsics,
                "camera_pose": camera_pose,
                "mask": mask,
                "image": image,
                "pts3d": pts3d,
            }
        )
        key_shapes.append(
            {
                "name": path.name,
                "keys": {k: list(v.shape) if hasattr(v, "shape") else str(type(v)) for k, v in pred.items()},
                "valid_depth_ratio": float((mask & np.isfinite(depth) & (depth > 0)).mean()),
            }
        )

    np.savez_compressed(
        args.shared_output / "cameras_mapanything.npz",
        names=np.array([r["name"] for r in records]),
        intrinsics=np.stack([r["intrinsics"] for r in records]),
        camera_poses=np.stack([r["camera_pose"] for r in records]),
    )
    (args.branch_output / "prediction_manifest.json").write_text(
        json.dumps(key_shapes, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    colmap_stats = write_colmap_text(records, args.shared_output)
    (args.shared_output / "mapanything_colmap" / "export.json").write_text(
        json.dumps(colmap_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    mesh_stats = build_tsdf_mesh(records, args.branch_output)
    (args.branch_output / "dense" / "mesh_generation.json").write_text(
        json.dumps(mesh_stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(mesh_stats, indent=2))


if __name__ == "__main__":
    main()
