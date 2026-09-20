from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import trimesh
from PIL import Image, ImageDraw


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry in {path}")
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def render(
    mesh_path: Path, intrinsics: np.ndarray, camera_pose: np.ndarray, size: tuple[int, int]
) -> tuple[Image.Image, dict[str, float | int | str]]:
    mesh = load_mesh(mesh_path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    uv = getattr(mesh.visual, "uv", None)
    material = getattr(mesh.visual, "material", None)
    texture = getattr(material, "image", None)
    if uv is not None and texture is not None:
        texture_array = np.asarray(texture.convert("RGB"))
        face_uv = np.asarray(uv, dtype=np.float64)[faces].mean(axis=1)
        x = np.clip(np.rint(face_uv[:, 0] * (texture_array.shape[1] - 1)), 0, texture_array.shape[1] - 1).astype(int)
        y_flipped = np.clip(np.rint((1.0 - face_uv[:, 1]) * (texture_array.shape[0] - 1)), 0, texture_array.shape[0] - 1).astype(int)
        # OBJ texture coordinates use a bottom-left origin, while image arrays
        # use a top-left origin. OpenMVS exports standard OBJ coordinates too.
        colors = texture_array[y_flipped, x]
        uv_convention = "flipped_v_obj"
    else:
        vertex_colors = np.asarray(mesh.visual.to_color().vertex_colors[:, :3], dtype=np.float32)
        colors = np.rint(vertex_colors[faces].mean(axis=1)).astype(np.uint8)
        uv_convention = "vertex_color_fallback"

    homogeneous = np.column_stack([vertices, np.ones(len(vertices), dtype=np.float64)])
    camera_vertices = homogeneous @ np.linalg.inv(camera_pose).T
    z = camera_vertices[:, 2]
    projected_h = camera_vertices[:, :3] @ intrinsics.T
    projected = projected_h[:, :2] / np.maximum(projected_h[:, 2:3], 1e-9)

    height, width = size
    image = np.full((height, width, 3), 255, dtype=np.uint8)
    triangle_z = z[faces]
    valid = np.all(triangle_z > 1e-6, axis=1)
    triangle_uv = projected[faces]
    bbox_min = triangle_uv.min(axis=1)
    bbox_max = triangle_uv.max(axis=1)
    valid &= (bbox_max[:, 0] >= 0) & (bbox_max[:, 1] >= 0)
    valid &= (bbox_min[:, 0] < width) & (bbox_min[:, 1] < height)
    valid_indices = np.flatnonzero(valid)
    # Painter order is sufficient for a deterministic visual sanity check.
    order = valid_indices[np.argsort(triangle_z[valid_indices].mean(axis=1))[::-1]]
    for face_index in order:
        polygon = np.rint(triangle_uv[face_index]).astype(np.int32)
        polygon[:, 0] = np.clip(polygon[:, 0], -2 * width, 3 * width)
        polygon[:, 1] = np.clip(polygon[:, 1], -2 * height, 3 * height)
        cv2.fillConvexPoly(image, polygon, colors[face_index].tolist(), lineType=cv2.LINE_AA)
    luminance = colors.mean(axis=1)
    stats: dict[str, float | int | str] = {
        "mesh": str(mesh_path.resolve()),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "projectable_face_ratio": float(valid.mean()),
        "near_black_face_color_ratio": float((luminance < 10).mean()),
        "median_face_color_luminance": float(np.median(luminance)),
        "uv_convention": uv_convention,
    }
    return Image.fromarray(image), stats


def label(image: Image.Image, text: str) -> Image.Image:
    canvas = Image.new("RGB", (image.width, image.height + 42), "white")
    canvas.paste(image.convert("RGB"), (0, 42))
    ImageDraw.Draw(canvas).text((14, 12), text, fill="black")
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--camera-name", default="r_0.png")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Preview scale in (0, 1]; intrinsics are scaled consistently.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("meshes", nargs="+", help="label=path pairs")
    args = parser.parse_args()

    data = np.load(args.cameras, allow_pickle=False)
    names = [str(value) for value in data["names"]]
    camera_index = names.index(args.camera_name)
    source = Image.open(args.images / args.camera_name).convert("RGBA")
    if not 0 < args.scale <= 1:
        raise ValueError("--scale must be in (0, 1]")
    original_size = source.size
    if args.scale != 1:
        source = source.resize(
            (
                max(1, int(round(source.width * args.scale))),
                max(1, int(round(source.height * args.scale))),
            ),
            Image.Resampling.LANCZOS,
        )
    intrinsics = np.asarray(data["intrinsics"][camera_index], dtype=np.float64).copy()
    intrinsics[0] *= source.width / original_size[0]
    intrinsics[1] *= source.height / original_size[1]
    background = Image.new("RGBA", source.size, "white")
    background.alpha_composite(source)
    panels = [label(background.convert("RGB"), f"input: {args.camera_name}")]
    records: dict[str, dict[str, float | int | str]] = {}

    for item in args.meshes:
        label_text, raw_path = item.split("=", 1)
        rendered, stats = render(
            Path(raw_path),
            intrinsics,
            data["camera_poses"][camera_index],
            (source.height, source.width),
        )
        panels.append(label(rendered, label_text))
        records[label_text] = stats

    sheet = Image.new("RGB", (sum(panel.width for panel in panels), max(panel.height for panel in panels)), "white")
    x = 0
    for panel in panels:
        sheet.paste(panel, (x, 0))
        x += panel.width
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(args.output)
    metrics_path = args.output.with_suffix(".json")
    metrics_path.write_text(
        json.dumps(
            {
                "camera_name": args.camera_name,
                "preview": str(args.output.resolve()),
                "purpose": "software-rendered visual sanity check; not a geometric accuracy metric",
                "branches": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(args.output.resolve())


if __name__ == "__main__":
    main()
