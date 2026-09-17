from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def qvec_to_rotmat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=np.float64,
    )


def parse_cameras(path: Path) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model = int(fields[0]), fields[1]
        params = np.asarray([float(v) for v in fields[4:]], dtype=np.float64)
        if model == "PINHOLE":
            fx, fy, cx, cy = params[:4]
        elif model == "SIMPLE_PINHOLE":
            fx, cx, cy = params[:3]
            fy = fx
        else:
            raise ValueError(f"Unsupported refined camera model: {model}")
        result[camera_id] = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
    return result


def parse_images(path: Path) -> dict[str, tuple[int, np.ndarray]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    result: dict[str, tuple[int, np.ndarray]] = {}
    data_lines = [line.strip() for line in lines if not line.lstrip().startswith("#")]
    for index in range(0, len(data_lines), 2):
        if not data_lines[index]:
            continue
        fields = data_lines[index].split()
        qvec = np.asarray([float(v) for v in fields[1:5]], dtype=np.float64)
        tvec = np.asarray([float(v) for v in fields[5:8]], dtype=np.float64)
        camera_id = int(fields[8])
        name = " ".join(fields[9:])
        world_to_camera = np.eye(4, dtype=np.float64)
        world_to_camera[:3, :3] = qvec_to_rotmat(qvec)
        world_to_camera[:3, 3] = tvec
        result[name] = (camera_id, np.linalg.inv(world_to_camera))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--template", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cameras = parse_cameras(args.model / "cameras.txt")
    images = parse_images(args.model / "images.txt")
    if args.template:
        template = np.load(args.template, allow_pickle=False)
        names = [str(value) for value in template["names"]]
    else:
        names = sorted(images)
    missing = [name for name in names if name not in images]
    if missing:
        raise RuntimeError(f"Refined model is missing registered images: {missing}")
    intrinsic_stack = np.stack([cameras[images[name][0]] for name in names])
    pose_stack = np.stack([images[name][1] for name in names])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, names=np.asarray(names), intrinsics=intrinsic_stack, camera_poses=pose_stack)
    summary = {
        "registered_images": len(names),
        "camera_count": len(set(images[name][0] for name in names)),
        "source_model": str(args.model),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
