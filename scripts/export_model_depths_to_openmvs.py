from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np


def read_dmap_header(path: Path) -> dict:
    with path.open("rb") as handle:
        if handle.read(2) != b"DR":
            raise ValueError(f"Invalid DMAP signature: {path}")
        content_type = int(np.frombuffer(handle.read(1), dtype=np.uint8)[0])
        handle.read(1)
        image_width, image_height = np.frombuffer(handle.read(8), dtype=np.uint32)
        depth_width, depth_height = np.frombuffer(handle.read(8), dtype=np.uint32)
        depth_min, depth_max = np.frombuffer(handle.read(8), dtype=np.float32)
        name_size = int(np.frombuffer(handle.read(2), dtype=np.uint16)[0])
        file_name = handle.read(name_size).decode("utf-8")
        view_count = int(np.frombuffer(handle.read(4), dtype=np.uint32)[0])
        view_ids = np.frombuffer(handle.read(4 * view_count), dtype=np.uint32).copy()
        intrinsics = np.frombuffer(handle.read(72), dtype=np.float64).reshape(3, 3).copy()
        rotation = np.frombuffer(handle.read(72), dtype=np.float64).reshape(3, 3).copy()
        center = np.frombuffer(handle.read(24), dtype=np.float64).copy()
    return {
        "content_type": content_type,
        "image_width": int(image_width),
        "image_height": int(image_height),
        "depth_width": int(depth_width),
        "depth_height": int(depth_height),
        "depth_min": float(depth_min),
        "depth_max": float(depth_max),
        "file_name": file_name,
        "view_ids": view_ids,
        "K": intrinsics,
        "R": rotation,
        "C": center,
    }


def write_dmap(path: Path, header: dict, depth: np.ndarray) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0).astype(np.float32)
    file_name = str(header["file_name"])
    view_ids = np.asarray(header["view_ids"], dtype=np.uint32)
    with path.open("wb") as handle:
        handle.write(b"DR")
        handle.write(np.array([1, 0], dtype=np.uint8).tobytes())
        handle.write(np.array([header["image_width"], header["image_height"]], dtype=np.uint32).tobytes())
        handle.write(np.array([depth.shape[1], depth.shape[0]], dtype=np.uint32).tobytes())
        positive = depth[valid]
        limits = [float(positive.min()), float(positive.max())] if len(positive) else [0.0, 0.0]
        handle.write(np.asarray(limits, dtype=np.float32).tobytes())
        encoded_name = file_name.encode("utf-8")
        handle.write(np.array([len(encoded_name)], dtype=np.uint16).tobytes())
        handle.write(encoded_name)
        handle.write(np.array([len(view_ids)], dtype=np.uint32).tobytes())
        handle.write(view_ids.tobytes())
        handle.write(np.asarray(header["K"], dtype=np.float64).tobytes())
        handle.write(np.asarray(header["R"], dtype=np.float64).tobytes())
        handle.write(np.asarray(header["C"], dtype=np.float64).tobytes())
        handle.write(depth.tobytes())


def resize_masked_depth(depth: np.ndarray, mask: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    depth = np.nan_to_num(depth.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    valid = mask.astype(bool) & (depth > 0)
    weights = cv2.resize(valid.astype(np.float32), (width, height), interpolation=cv2.INTER_LINEAR)
    numerator = cv2.resize(np.where(valid, depth, 0), (width, height), interpolation=cv2.INTER_LINEAR)
    resized = numerator / np.maximum(weights, 1e-6)
    resized_valid = weights >= 0.5
    resized[~resized_valid] = 0
    return resized.astype(np.float32), resized_valid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-dmaps", type=Path, required=True)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.scene, args.output / "scene.mvs")

    templates = sorted(args.template_dmaps.glob("depth*.dmap"))
    if not templates:
        raise FileNotFoundError(f"No template DMAPs in {args.template_dmaps}")
    records = []
    for template in templates:
        header = read_dmap_header(template)
        image_name = Path(header["file_name"]).name
        stem = Path(image_name).stem
        depth_path = args.depth / f"{stem}_depth.npy"
        mask_path = args.depth / f"{stem}_mask.npy"
        if not depth_path.exists() or not mask_path.exists():
            raise FileNotFoundError(f"Missing model depth/mask for {image_name}")
        depth = np.load(depth_path)
        mask = np.load(mask_path)
        resized, valid = resize_masked_depth(
            depth,
            mask,
            int(header["image_width"]),
            int(header["image_height"]),
        )
        output_path = args.output / template.name
        write_dmap(output_path, header, resized)
        records.append(
            {
                "dmap": output_path.name,
                "image": image_name,
                "source_depth": str(depth_path.resolve()),
                "source_shape": list(depth.shape),
                "output_shape": list(resized.shape),
                "valid_ratio": float(valid.mean()),
                "depth_min": float(resized[valid].min()) if np.any(valid) else 0.0,
                "depth_max": float(resized[valid].max()) if np.any(valid) else 0.0,
            }
        )

    manifest = {
        "scene": str((args.output / "scene.mvs").resolve()),
        "template_dmaps": str(args.template_dmaps.resolve()),
        "source_depth": str(args.depth.resolve()),
        "count": len(records),
        "mean_valid_ratio": float(np.mean([record["valid_ratio"] for record in records])),
        "method": "replace OpenMVS DMAP depth payload with masked model depth while preserving camera metadata",
        "records": records,
    }
    (args.output / "dmap-export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
