from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


DMAP_DEPTH = 1
DMAP_NORMAL = 2
DMAP_CONFIDENCE = 4
DMAP_KNOWN_FLAGS = DMAP_DEPTH | DMAP_NORMAL | DMAP_CONFIDENCE


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_exact(handle, size: int, path: Path) -> bytes:
    data = handle.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated DMAP {path}: expected {size} bytes, got {len(data)}")
    return data


def read_dmap_header(path: Path) -> dict:
    path = Path(path)
    with path.open("rb") as handle:
        if _read_exact(handle, 2, path) != b"DR":
            raise ValueError(f"Invalid DMAP signature: {path}")
        content_type = int(np.frombuffer(_read_exact(handle, 1, path), dtype=np.uint8)[0])
        compression = int(np.frombuffer(_read_exact(handle, 1, path), dtype=np.uint8)[0])
        image_width, image_height = np.frombuffer(
            _read_exact(handle, 8, path), dtype=np.uint32
        )
        depth_width, depth_height = np.frombuffer(
            _read_exact(handle, 8, path), dtype=np.uint32
        )
        depth_min, depth_max = np.frombuffer(
            _read_exact(handle, 8, path), dtype=np.float32
        )
        name_size = int(
            np.frombuffer(_read_exact(handle, 2, path), dtype=np.uint16)[0]
        )
        file_name = _read_exact(handle, name_size, path).decode("utf-8")
        view_count = int(
            np.frombuffer(_read_exact(handle, 4, path), dtype=np.uint32)[0]
        )
        view_ids = np.frombuffer(
            _read_exact(handle, 4 * view_count, path), dtype=np.uint32
        ).copy()
        intrinsics = np.frombuffer(
            _read_exact(handle, 72, path), dtype=np.float64
        ).reshape(3, 3).copy()
        rotation = np.frombuffer(
            _read_exact(handle, 72, path), dtype=np.float64
        ).reshape(3, 3).copy()
        center = np.frombuffer(
            _read_exact(handle, 24, path), dtype=np.float64
        ).copy()
        payload_offset = handle.tell()
    return {
        "content_type": content_type,
        "compression": compression,
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
        "payload_offset": payload_offset,
    }


def expected_dmap_size(header: dict) -> int:
    flags = int(header["content_type"])
    if flags & ~DMAP_KNOWN_FLAGS:
        raise ValueError(f"Unsupported DMAP content flags: {flags}")
    if not flags & DMAP_DEPTH:
        raise ValueError(f"DMAP does not contain depth: flags={flags}")
    pixels = int(header["depth_width"]) * int(header["depth_height"])
    float_count = pixels
    if flags & DMAP_NORMAL:
        float_count += pixels * 3
    if flags & DMAP_CONFIDENCE:
        float_count += pixels
    return int(header["payload_offset"]) + float_count * np.dtype(np.float32).itemsize


def read_dmap(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    path = Path(path)
    header = read_dmap_header(path)
    if int(header["compression"]) != 0:
        raise ValueError(f"Compressed DMAP is not supported by the adapter: {path}")
    expected_size = expected_dmap_size(header)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise ValueError(
            f"DMAP size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    height = int(header["depth_height"])
    width = int(header["depth_width"])
    pixels = height * width
    payload: dict[str, np.ndarray] = {}
    with path.open("rb") as handle:
        handle.seek(int(header["payload_offset"]))
        payload["depth"] = np.frombuffer(
            _read_exact(handle, pixels * 4, path), dtype=np.float32
        ).reshape(height, width).copy()
        if int(header["content_type"]) & DMAP_NORMAL:
            payload["normal"] = np.frombuffer(
                _read_exact(handle, pixels * 3 * 4, path), dtype=np.float32
            ).reshape(height, width, 3).copy()
        if int(header["content_type"]) & DMAP_CONFIDENCE:
            payload["confidence"] = np.frombuffer(
                _read_exact(handle, pixels * 4, path), dtype=np.float32
            ).reshape(height, width).copy()
    return header, payload


def _normalized_payload(
    depth: np.ndarray,
    confidence: np.ndarray | None,
    normal: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Depth must be HxW, got {depth.shape}")
    valid = np.isfinite(depth) & (depth > 0)
    depth = np.where(valid, depth, 0).astype(np.float32)
    if confidence is not None:
        confidence = np.asarray(confidence, dtype=np.float32)
        if confidence.shape != depth.shape:
            raise ValueError(
                f"Confidence/depth shape mismatch: {confidence.shape} != {depth.shape}"
            )
        confidence = np.where(valid & np.isfinite(confidence), confidence, 0)
        confidence = np.clip(confidence, 0, 1).astype(np.float32)
    if normal is not None:
        normal = np.asarray(normal, dtype=np.float32)
        if normal.shape != (*depth.shape, 3):
            raise ValueError(
                f"Normal/depth shape mismatch: {normal.shape} != {(*depth.shape, 3)}"
            )
        normal = np.where(valid[..., None] & np.isfinite(normal), normal, 0)
        lengths = np.linalg.norm(normal, axis=2, keepdims=True)
        usable = valid[..., None] & (lengths > 1e-8)
        normalized = np.zeros_like(normal, dtype=np.float32)
        np.divide(
            normal,
            np.maximum(lengths, 1e-8),
            out=normalized,
            where=usable,
        )
        normal = normalized
    return depth, confidence, normal, valid


def write_dmap(
    path: Path,
    header: dict,
    depth: np.ndarray,
    confidence: np.ndarray | None = None,
    normal: np.ndarray | None = None,
) -> None:
    path = Path(path)
    depth, confidence, normal, valid = _normalized_payload(
        depth, confidence, normal
    )
    height, width = depth.shape
    image_size = (int(header["image_width"]), int(header["image_height"]))
    if image_size != (width, height):
        raise ValueError(
            "OpenMVS external DMAP requires image and depth resolutions to match: "
            f"header image={image_size}, payload depth={(width, height)}"
        )
    flags = DMAP_DEPTH
    if normal is not None:
        flags |= DMAP_NORMAL
    if confidence is not None:
        flags |= DMAP_CONFIDENCE
    file_name = str(header["file_name"])
    encoded_name = file_name.encode("utf-8")
    view_ids = np.asarray(header["view_ids"], dtype=np.uint32)
    positive = depth[valid]
    limits = (
        np.asarray([positive.min(), positive.max()], dtype=np.float32)
        if positive.size
        else np.zeros(2, dtype=np.float32)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(b"DR")
            handle.write(np.asarray([flags, 0], dtype=np.uint8).tobytes())
            handle.write(np.asarray([width, height], dtype=np.uint32).tobytes())
            handle.write(np.asarray([width, height], dtype=np.uint32).tobytes())
            handle.write(limits.tobytes())
            handle.write(np.asarray([len(encoded_name)], dtype=np.uint16).tobytes())
            handle.write(encoded_name)
            handle.write(np.asarray([len(view_ids)], dtype=np.uint32).tobytes())
            handle.write(view_ids.tobytes())
            handle.write(np.asarray(header["K"], dtype=np.float64).reshape(3, 3).tobytes())
            handle.write(np.asarray(header["R"], dtype=np.float64).reshape(3, 3).tobytes())
            handle.write(np.asarray(header["C"], dtype=np.float64).reshape(3).tobytes())
            handle.write(depth.tobytes())
            if normal is not None:
                handle.write(normal.tobytes())
            if confidence is not None:
                handle.write(confidence.tobytes())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_header(header: dict) -> dict:
    return {
        "content_type": int(header["content_type"]),
        "compression": int(header["compression"]),
        "image_size": [int(header["image_height"]), int(header["image_width"])],
        "depth_size": [int(header["depth_height"]), int(header["depth_width"])],
        "depth_range": [float(header["depth_min"]), float(header["depth_max"])],
        "file_name": str(header["file_name"]),
        "view_ids": [int(value) for value in np.asarray(header["view_ids"]).tolist()],
        "K": np.asarray(header["K"], dtype=np.float64).tolist(),
        "R": np.asarray(header["R"], dtype=np.float64).tolist(),
        "C": np.asarray(header["C"], dtype=np.float64).tolist(),
    }
