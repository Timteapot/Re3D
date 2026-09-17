from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


SUPPORTED = {".png", ".jpg", ".jpeg"}


def natural_key(path: Path) -> tuple[object, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Copy one Re3D scene and create COLMAP masks plus a reproducibility manifest."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-threshold", type=int, default=128)
    args = parser.parse_args()

    paths = sorted(
        (path for path in args.source.iterdir() if path.suffix.lower() in SUPPORTED),
        key=natural_key,
    )
    if len(paths) < 3:
        raise RuntimeError(f"At least three images are required; found {len(paths)}")
    stems = [path.stem for path in paths]
    if len(stems) != len(set(stems)):
        raise RuntimeError("Image stems must be unique across file extensions")

    images_dir = args.output / "images"
    masks_dir = args.output / "masks"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    sizes: set[tuple[int, int]] = set()
    for source in paths:
        target = images_dir / source.name
        shutil.copy2(source, target)
        with Image.open(source) as image:
            sizes.add(image.size)
            if "A" in image.getbands():
                alpha = image.getchannel("A")
            else:
                alpha = Image.new("L", image.size, 255)
            foreground = alpha.point(
                lambda value: 255 if value >= args.alpha_threshold else 0
            )
            # pycolmap mask_path uses <full-image-name>.png.
            foreground.save(masks_dir / f"{source.name}.png")
            histogram = foreground.histogram()
            records.append(
                {
                    "name": source.name,
                    "source": str(source.resolve()),
                    "width": image.width,
                    "height": image.height,
                    "mode": image.mode,
                    "foreground_fraction": histogram[255] / (image.width * image.height),
                    "sha256": sha256(source),
                }
            )
    if len(sizes) != 1:
        raise RuntimeError(
            "The migrated pipeline currently requires a uniform image size because it uses "
            "one COLMAP camera and global source dimensions."
        )
    width, height = next(iter(sizes))
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": str(args.source.resolve()),
        "selection": "all supported images in natural filename order",
        "supported_extensions": sorted(SUPPORTED),
        "count": len(records),
        "uniform_size": [width, height],
        "alpha_threshold": args.alpha_threshold,
        "images": records,
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "input-manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "images"}, indent=2))


if __name__ == "__main__":
    main()
