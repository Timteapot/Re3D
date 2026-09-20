from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an isolated OpenMVS input with RGB images and alpha masks."
    )
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--sparse", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha-threshold", type=int, default=128)
    args = parser.parse_args()

    output_images = args.output / "images"
    output_sparse = args.output / "sparse"
    output_images.mkdir(parents=True, exist_ok=True)
    if output_sparse.exists():
        shutil.rmtree(output_sparse)
    shutil.copytree(args.sparse, output_sparse)

    records = []
    supported = {".png", ".jpg", ".jpeg"}
    image_paths = sorted(path for path in args.images.iterdir() if path.suffix.lower() in supported)
    for path in image_paths:
        rgba = np.asarray(Image.open(path).convert("RGBA"))
        rgb = rgba[:, :, :3].copy()
        alpha = rgba[:, :, 3]
        foreground = alpha >= args.alpha_threshold
        if not np.any(foreground):
            raise ValueError(f"No opaque foreground in {path}")

        # OpenMVS reads RGB images. Extend foreground colors through transparent
        # pixels so bilinear sampling at silhouettes cannot pull in black RGB.
        nearest = distance_transform_edt(
            ~foreground, return_distances=False, return_indices=True
        )
        filled = rgb[nearest[0], nearest[1]]
        rgb[~foreground] = filled[~foreground]

        image_output = output_images / path.name
        mask_output = output_images / f"{path.stem}.mask.png"
        Image.fromarray(rgb, mode="RGB").save(image_output)
        Image.fromarray((foreground.astype(np.uint8) * 255), mode="L").save(mask_output)
        records.append(
            {
                "name": path.name,
                "foreground_fraction": float(foreground.mean()),
                "image": str(image_output.resolve()),
                "mask": str(mask_output.resolve()),
            }
        )

    manifest = {
        "source_images": str(args.images.resolve()),
        "source_sparse": str(args.sparse.resolve()),
        "output": str(args.output.resolve()),
        "count": len(records),
        "alpha_threshold": args.alpha_threshold,
        "image_method": "nearest opaque-pixel color extension",
        "mask_convention": "<image-stem>.mask.png; 0 background, 255 foreground",
        "images": records,
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: value for key, value in manifest.items() if key != "images"},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
