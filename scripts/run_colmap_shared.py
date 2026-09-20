from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import pycolmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--camera-model", default="PINHOLE")
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="CPU worker limit for SIFT extraction and matching; avoids excessive memory use on high-core hosts",
    )
    parser.add_argument(
        "--max-num-features",
        type=int,
        default=6000,
        help="Maximum SIFT keypoints per image; matching cost grows approximately quadratically",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    database = args.output / "database.db"
    candidates = args.output / "mapper_candidates"
    selected_binary = args.output / "mapper" / "0"
    selected_text = args.output / "refined_text"
    for path in (candidates, selected_binary, selected_text):
        path.mkdir(parents=True, exist_ok=True)

    reader = pycolmap.ImageReaderOptions()
    if args.masks:
        reader.mask_path = str(args.masks.resolve())
    extraction = pycolmap.FeatureExtractionOptions()
    extraction.max_image_size = 1600
    extraction.num_threads = args.num_threads
    sift_extraction = pycolmap.SiftExtractionOptions()
    sift_extraction.max_num_features = args.max_num_features
    extraction.sift = sift_extraction
    pycolmap.extract_features(
        database_path=str(database),
        image_path=str(args.images),
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=args.camera_model,
        reader_options=reader,
        extraction_options=extraction,
        device=pycolmap.Device.cpu,
    )

    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads = args.num_threads
    matching.guided_matching = True
    pycolmap.match_exhaustive(
        database_path=str(database),
        matching_options=matching,
        device=pycolmap.Device.cpu,
    )

    options = pycolmap.IncrementalPipelineOptions()
    options.multiple_models = True
    options.min_model_size = 3
    options.max_num_models = 10
    options.ba_refine_principal_point = False
    models = pycolmap.incremental_mapping(
        database_path=str(database),
        image_path=str(args.images),
        output_path=str(candidates),
        options=options,
    )
    if not models:
        raise RuntimeError("COLMAP did not produce a reconstruction")
    model_id, reconstruction = max(models.items(), key=lambda item: item[1].num_reg_images())
    reconstruction.write(str(selected_binary))
    reconstruction.write_text(str(selected_text))

    with sqlite3.connect(database) as connection:
        database_images = int(connection.execute("SELECT COUNT(*) FROM images").fetchone()[0])
        verified_pairs = int(
            connection.execute("SELECT COUNT(*) FROM two_view_geometries WHERE rows >= 15").fetchone()[0]
        )
    stats = {
        "database_images": database_images,
        "verified_pairs_min_15": verified_pairs,
        "candidate_models": len(models),
        "selected_candidate_id": int(model_id),
        "registered_images": int(reconstruction.num_reg_images()),
        "cameras": int(reconstruction.num_cameras()),
        "points3D": int(reconstruction.num_points3D()),
        "observations": int(reconstruction.compute_num_observations()),
        "mean_track_length": float(reconstruction.compute_mean_track_length()),
        "mean_observations_per_image": float(reconstruction.compute_mean_observations_per_reg_image()),
        "mean_reprojection_error_px": float(reconstruction.compute_mean_reprojection_error()),
        "binary_model": str(selected_binary.resolve()),
        "text_model": str(selected_text.resolve()),
    }
    (args.output / "reconstruction_metrics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
