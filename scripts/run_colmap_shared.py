from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

import pycolmap

from sfm_quality import reconstruction_image_quality, select_deferred_images


def mapping_options(*, random_seed: int) -> pycolmap.IncrementalPipelineOptions:
    options = pycolmap.IncrementalPipelineOptions()
    options.multiple_models = True
    options.min_model_size = 3
    options.max_num_models = 10
    options.ba_refine_principal_point = False
    options.random_seed = random_seed
    options.mapper.random_seed = random_seed
    options.triangulation.random_seed = random_seed
    return options


def audit_reconstruction(
    reconstruction,
    *,
    relative_observation_floor: float,
    minimum_triangulated_ratio: float,
    max_images: int,
) -> tuple[list, dict]:
    images = reconstruction_image_quality(reconstruction)
    deferred, thresholds = select_deferred_images(
        images,
        relative_observation_floor=relative_observation_floor,
        minimum_triangulated_ratio=minimum_triangulated_ratio,
        max_images=max_images,
    )
    report = {
        **thresholds,
        "relative_observation_floor": relative_observation_floor,
        "minimum_triangulated_ratio": minimum_triangulated_ratio,
        "images": [item.to_dict() for item in images],
        "deferred_images": [item.to_dict() for item in deferred],
    }
    return deferred, report


def retry_deferred_images(
    *,
    reconstruction,
    database: Path,
    images: Path,
    output: Path,
    args,
):
    deferred, initial_audit = audit_reconstruction(
        reconstruction,
        relative_observation_floor=args.defer_relative_observation_floor,
        minimum_triangulated_ratio=args.defer_min_triangulated_ratio,
        max_images=args.defer_max_images,
    )
    report = {"enabled": True, "initial_audit": initial_audit}
    if not deferred:
        report.update({"retried_images": [], "recovered_images": [], "excluded_images": []})
        return reconstruction, report

    workspace = output / "deferred_registration"
    initial_path = workspace / "initial" / "0"
    core_path = workspace / "core" / "0"
    retry_candidates = workspace / "retry_candidates"
    for path in (initial_path, core_path, retry_candidates):
        path.mkdir(parents=True, exist_ok=True)
    reconstruction.write(str(initial_path))

    core = pycolmap.Reconstruction(str(initial_path))
    deferred_names = [item.name for item in deferred]
    for item in deferred:
        core.deregister_frame(item.frame_id)
    core.write(str(core_path))

    retry_options = mapping_options(random_seed=args.random_seed)
    retry_options.multiple_models = False
    retry_options.fix_existing_frames = True
    retry_options.mapper.fix_existing_frames = True
    retry_options.mapper.abs_pose_max_error = args.retry_abs_pose_max_error
    retry_options.mapper.abs_pose_min_num_inliers = args.retry_abs_pose_min_num_inliers
    retry_options.mapper.abs_pose_min_inlier_ratio = args.retry_abs_pose_min_inlier_ratio
    retry_models = pycolmap.incremental_mapping(
        database_path=str(database),
        image_path=str(images),
        output_path=str(retry_candidates),
        input_path=str(core_path),
        options=retry_options,
    )
    if not retry_models:
        report.update(
            {
                "retried_images": deferred_names,
                "recovered_images": [],
                "excluded_images": deferred_names,
                "retry_failure": "COLMAP did not return a retry reconstruction",
            }
        )
        return core, report

    _, retried = max(retry_models.items(), key=lambda item: item[1].num_reg_images())
    failed_again, retry_audit = audit_reconstruction(
        retried,
        relative_observation_floor=args.defer_relative_observation_floor,
        minimum_triangulated_ratio=args.defer_min_triangulated_ratio,
        max_images=args.defer_max_images,
    )
    failed_names = {item.name for item in failed_again if item.name in deferred_names}
    registered_names = {
        retried.image(image_id).name for image_id in retried.reg_image_ids()
    }
    excluded_names = [
        name for name in deferred_names if name not in registered_names or name in failed_names
    ]
    for name in excluded_names:
        image = retried.find_image_with_name(name)
        if image is not None and image.has_pose:
            retried.deregister_frame(image.frame_id)
    recovered_names = [name for name in deferred_names if name not in excluded_names]
    report.update(
        {
            "retried_images": deferred_names,
            "recovered_images": recovered_names,
            "excluded_images": excluded_names,
            "retry_audit": retry_audit,
        }
    )
    return retried, report


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
    parser.add_argument("--deferred-registration", action="store_true")
    parser.add_argument("--defer-relative-observation-floor", type=float, default=0.15)
    parser.add_argument("--defer-min-triangulated-ratio", type=float, default=0.05)
    parser.add_argument("--defer-max-images", type=int, default=5)
    parser.add_argument("--retry-abs-pose-max-error", type=float, default=6.0)
    parser.add_argument("--retry-abs-pose-min-num-inliers", type=int, default=80)
    parser.add_argument("--retry-abs-pose-min-inlier-ratio", type=float, default=0.30)
    parser.add_argument("--random-seed", type=int, default=0)
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

    options = mapping_options(random_seed=args.random_seed)
    models = pycolmap.incremental_mapping(
        database_path=str(database),
        image_path=str(args.images),
        output_path=str(candidates),
        options=options,
    )
    if not models:
        raise RuntimeError("COLMAP did not produce a reconstruction")
    model_id, reconstruction = max(models.items(), key=lambda item: item[1].num_reg_images())
    deferred_report = {"enabled": False}
    if args.deferred_registration:
        reconstruction, deferred_report = retry_deferred_images(
            reconstruction=reconstruction,
            database=database,
            images=args.images,
            output=args.output,
            args=args,
        )
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
        "deferred_registration": deferred_report,
    }
    (args.output / "reconstruction_metrics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
