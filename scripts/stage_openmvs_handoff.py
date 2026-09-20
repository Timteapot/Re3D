from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

from openmvs_dmap import read_dmap_header, sha256_file


def stage(source: Path, output: Path) -> dict:
    export_manifest = source / "dmap-export.json"
    scene = source / "scene.mvs"
    dmaps = sorted(source.glob("depth*.dmap"))
    if not export_manifest.is_file() or not scene.is_file() or not dmaps:
        raise FileNotFoundError(
            f"Incomplete canonical handoff in {source}; expected manifest, scene and DMAPs"
        )
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"OpenMVS staging directory must be empty: {output}")

    temporary = output.with_name(output.name + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"Remove stale temporary staging directory: {temporary}")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.rmdir()
    temporary.mkdir()
    sources = [scene, export_manifest, *dmaps]
    records: list[dict] = []
    try:
        for source_path in sources:
            staged_path = temporary / source_path.name
            shutil.copy2(source_path, staged_path)
            source_hash = sha256_file(source_path)
            staged_hash = sha256_file(staged_path)
            if source_hash != staged_hash:
                raise RuntimeError(f"Copy hash mismatch for {source_path.name}")
            records.append(
                {
                    "file": source_path.name,
                    "source_sha256": source_hash,
                    "staged_sha256": staged_hash,
                }
            )
        missing_images: list[str] = []
        for dmap in dmaps:
            staged_dmap = temporary / dmap.name
            image_name = str(read_dmap_header(staged_dmap)["file_name"])
            referenced_image = (temporary / image_name).resolve()
            if not referenced_image.is_file():
                missing_images.append(f"{dmap.name} -> {referenced_image}")
        if missing_images:
            preview = "; ".join(missing_images[:5])
            suffix = " ..." if len(missing_images) > 5 else ""
            raise FileNotFoundError(
                "Staging would break OpenMVS relative image references: "
                f"{preview}{suffix}"
            )
        manifest = {
            "schema_version": 1,
            "source": str(source.resolve()),
            "output": str(output.resolve()),
            "file_count": len(records),
            "records": records,
        }
        (temporary / "staging-manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def verify(manifest_path: Path) -> dict:
    errors: list[str] = []
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {
            "schema_version": 1,
            "all_passed": False,
            "errors": [f"Cannot read staging manifest {manifest_path}: {exc}"],
            "records": [],
        }
    source = Path(manifest.get("source", ""))
    results: list[dict] = []
    for record in manifest.get("records", []):
        path = source / str(record.get("file", ""))
        file_errors: list[str] = []
        if not path.is_file():
            file_errors.append("canonical source file is missing")
        elif sha256_file(path) != record.get("source_sha256"):
            file_errors.append("canonical source hash changed after staging/fusion")
        if file_errors:
            errors.extend(f"{path.name}: {message}" for message in file_errors)
        results.append(
            {"file": path.name, "passed": not file_errors, "errors": file_errors}
        )
    if manifest.get("file_count") != len(results):
        errors.append("staging manifest file_count does not match its records")
    return {
        "schema_version": 1,
        "all_passed": not errors,
        "staging_manifest": str(manifest_path.resolve()),
        "canonical_source": str(source.resolve()),
        "file_count": len(results),
        "errors": errors,
        "records": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage an immutable DMAP handoff or verify its source hashes."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    stage_parser = subparsers.add_parser("stage")
    stage_parser.add_argument("--source", type=Path, required=True)
    stage_parser.add_argument("--output", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--manifest", type=Path, required=True)
    verify_parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "stage":
        result = stage(args.source, args.output)
        print(
            json.dumps(
                {key: value for key, value in result.items() if key != "records"},
                indent=2,
            )
        )
        return
    result = verify(args.manifest)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    if not result["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
