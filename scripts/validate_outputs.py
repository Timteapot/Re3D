from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh


def counts(path: Path) -> tuple[int, int]:
    loaded = trimesh.load(path, force="scene", process=False)
    geometries = loaded.geometry.values() if isinstance(loaded, trimesh.Scene) else [loaded]
    return (
        int(sum(len(mesh.vertices) for mesh in geometries)),
        int(sum(len(mesh.faces) for mesh in geometries)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--branches", nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    result: dict[str, object] = {"all_passed": True, "branches": {}}
    for label in args.branches:
        folder = args.outputs / label
        obj = folder / "mesh.obj"
        glb = folder / "mesh.glb"
        mtl = folder / "mesh.mtl"
        texture = folder / "texture.jpg"
        errors = [
            f"missing {path.name}"
            for path in (obj, glb, mtl, texture)
            if not path.exists() or path.stat().st_size == 0
        ]
        obj_counts = counts(obj) if obj.exists() else (0, 0)
        glb_counts = counts(glb) if glb.exists() else (0, 0)
        if obj_counts[1] <= 0 or glb_counts[1] <= 0:
            errors.append("OBJ or GLB has no faces")
        if obj_counts[1] != glb_counts[1]:
            errors.append("OBJ and GLB face counts differ")
        obj_text = obj.read_text(encoding="utf-8", errors="replace") if obj.exists() else ""
        mtl_text = mtl.read_text(encoding="utf-8", errors="replace") if mtl.exists() else ""
        if "mtllib mesh.mtl" not in obj_text:
            errors.append("OBJ does not reference mesh.mtl")
        if "map_Kd texture.jpg" not in mtl_text or "d 1.000000" not in mtl_text:
            errors.append("MTL texture or opacity normalization is missing")
        result["branches"][label] = {
            "status": "pass" if not errors else "fail",
            "errors": errors,
            "obj_vertices": obj_counts[0],
            "obj_faces": obj_counts[1],
            "glb_vertices": glb_counts[0],
            "glb_faces": glb_counts[1],
        }
        if errors:
            result["all_passed"] = False
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_passed"] else 1)


if __name__ == "__main__":
    main()
