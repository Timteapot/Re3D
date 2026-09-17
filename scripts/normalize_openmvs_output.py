from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import trimesh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=Path, required=True)
    parser.add_argument("--mtl", type=Path, required=True)
    parser.add_argument("--texture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    obj_out = args.output / "mesh.obj"
    mtl_out = args.output / "mesh.mtl"
    texture_out = args.output / ("texture" + args.texture.suffix.lower())

    obj_text = args.obj.read_text(encoding="utf-8", errors="replace")
    lines = obj_text.splitlines()
    replaced = False
    for index, line in enumerate(lines):
        if line.startswith("mtllib "):
            lines[index] = "mtllib mesh.mtl"
            replaced = True
            break
    if not replaced:
        lines.insert(0, "mtllib mesh.mtl")
    obj_out.write_text("\n".join(lines) + "\n", encoding="utf-8")

    mtl_text = args.mtl.read_text(encoding="utf-8", errors="replace")
    mtl_lines = mtl_text.splitlines()
    opacity_written = False
    for index, line in enumerate(mtl_lines):
        if line.startswith("map_Kd "):
            mtl_lines[index] = f"map_Kd {texture_out.name}"
        elif line.startswith("Tr ") or line.startswith("d "):
            if opacity_written:
                mtl_lines[index] = ""
            else:
                mtl_lines[index] = "d 1.000000"
                opacity_written = True
    if not opacity_written:
        mtl_lines.append("d 1.000000")
    mtl_lines = [line for line in mtl_lines if line]
    mtl_out.write_text("\n".join(mtl_lines) + "\n", encoding="utf-8")
    shutil.copy2(args.texture, texture_out)

    scene = trimesh.load(obj_out, force="scene", process=False)
    glb_out = args.output / "mesh.glb"
    scene.export(glb_out)

    manifest = {
        "source_obj": str(args.obj.resolve()),
        "source_mtl": str(args.mtl.resolve()),
        "source_texture": str(args.texture.resolve()),
        "obj": str(obj_out.resolve()),
        "mtl": str(mtl_out.resolve()),
        "texture": str(texture_out.resolve()),
        "glb": str(glb_out.resolve()),
    }
    (args.output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
