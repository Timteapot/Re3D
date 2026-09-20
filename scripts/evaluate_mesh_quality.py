from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy import sparse
from scipy.sparse.csgraph import connected_components


def load_mesh(path: Path) -> trimesh.Trimesh:
    loaded = trimesh.load(path, force="mesh", process=False)
    if isinstance(loaded, trimesh.Scene):
        if not loaded.geometry:
            raise ValueError(f"No geometry in {path}")
        return trimesh.util.concatenate(tuple(loaded.geometry.values()))
    return loaded


def evaluate(path: Path) -> dict:
    mesh = load_mesh(path)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError(f"Mesh has no vertices or faces: {path}")
    finite_vertices = np.all(np.isfinite(vertices), axis=1)
    valid_indices = np.all((faces >= 0) & (faces < len(vertices)), axis=1)
    if not np.all(valid_indices):
        raise ValueError(f"Mesh contains out-of-range face indices: {path}")

    triangles = vertices[faces]
    double_area = np.linalg.norm(
        np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]),
        axis=1,
    )
    edges = np.vstack(
        (faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]])
    )
    edges.sort(axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    adjacency = sparse.coo_matrix(
        (
            np.ones(len(unique_edges) * 2, dtype=np.uint8),
            (
                np.concatenate((unique_edges[:, 0], unique_edges[:, 1])),
                np.concatenate((unique_edges[:, 1], unique_edges[:, 0])),
            ),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    component_count, vertex_labels = connected_components(
        adjacency, directed=False, return_labels=True
    )
    component_faces = np.bincount(vertex_labels[faces[:, 0]], minlength=component_count)
    largest_faces = int(component_faces.max()) if len(component_faces) else 0
    return {
        "mesh": str(path.resolve()),
        "vertices": int(len(vertices)),
        "faces": int(len(faces)),
        "finite_vertex_ratio": float(finite_vertices.mean()),
        "degenerate_faces": int(np.count_nonzero(double_area <= 1e-12)),
        "connected_components": int(component_count),
        "largest_component_faces": largest_faces,
        "largest_component_face_ratio": largest_faces / len(faces),
        "boundary_edges": int(np.count_nonzero(edge_counts == 1)),
        "nonmanifold_edges": int(np.count_nonzero(edge_counts > 2)),
        "watertight": bool(np.all(edge_counts == 2)),
        "surface_area": float(double_area.sum() * 0.5),
        "bounds": np.asarray([vertices.min(axis=0), vertices.max(axis=0)]).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute structural mesh quality indicators for comparisons."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("meshes", nargs="+", help="label=path pairs")
    args = parser.parse_args()
    results: dict[str, dict] = {}
    for item in args.meshes:
        label, raw_path = item.split("=", 1)
        print(f"Evaluating {label}: {raw_path}", flush=True)
        results[label] = evaluate(Path(raw_path))
    report = {"schema_version": 1, "meshes": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
