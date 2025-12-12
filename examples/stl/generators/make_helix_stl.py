import argparse
import numpy as np
import trimesh
import sys
from pathlib import Path
# Ensure project root is on PYTHONPATH when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def make_helix_stl(
    output_path: str,
    turns: int = 3,
    radius: float = 10.0,
    pitch: float = 10.0,
    tube_radius: float = 2.0,
    samples_per_turn: int = 120,
    tube_segments: int = 24,
) -> None:
    """
    Build a helical tube mesh and save as STL.

    pitch: axial rise per turn (same units as radius).
    """
    t = np.linspace(0, 2 * np.pi * turns, samples_per_turn * turns)
    centerline = np.stack(
        [radius * np.cos(t), radius * np.sin(t), (pitch / (2 * np.pi)) * t], axis=1
    )

    tangents = np.gradient(centerline, axis=0)
    tangents /= np.linalg.norm(tangents, axis=1, keepdims=True) + 1e-9

    arbitrary = np.array([0.0, 0.0, 1.0])
    normals = np.cross(tangents, arbitrary)
    bad = np.linalg.norm(normals, axis=1) < 1e-6
    normals[bad] = np.cross(tangents[bad], np.array([1.0, 0.0, 0.0]))
    normals /= np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9
    binormals = np.cross(tangents, normals)

    angles = np.linspace(0, 2 * np.pi, tube_segments, endpoint=False)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=1)

    rings = []
    for p, n, b in zip(centerline, normals, binormals):
        ring = p + tube_radius * (circle[:, 0, None] * n + circle[:, 1, None] * b)
        rings.append(ring)
    vertices = np.vstack(rings)

    faces = []
    ring_size = tube_segments
    for i in range(len(centerline) - 1):
        a = i * ring_size
        c = (i + 1) * ring_size
        for j in range(ring_size):
            k = (j + 1) % ring_size
            faces.append([a + j, c + j, c + k])
            faces.append([a + j, c + k, a + k])

    mesh = trimesh.Trimesh(vertices=vertices, faces=np.array(faces), process=True)
    mesh.export(output_path)
    print(f"Saved helix STL to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/helix.stl")
    parser.add_argument("--turns", type=int, default=3)
    parser.add_argument("--radius", type=float, default=10.0)
    parser.add_argument("--pitch", type=float, default=10.0)
    parser.add_argument("--tube_radius", type=float, default=2.0)
    parser.add_argument("--samples_per_turn", type=int, default=120)
    parser.add_argument("--tube_segments", type=int, default=24)
    args = parser.parse_args()

    make_helix_stl(
        output_path=args.out,
        turns=args.turns,
        radius=args.radius,
        pitch=args.pitch,
        tube_radius=args.tube_radius,
        samples_per_turn=args.samples_per_turn,
        tube_segments=args.tube_segments,
    )


if __name__ == "__main__":
    main()

