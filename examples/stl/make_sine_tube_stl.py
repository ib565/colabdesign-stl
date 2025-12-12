import argparse
import numpy as np
import trimesh


def make_sine_tube_stl(
    output_path: str,
    length: float = 30.0,
    amplitude: float = 6.0,
    tube_radius: float = 1.5,
    samples: int = 300,
    tube_segments: int = 24,
) -> None:
    """Generate a single-oscillation sine tube STL (monotonic in Z)."""
    t = np.linspace(0, 1.0, samples)
    z = length * t
    x = amplitude * np.sin(2 * np.pi * t)  # one oscillation
    y = np.zeros_like(t)
    centerline = np.stack([x, y, z], axis=1)

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
    print(f"Saved sine-tube STL to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/stl/sine_tube.stl")
    parser.add_argument("--length", type=float, default=30.0)
    parser.add_argument("--amplitude", type=float, default=6.0)
    parser.add_argument("--tube_radius", type=float, default=1.5)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--tube_segments", type=int, default=24)
    args = parser.parse_args()

    make_sine_tube_stl(
        output_path=args.out,
        length=args.length,
        amplitude=args.amplitude,
        tube_radius=args.tube_radius,
        samples=args.samples,
        tube_segments=args.tube_segments,
    )


if __name__ == "__main__":
    main()

