import argparse
import trimesh


def make_cylinder_stl(
    output_path: str,
    radius: float = 5.0,
    height: float = 30.0,
    sections: int = 64,
) -> None:
    """Generate a simple cylinder STL for centerline testing."""
    mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=sections)
    mesh.export(output_path)
    print(f"Saved cylinder STL to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="examples/stl/cylinder.stl")
    parser.add_argument("--radius", type=float, default=5.0)
    parser.add_argument("--height", type=float, default=30.0)
    parser.add_argument("--sections", type=int, default=64)
    args = parser.parse_args()

    make_cylinder_stl(
        output_path=args.out,
        radius=args.radius,
        height=args.height,
        sections=args.sections,
    )


if __name__ == "__main__":
    main()

