import argparse
import numpy as np

from src.stl_processing import plot_point_cloud, stl_to_points


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stl", default="examples/helix.stl")
    parser.add_argument("--num_points", type=int, default=1000)
    parser.add_argument("--extent", type=float, default=100.0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    pts = stl_to_points(
        args.stl,
        num_points=args.num_points,
        target_extent=args.extent,
        seed=args.seed,
    )

    print(f"points shape: {pts.shape}")
    print(f"mean (should be ~0): {pts.mean(axis=0)}")
    bbox = pts.max(axis=0) - pts.min(axis=0)
    print(f"bbox: {bbox} (longest={bbox.max():.2f}, target={args.extent})")

    assert pts.shape == (args.num_points, 3)
    assert np.abs(pts.mean(axis=0)).max() < 1.0
    assert bbox.max() <= args.extent * 1.01

    if args.plot:
        plot_point_cloud(pts, title="Sampled STL points")


if __name__ == "__main__":
    main()

