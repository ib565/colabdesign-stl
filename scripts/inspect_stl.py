#!/usr/bin/env python3
"""
Inspect an STL file: print mesh statistics and visualize point cloud sampling.

This script helps users understand what their STL file contains before running
protein design. It shows mesh properties, samples points (surface or centerline),
and generates visualization plots.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import trimesh

from src.stl_processing import (
    plot_point_cloud,
    stl_to_centerline_points,
    stl_to_points,
)


def inspect_mesh(stl_path: Path) -> None:
    """Load and print mesh statistics."""
    print(f"\n{'='*60}")
    print(f"Inspecting: {stl_path}")
    print(f"{'='*60}\n")
    
    if not stl_path.exists():
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    
    mesh = trimesh.load_mesh(str(stl_path), force="mesh")
    
    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError(f"STL file is empty or invalid: {stl_path}")
    
    # Basic stats
    print("Mesh Statistics:")
    print(f"  Vertices: {len(mesh.vertices):,}")
    print(f"  Faces: {len(mesh.faces):,}")
    print(f"  Watertight: {mesh.is_watertight}")
    print(f"  Volume: {mesh.volume:.3f} (mesh units)")
    print(f"  Surface area: {mesh.area:.3f} (mesh units)")
    
    # Bounding box
    bounds = mesh.bounds
    extents = bounds[1] - bounds[0]
    center = bounds.mean(axis=0)
    
    print(f"\nBounding Box:")
    print(f"  Min: ({bounds[0, 0]:.3f}, {bounds[0, 1]:.3f}, {bounds[0, 2]:.3f})")
    print(f"  Max: ({bounds[1, 0]:.3f}, {bounds[1, 1]:.3f}, {bounds[1, 2]:.3f})")
    print(f"  Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    print(f"  Extents: ({extents[0]:.3f}, {extents[1]:.3f}, {extents[2]:.3f})")
    print(f"  Longest dimension: {extents.max():.3f}")
    
    return mesh, extents


def sample_and_visualize(
    stl_path: Path,
    mode: str,
    num_points: int,
    target_extent: float,
    surface_samples: int,
    seed: int,
    output_dir: Path,
) -> None:
    """Sample points and create visualization."""
    print(f"\n{'='*60}")
    print(f"Sampling points ({mode} mode)")
    print(f"{'='*60}\n")
    
    if mode == "surface":
        points = stl_to_points(
            str(stl_path),
            num_points=num_points,
            target_extent=target_extent,
            center=True,
            seed=seed if seed >= 0 else None,
        )
        print(f"Sampled {len(points)} surface points")
        print(f"Target extent: {target_extent:.1f} Å")
        
    elif mode == "centerline":
        points = stl_to_centerline_points(
            str(stl_path),
            num_points=num_points,
            surface_samples=surface_samples,
            bins=None,
            smooth_window=5,
            seed=seed if seed >= 0 else None,
            target_arclength=None,
        )
        print(f"Extracted {len(points)} centerline points")
        print(f"Surface samples used: {surface_samples:,}")
        
        # Compute arclength
        seg = np.diff(points, axis=0)
        arclength = float(np.sum(np.linalg.norm(seg, axis=1)))
        avg_step = arclength / max(len(points) - 1, 1)
        print(f"Arclength: {arclength:.2f} Å")
        print(f"Average step: {avg_step:.2f} Å")
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Extent after normalization
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    print(f"Point cloud extent: {extent:.2f} Å")
    print(f"Point cloud center: ({points.mean(axis=0)[0]:.2f}, {points.mean(axis=0)[1]:.2f}, {points.mean(axis=0)[2]:.2f})")
    
    # Save visualization
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"{stl_path.stem}_{mode}_points.png"
    
    # Use connected lines for centerline mode (better visualization)
    plot_point_cloud(
        points,
        title=f"{stl_path.stem} ({mode})",
        show=False,
        save_path=str(plot_path),
        connected=(mode == "centerline"),
        view_angle=(30, 45),  # Isometric view
    )
    print(f"\nVisualization saved: {plot_path}")
    
    # Save points as numpy array
    npy_path = output_dir / f"{stl_path.stem}_{mode}_points.npy"
    np.save(str(npy_path), points)
    print(f"Points saved: {npy_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect STL file and visualize point cloud sampling."
    )
    parser.add_argument(
        "stl_path",
        type=str,
        help="Path to STL file (or name like 'cylinder' to resolve from examples/stl/)",
    )
    parser.add_argument(
        "--mode",
        choices=["surface", "centerline"],
        default="centerline",
        help="Sampling mode: surface (random) or centerline (ordered path)",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=80,
        help="Number of target points (for centerline, should match protein length)",
    )
    parser.add_argument(
        "--target-extent",
        type=float,
        default=30.0,
        help="Target extent in Å (for surface mode normalization)",
    )
    parser.add_argument(
        "--surface-samples",
        type=int,
        default=10000,
        help="Surface samples for centerline extraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (use -1 for stochastic)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/inspect",
        help="Output directory for plots and .npy files",
    )
    parser.add_argument(
        "--no-viz",
        action="store_true",
        help="Skip visualization (only print stats)",
    )
    
    args = parser.parse_args()
    
    # Resolve STL path
    stl_path = Path(args.stl_path)
    if not stl_path.exists() and not stl_path.suffix:
        # Try resolving from examples/stl/
        from examples.stl.generators.resolve_stl import resolve_or_generate_stl
        try:
            stl_path = resolve_or_generate_stl(args.stl_path)
        except FileNotFoundError:
            stl_path = Path(args.stl_path)
    
    # Inspect mesh
    mesh, extents = inspect_mesh(stl_path)
    
    # Sample and visualize
    if not args.no_viz:
        sample_and_visualize(
            stl_path=stl_path,
            mode=args.mode,
            num_points=args.num_points,
            target_extent=args.target_extent,
            surface_samples=args.surface_samples,
            seed=args.seed,
            output_dir=Path(args.output_dir),
        )


if __name__ == "__main__":
    main()

