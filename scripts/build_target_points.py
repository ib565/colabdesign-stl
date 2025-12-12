#!/usr/bin/env python3
"""
Build target points from various sources (STL, line, helix, hairpin) and save
as .npy + visualization.

This script decouples "target point generation" from "protein design", making
it easier to inspect and debug target geometries before running expensive
AlphaFold optimization.
"""

import argparse
import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from src.stl_processing import (
    make_helix_path,
    make_hairpin_path,
    normalize_points,
    plot_point_cloud,
    stl_to_centerline_points,
)
from examples.stl.generators.resolve_stl import resolve_or_generate_stl


def build_line_path(num_points: int, length: float, target_extent: float) -> np.ndarray:
    """Build a straight line path."""
    z = np.linspace(-length / 2.0, length / 2.0, num_points, dtype=np.float32)
    zeros = np.zeros_like(z)
    line = np.stack([zeros, zeros, z], axis=1)
    return normalize_points(line, target_extent=target_extent, center=True)


def build_helix_path(
    num_points: int,
    radius: float,
    pitch: float,
    turns: float,
    target_extent: float,
) -> np.ndarray:
    """Build a helical path."""
    helix = make_helix_path(
        num_points=num_points,
        radius=radius,
        pitch=pitch,
        turns=turns,
    )
    return normalize_points(helix, target_extent=target_extent, center=True)


def build_hairpin_path(num_points: int, target_extent: float) -> np.ndarray:
    """Build a hairpin path."""
    hairpin = make_hairpin_path(num_points=num_points)
    return normalize_points(hairpin, target_extent=target_extent, center=True)


def build_stl_centerline(
    stl_path: Path,
    num_points: int,
    target_extent: float,
    surface_samples: int,
    bins: int,
    smooth_window: int,
    seed: int,
    target_arclength: float,
    normalize: bool,
) -> np.ndarray:
    """Build centerline from STL."""
    stl_resolved = resolve_or_generate_stl(str(stl_path))
    pts = stl_to_centerline_points(
        str(stl_resolved),
        num_points=num_points,
        surface_samples=surface_samples,
        bins=bins,
        smooth_window=smooth_window,
        seed=seed if seed >= 0 else None,
        target_arclength=target_arclength,
    )
    if normalize:
        return normalize_points(pts, target_extent=target_extent, center=True)
    else:
        return (pts - pts.mean(axis=0)).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(
        description="Build target points from various sources and save as .npy + visualization."
    )
    parser.add_argument(
        "--mode",
        choices=["line", "helix", "hairpin", "stl_centerline"],
        required=True,
        help="Target path mode",
    )
    
    # Common parameters
    parser.add_argument(
        "--num-points",
        type=int,
        default=80,
        help="Number of target points",
    )
    parser.add_argument(
        "--target-extent",
        type=float,
        default=30.0,
        help="Target extent in Å (for normalization)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/target_points",
        help="Output directory",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (use -1 for stochastic)",
    )
    
    # Line parameters
    parser.add_argument(
        "--line-length",
        type=float,
        default=40.0,
        help="Line length (for line mode)",
    )
    
    # Helix parameters
    parser.add_argument(
        "--helix-radius",
        type=float,
        default=5.0,
        help="Helix radius (for helix mode)",
    )
    parser.add_argument(
        "--helix-pitch",
        type=float,
        default=5.0,
        help="Helix pitch (for helix mode)",
    )
    parser.add_argument(
        "--helix-turns",
        type=float,
        default=1.0,
        help="Helix turns (for helix mode)",
    )
    
    # STL centerline parameters
    parser.add_argument(
        "--stl-path",
        type=str,
        help="STL file path (for stl_centerline mode)",
    )
    parser.add_argument(
        "--surface-samples",
        type=int,
        default=10000,
        help="Surface samples for centerline extraction",
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=None,
        help="Bins for centerline extraction (default: 4*num_points)",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Moving average window for centerline smoothing",
    )
    parser.add_argument(
        "--target-arclength",
        type=float,
        default=None,
        help="Target arclength (optional, for stl_centerline)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Skip normalization (only center)",
    )
    
    args = parser.parse_args()
    
    # Build target points
    print(f"Building target points ({args.mode} mode)...")
    
    if args.mode == "line":
        points = build_line_path(
            num_points=args.num_points,
            length=args.line_length,
            target_extent=args.target_extent,
        )
        name = f"line_L{args.num_points}_extent{args.target_extent}"
        
    elif args.mode == "helix":
        points = build_helix_path(
            num_points=args.num_points,
            radius=args.helix_radius,
            pitch=args.helix_pitch,
            turns=args.helix_turns,
            target_extent=args.target_extent,
        )
        name = f"helix_L{args.num_points}_r{args.helix_radius}_p{args.helix_pitch}_t{args.helix_turns}"
        
    elif args.mode == "hairpin":
        points = build_hairpin_path(
            num_points=args.num_points,
            target_extent=args.target_extent,
        )
        name = f"hairpin_L{args.num_points}_extent{args.target_extent}"
        
    elif args.mode == "stl_centerline":
        if not args.stl_path:
            raise ValueError("--stl-path required for stl_centerline mode")
        points = build_stl_centerline(
            stl_path=Path(args.stl_path),
            num_points=args.num_points,
            target_extent=args.target_extent,
            surface_samples=args.surface_samples,
            bins=args.bins,
            smooth_window=args.smooth_window,
            seed=args.seed,
            target_arclength=args.target_arclength,
            normalize=not args.no_normalize,
        )
        stl_name = Path(args.stl_path).stem
        name = f"{stl_name}_centerline_L{args.num_points}_extent{args.target_extent}"
    
    # Diagnostics
    extent = (points.max(axis=0) - points.min(axis=0)).max()
    center = points.mean(axis=0)
    
    if args.mode in ["line", "helix", "hairpin", "stl_centerline"]:
        seg = np.diff(points, axis=0)
        arclength = float(np.sum(np.linalg.norm(seg, axis=1)))
        avg_step = arclength / max(len(points) - 1, 1)
        print(f"Arclength: {arclength:.2f} Å")
        print(f"Average step: {avg_step:.2f} Å")
    
    print(f"Extent: {extent:.2f} Å")
    print(f"Center: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})")
    
    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    npy_path = output_dir / f"{name}.npy"
    np.save(str(npy_path), points)
    print(f"\nPoints saved: {npy_path}")
    
    png_path = output_dir / f"{name}.png"
    # Use connected lines for all path modes (better visualization)
    plot_point_cloud(
        points,
        title=f"{name}",
        show=False,
        save_path=str(png_path),
        connected=True,  # All modes here are paths (line/helix/hairpin/centerline)
        view_angle=(30, 45),  # Isometric view
    )
    print(f"Visualization saved: {png_path}")


if __name__ == "__main__":
    main()

