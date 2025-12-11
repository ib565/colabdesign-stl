import os
from typing import Optional

import numpy as np
import trimesh


def _validate_points_np(points: np.ndarray) -> np.ndarray:
    """Ensure points are an array of shape (N, 3) and non-empty."""
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3); got {arr.shape}.")
    if arr.shape[0] == 0:
        raise ValueError("Points must be non-empty.")
    return arr


def normalize_points(
    points: np.ndarray,
    *,
    target_extent: float = 100.0,
    center: bool = True,
) -> np.ndarray:
    """
    Center and isotropically scale a point cloud to a target extent.

    Args:
        points: Array of shape (N, 3).
        target_extent: Longest dimension (Å) after scaling.
        center: If True, center the cloud at the origin.

    Returns:
        Array of shape (N, 3) with dtype float32.
    """
    pts = _validate_points_np(points)
    if center:
        pts = pts - pts.mean(axis=0)

    extent = (pts.max(axis=0) - pts.min(axis=0)).max()
    if extent <= 0:
        raise ValueError("Point cloud has zero extent; cannot scale.")

    pts = pts * (target_extent / extent)
    return pts.astype(np.float32)


def make_helix_path(
    num_points: int = 100,
    radius: float = 10.0,
    pitch: float = 5.0,
    turns: float = 3.0,
) -> np.ndarray:
    """
    Generate points along a helical path (not a tube surface).

    Returns:
        Array of shape (num_points, 3) with dtype float32.
    """
    t = np.linspace(0, 2 * np.pi * turns, num_points)
    x = radius * np.cos(t)
    y = radius * np.sin(t)
    z = (pitch / (2 * np.pi)) * t
    return np.stack([x, y, z], axis=1).astype(np.float32)


def stl_to_points(
    stl_path: str,
    num_points: int = 100,
    target_extent: float = 100.0,
    center: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Load an STL mesh and sample a centered, scaled point cloud.

    Args:
        stl_path: Path to STL file.
        num_points: Number of points to sample from the surface.
        target_extent: Longest dimension (Å) after scaling.
        center: If True, center the cloud at the origin.
        seed: Optional random seed for reproducibility.

    Returns:
        Array of shape (num_points, 3) with dtype float32.
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")

    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError(f"STL file is empty or invalid: {stl_path}")

    if seed is not None:
        np.random.seed(seed)

    points, _ = trimesh.sample.sample_surface(mesh, num_points)

    return normalize_points(points, target_extent=target_extent, center=center)


def plot_point_cloud(
    points: np.ndarray,
    target: Optional[np.ndarray] = None,
    title: str = "Point cloud",
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Quick 3D scatter for visual inspection."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=4, alpha=0.6, label="points")
    if target is not None:
        ax.scatter(target[:, 0], target[:, 1], target[:, 2], s=4, alpha=0.6, label="target")
    ax.set_title(title)
    ax.legend()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

