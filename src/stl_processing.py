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

def make_hairpin_path(
    num_points: int = 80,
    height: float = 25.0,
    turn_radius: float = 8.0,
) -> np.ndarray:
    n_third = num_points // 3
    
    seg1_z = np.linspace(0, height, n_third)
    seg1_x = np.zeros(n_third)
    
    theta = np.linspace(0, np.pi, n_third)
    turn_x = turn_radius * (1 - np.cos(theta))
    turn_z = height + turn_radius * np.sin(theta)
    
    seg2_z = np.linspace(height, 0, num_points - 2 * n_third)
    seg2_x = np.full(num_points - 2 * n_third, 2 * turn_radius)
    
    x = np.concatenate([seg1_x, turn_x, seg2_x])
    z = np.concatenate([seg1_z, turn_z, seg2_z])
    y = np.zeros(num_points)
    
    return np.stack([x, y, z], axis=1).astype(np.float32)

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


def _pca_canonicalize(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center points and rotate into PCA frame (columns = principal axes)."""
    pts = _validate_points_np(points)
    centered = pts - pts.mean(axis=0)
    # SVD on covariance-equivalent matrix for robustness
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    rot = vt.T  # (3,3) columns = principal directions
    return centered @ rot, rot


def _polyline_arclength(polyline: np.ndarray) -> float:
    """Return total arclength of an ordered polyline."""
    if len(polyline) < 2:
        return 0.0
    seg = np.diff(polyline, axis=0)
    return float(np.sum(np.linalg.norm(seg, axis=1)))


def _resample_polyline_by_arclength(polyline: np.ndarray, num_points: int) -> np.ndarray:
    """Uniformly resample an ordered polyline to num_points via arclength interp."""
    if len(polyline) == 0:
        raise ValueError("Polyline is empty.")
    if len(polyline) == 1:
        return np.repeat(polyline.astype(np.float32), num_points, axis=0)

    seg = np.diff(polyline, axis=0)
    seg_len = np.linalg.norm(seg, axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    total = s[-1]
    if total <= 0:
        raise ValueError("Polyline has zero length; cannot resample.")

    target_s = np.linspace(0.0, total, num_points)
    resampled = np.empty((num_points, 3), dtype=np.float32)
    for dim in range(3):
        resampled[:, dim] = np.interp(target_s, s, polyline[:, dim])
    return resampled


def _moving_average(points: np.ndarray, window: int) -> np.ndarray:
    """Simple centered moving average; falls back to original if window<=1."""
    if window <= 1 or len(points) < 2:
        return points
    kernel = np.ones(window) / window
    pad = window // 2
    padded = np.pad(points, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.stack(
        [np.convolve(padded[:, i], kernel, mode="valid") for i in range(3)], axis=1
    )
    return smoothed.astype(np.float32)


def stl_to_centerline_points(
    stl_path: str,
    *,
    num_points: int,
    surface_samples: int = 10_000,
    bins: Optional[int] = None,
    smooth_window: int = 5,
    seed: Optional[int] = None,
    target_arclength: Optional[float] = None,
) -> np.ndarray:
    """
    Extract an ordered centerline polyline from a tube-like STL mesh.

    Steps:
      1) Sample many surface points.
      2) PCA canonicalize to choose a stable main axis.
      3) Bin along PC0, take centroids per bin (skip empty bins).
      4) Smooth (moving average).
      5) Resample by arclength to exactly ``num_points``.
      6) Optionally scale to a target arclength.

    Returns:
        (num_points, 3) float32, centered, ordered along the principal axis.
    """
    if not os.path.exists(stl_path):
        raise FileNotFoundError(f"STL file not found: {stl_path}")
    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError(f"STL file is empty or invalid: {stl_path}")

    if seed is not None:
        np.random.seed(seed)
    pts, _ = trimesh.sample.sample_surface(mesh, surface_samples)
    pts = _validate_points_np(pts)

    # PCA canonicalization
    pts_rot, rot = _pca_canonicalize(pts)
    axis_vals = pts_rot[:, 0]

    # Bin along PC0 with deterministic handling of empty bins (linear interp)
    b = bins or max(4 * num_points, 16)
    edges = np.linspace(axis_vals.min(), axis_vals.max(), b + 1)
    inds = np.digitize(axis_vals, edges) - 1
    centroids = np.full((b, 3), np.nan, dtype=np.float64)
    counts = np.zeros(b, dtype=np.int32)
    for bi in range(b):
        mask = inds == bi
        if np.any(mask):
            centroids[bi] = pts_rot[mask].mean(axis=0)
            counts[bi] = mask.sum()

    # Require at least 2 bins populated
    if np.count_nonzero(np.isfinite(centroids[:, 0])) < 2:
        raise ValueError("Centerline extraction failed: too few populated bins.")

    # Fill leading/trailing NaNs with nearest valid
    valid_idx = np.where(np.isfinite(centroids[:, 0]))[0]
    first, last = valid_idx[0], valid_idx[-1]
    centroids[:first] = centroids[first]
    centroids[last + 1 :] = centroids[last]

    # Linear interpolate interior NaNs per axis
    x = np.arange(b)
    for dim in range(3):
        y = centroids[:, dim]
        nan_mask = ~np.isfinite(y)
        if nan_mask.any():
            y[nan_mask] = np.interp(x[nan_mask], x[~nan_mask], y[~nan_mask])
            centroids[:, dim] = y

    poly = centroids.astype(np.float32)

    # Smooth
    poly = _moving_average(poly, smooth_window)

    # Ensure deterministic direction: if decreasing along PC0, flip
    if poly[-1, 0] < poly[0, 0]:
        poly = poly[::-1]

    # Resample to num_points along arclength
    poly = _resample_polyline_by_arclength(poly, num_points)

    # Optional arclength scaling
    if target_arclength is not None:
        current = _polyline_arclength(poly)
        if current <= 0:
            raise ValueError("Resampled centerline has zero arclength.")
        scale = float(target_arclength) / current
        poly = poly * scale

    # Return centered in original frame orientation (but we keep PCA frame;
    # caller can optionally rotate back if needed—here we keep PCA frame for stability).
    poly = poly - poly.mean(axis=0)
    return poly.astype(np.float32)


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

