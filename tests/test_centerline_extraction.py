import numpy as np
import pytest

from src.stl_processing import (
    _polyline_arclength,
    stl_to_centerline_points,
)
from examples.stl.generators.make_cylinder_stl import make_cylinder_stl
from examples.stl.generators.make_sine_tube_stl import make_sine_tube_stl


def test_cylinder_centerline(tmp_path):
    stl_path = tmp_path / "cyl.stl"
    make_cylinder_stl(str(stl_path), radius=5.0, height=30.0, sections=32)

    pts = stl_to_centerline_points(
        str(stl_path),
        num_points=50,
        surface_samples=5000,
        target_arclength=None,
        seed=0,
    )
    assert pts.shape == (50, 3)
    assert np.isfinite(pts).all()
    # Centered
    assert np.abs(pts.mean(axis=0)).max() < 1e-3
    # Nearly straight in PCA frame: small radial spread (allow modest noise)
    radial = np.linalg.norm(pts[:, 1:], axis=1)
    assert np.max(radial) < 1.5


def test_sine_centerline_has_curvature(tmp_path):
    stl_path = tmp_path / "sine.stl"
    make_sine_tube_stl(str(stl_path), amplitude=5.0, length=30.0, tube_radius=1.5)

    pts = stl_to_centerline_points(
        str(stl_path),
        num_points=60,
        surface_samples=8000,
        target_arclength=None,
        seed=1,
    )
    assert pts.shape == (60, 3)
    # Check nontrivial x-range (curvature)
    x_range = pts[:, 0].max() - pts[:, 0].min()
    assert x_range > 5.0
    # Monotonic-ish along principal axis (post PCA + enforced direction)
    diffs = np.diff(pts[:, 0])
    assert np.sum(diffs < 0) < len(diffs) * 0.4  # allow some noise


def test_arclength_rescale(tmp_path):
    stl_path = tmp_path / "cyl2.stl"
    make_cylinder_stl(str(stl_path), radius=4.0, height=20.0, sections=32)
    target_arc = 80.0
    pts = stl_to_centerline_points(
        str(stl_path),
        num_points=40,
        surface_samples=4000,
        target_arclength=target_arc,
        seed=0,
    )
    arc = _polyline_arclength(pts)
    assert np.isfinite(arc)
    assert arc == pytest.approx(target_arc, rel=0.05)

