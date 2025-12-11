import numpy as np
import pytest

from src.losses import _kabsch_align, chamfer_distance
from src.stl_processing import make_hairpin_path, normalize_points


def _random_rotation(seed: int = 0) -> np.ndarray:
    """Generate a deterministic random proper rotation."""
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(3, 3))
    q, r = np.linalg.qr(m)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q.astype(np.float32)


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def test_self_alignment_hits_zero():
    target = normalize_points(make_hairpin_path(num_points=120), target_extent=100.0, center=True)
    rot = _random_rotation(seed=1)
    trans = np.array([10.0, -4.0, 7.5], dtype=np.float32)

    pred = target @ rot.T + trans

    target_centered = target - target.mean(axis=0)
    pred_centered = pred - pred.mean(axis=0)

    aligned = _kabsch_align(pred_centered, target_centered)
    chamfer = chamfer_distance(aligned, target_centered, use_sqrt=True)

    assert chamfer == pytest.approx(0.0, abs=1e-3)
    assert _rmsd(aligned, target_centered) == pytest.approx(0.0, abs=1e-3)


def test_line_rotation_aligns_back():
    line = np.linspace([0, 0, 0], [100, 0, 0], 80, dtype=np.float32)
    target = normalize_points(line, target_extent=80.0, center=True)

    rot = _random_rotation(seed=2)
    pred = target @ rot.T  # rotation only; no translation

    target_centered = target - target.mean(axis=0)
    pred_centered = pred - pred.mean(axis=0)

    aligned = _kabsch_align(pred_centered, target_centered)
    chamfer = chamfer_distance(aligned, target_centered, use_sqrt=True)

    assert chamfer == pytest.approx(0.0, abs=1e-3)
    assert _rmsd(aligned, target_centered) == pytest.approx(0.0, abs=1e-3)

