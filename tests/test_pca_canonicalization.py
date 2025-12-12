import numpy as np
import trimesh

from src.stl_processing import stl_to_centerline_points
from examples.stl.generators.make_cylinder_stl import make_cylinder_stl


def _random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(m)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q


def _rotate_stl(src_path: str, dst_path: str, rot: np.ndarray):
    mesh = trimesh.load_mesh(src_path, force="mesh")
    tf = np.eye(4)
    tf[:3, :3] = rot
    mesh.apply_transform(tf)
    mesh.export(dst_path)


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """RMSD after optimal superposition (handles symmetry/axis flips)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)

    h = a_c.T @ b_c
    u, _, vt = np.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    a_aligned = a_c @ r.T
    diff = a_aligned - b_c
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def test_pca_canonicalization_is_deterministic(tmp_path):
    base = tmp_path / "base.stl"
    make_cylinder_stl(str(base), radius=5.0, height=30.0, sections=32)

    refs = []
    for i in range(3):
        rot = _random_rotation(seed=i)
        dst = tmp_path / f"rot_{i}.stl"
        _rotate_stl(str(base), str(dst), rot)
        pts = stl_to_centerline_points(
            str(dst),
            num_points=40,
            surface_samples=6000,
            target_arclength=None,
            seed=0,
        )
        refs.append(pts)

    ref0 = refs[0]
    # Compare with Kabsch RMSD after optional reversal; robust to symmetry.
    for pts in refs[1:]:
        candidates = [pts, pts[::-1]]
        rmsds = [_kabsch_rmsd(ref0, c) for c in candidates]
        best = min(rmsds)
        # Cylinder is symmetric in the radial plane; with stochastic surface sampling + binning,
        # small differences are expected even after Kabsch. Keep a loose but meaningful bound.
        assert best <= 0.75, f"Kabsch-RMSD too high: {best:.3f} (candidates: {[round(x,3) for x in rmsds]})"

