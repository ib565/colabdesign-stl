"""
Quick local sanity checks for centering + Kabsch alignment.

Run:
    python examples/alignment_sanity.py
"""

import argparse
import numpy as np
import sys
from pathlib import Path
# Ensure project root is on PYTHONPATH when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))






from src.losses import _kabsch_align, chamfer_distance
from src.stl_processing import make_hairpin_path, normalize_points


def _random_rotation(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.normal(size=(3, 3))
    q, _ = np.linalg.qr(m)
    if np.linalg.det(q) < 0:
        q[:, -1] *= -1
    return q.astype(np.float32)


def _rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff * diff, axis=1))))


def _center(arr: np.ndarray) -> np.ndarray:
    return arr - arr.mean(axis=0)


def check_self_alignment(num_points: int, target_extent: float) -> None:
    target = normalize_points(
        make_hairpin_path(num_points=num_points),
        target_extent=target_extent,
        center=True,
    )
    rot = _random_rotation(seed=1)
    trans = np.array([10.0, -4.0, 7.5], dtype=np.float32)
    pred = target @ rot.T + trans

    aligned = _kabsch_align(_center(pred), _center(target))
    chamfer = chamfer_distance(aligned, _center(target), use_sqrt=True)
    print(f"[Self-align] chamfer={float(chamfer):.3e}, rmsd={_rmsd(aligned, _center(target)):.3e}")


def check_line_alignment(length: float, num_points: int) -> None:
    line = np.linspace([0, 0, 0], [length, 0, 0], num_points, dtype=np.float32)
    target = normalize_points(line, target_extent=length, center=True)

    rot = _random_rotation(seed=2)
    pred = target @ rot.T

    aligned = _kabsch_align(_center(pred), _center(target))
    chamfer = chamfer_distance(aligned, _center(target), use_sqrt=True)
    print(f"[Line-align] chamfer={float(chamfer):.3e}, rmsd={_rmsd(aligned, _center(target)):.3e}")


def main():
    parser = argparse.ArgumentParser(description="Sanity checks for Kabsch alignment + centering.")
    parser.add_argument("--num-points", type=int, default=120, help="Points in hairpin path.")
    parser.add_argument("--target-extent", type=float, default=100.0, help="Extent used in normalization.")
    parser.add_argument("--line-length", type=float, default=80.0, help="Length of straight line test.")
    parser.add_argument("--line-points", type=int, default=80, help="Points in straight line test.")
    args = parser.parse_args()

    check_self_alignment(num_points=args.num_points, target_extent=args.target_extent)
    check_line_alignment(length=args.line_length, num_points=args.line_points)


if __name__ == "__main__":
    main()

