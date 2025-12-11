"""
High-level orchestrator for designing proteins that match STL shapes.

This wraps STL loading, Chamfer loss wiring, and ColabDesign execution into a
single class plus convenience plotting utilities.
"""

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .losses import make_shape_loss
from .stl_processing import normalize_points, plot_point_cloud, stl_to_points


def _resolve_data_dir(user_dir: Optional[str]) -> Optional[Path]:
    """Resolve the AlphaFold params directory."""
    if user_dir:
        return Path(user_dir)
    env_var = os.environ.get("AF_DATA_DIR")
    if env_var:
        return Path(env_var)
    env_dir = Path.cwd().parent / "ColabDesign"
    return env_dir if env_dir.exists() else None


def _kabsch_align_np(pred: np.ndarray, target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align pred onto target using NumPy Kabsch; returns (pred_aligned, target_centered)."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    pred_centered = pred - pred.mean(axis=0)
    target_centered = target - target.mean(axis=0)

    h = pred_centered.T @ target_centered
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T

    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T

    pred_aligned = pred_centered @ r.T
    return pred_aligned.astype(np.float32), target_centered.astype(np.float32)


def _chamfer_np(
    pred: np.ndarray, target: np.ndarray, *, use_sqrt: bool = False, eps: float = 1e-8
) -> float:
    """Chamfer distance (NumPy) between two point clouds."""
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    diff = pred[:, None, :] - target[None, :, :]
    sq = np.sum(diff * diff, axis=-1)

    if use_sqrt:
        loss_pred_to_target = np.mean(np.sqrt(np.min(sq, axis=1) + eps))
        loss_target_to_pred = np.mean(np.sqrt(np.min(sq, axis=0) + eps))
    else:
        loss_pred_to_target = np.mean(np.min(sq, axis=1))
        loss_target_to_pred = np.mean(np.min(sq, axis=0))

    return float(loss_pred_to_target + loss_target_to_pred)


class STLProteinDesigner:
    """Design proteins to match STL shapes via Chamfer loss."""

    def __init__(
        self,
        stl_path: Optional[str] = None,
        target_points: Optional[np.ndarray] = None,
        *,
        protein_length: int = 100,
        num_target_points: int = 1000,
        target_extent: float = 100.0,
        center: bool = True,
        sample_seed: Optional[int] = 0,
        chamfer_weight: float = 1.0,
        plddt_weight: float = 0.1,
        pae_weight: float = 0.05,
        use_sqrt: bool = False,
        data_dir: Optional[str] = None,
        verbose: int = 1,
    ) -> None:
        try:
            from colabdesign import mk_afdesign_model  # pyright: ignore[reportMissingImports]
        except ImportError as exc:
            raise ImportError(
                "colabdesign is required. Install it (e.g., `pip install -e ../ColabDesign`)."
            ) from exc

        if stl_path is None and target_points is None:
            raise ValueError("Provide either stl_path or target_points.")
        if stl_path is not None and target_points is not None:
            raise ValueError("Provide only one of stl_path or target_points, not both.")

        self.stl_path = Path(stl_path) if stl_path is not None else None

        # Load or normalize target point cloud.
        if self.stl_path is not None:
            if not self.stl_path.exists():
                raise FileNotFoundError(f"STL not found: {self.stl_path}")
            self.target_points = stl_to_points(
                str(self.stl_path),
                num_points=num_target_points,
                target_extent=target_extent,
                center=center,
                seed=sample_seed,
            )
        else:
            pts = np.asarray(target_points, dtype=np.float32)
            self.target_points = normalize_points(
                pts, target_extent=target_extent, center=center
            )

        loss_fn = make_shape_loss(self.target_points, use_sqrt=use_sqrt)

        resolved_data_dir = _resolve_data_dir(data_dir)
        if resolved_data_dir is None:
            raise FileNotFoundError(
                "AlphaFold params not found. Provide --data-dir or set AF_DATA_DIR."
            )

        self.model = mk_afdesign_model(
            protocol="hallucination",
            loss_callback=loss_fn,
            data_dir=resolved_data_dir,
            
        )
        if not getattr(self.model, "_model_names", []):
            raise RuntimeError(
                f"No model params loaded. Check data_dir={resolved_data_dir} or AF_DATA_DIR."
            )

        # Explicitly set weights; many defaults are zero in hallucination.
        weights = self.model.opt["weights"]
        weights.update(
            {
                "chamfer": chamfer_weight,
                "con": weights.get("con", 1.0),
                "i_con": weights.get("i_con", 0.0),
                "plddt": plddt_weight,
                "pae": pae_weight,
                "exp_res": weights.get("exp_res", 0.0),
                "helix": weights.get("helix", 0.0),
            }
        )

        self.model.prep_inputs(length=protein_length)
        self.verbose = verbose
        self.use_sqrt = use_sqrt

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def design(
        self,
        *,
        soft_iters: int = 200,
        temp_iters: int = 100,
        hard_iters: int = 20,
        run_seed: int = 0,
        save_best: bool = True,
        verbose: Optional[int] = None,
    ) -> str:
        """Run design and return the best sequence."""
        self.model.restart(mode="gumbel", seed=run_seed)
        self.model.design_3stage(
            soft_iters=soft_iters,
            temp_iters=temp_iters,
            hard_iters=hard_iters,
            save_best=save_best,
            verbose=(verbose or self.verbose),
        )
        seqs: Sequence[str] = self.model.get_seqs()
        return seqs[0] if seqs else ""

    def get_structure(self, save_path: Optional[str] = None, get_best: bool = True) -> str:
        """Return PDB string; optionally write to file."""
        pdb_str = self.model.save_pdb(filename=None, get_best=get_best)
        if save_path:
            Path(save_path).write_text(pdb_str)
        return pdb_str

    def get_metrics(self) -> Dict[str, float]:
        """Return final (best if available) metrics."""
        log = (
            self.model._tmp.get("best", {}).get("aux", {}).get("log")  # pyright: ignore[reportPrivateUsage]
            or self.model.aux.get("log", {})
        )
        metrics = {
            "chamfer": float(log.get("chamfer", float("nan"))),
            "plddt": float(log.get("plddt", float("nan"))),
            "pae": float(log.get("pae", float("nan"))),
        }

        try:
            ca = self.get_ca_coords(get_best=True, aligned=False)
            ca_aligned, target_centered = _kabsch_align_np(ca, self.target_points)
            metrics["chamfer_aligned"] = _chamfer_np(
                ca_aligned, target_centered, use_sqrt=self.use_sqrt
            )
        except Exception:
            metrics["chamfer_aligned"] = float("nan")

        return metrics

    def get_ca_coords(self, get_best: bool = True, aligned: bool = False) -> np.ndarray:
        """Return Cα coordinates (aligned via Kabsch if requested)."""
        aux = self.model._tmp.get("best", {}).get("aux") if get_best else self.model.aux  # pyright: ignore[reportPrivateUsage]
        if aux is None:
            raise RuntimeError("No auxiliary data available; run design() first.")
        atom_positions = aux.get("atom_positions")
        if atom_positions is None:
            all_aux = aux.get("all")
            if all_aux is not None and "atom_positions" in all_aux:
                atom_positions = all_aux["atom_positions"]
        if atom_positions is None:
            raise RuntimeError("Atom positions not found in model outputs.")
        # If stacked across models, take the first entry.
        atom_positions = np.asarray(atom_positions)
        if atom_positions.ndim == 4:
            atom_positions = atom_positions[0]
        ca = np.asarray(atom_positions)[:, 1, :]  # CA index = 1
        if aligned:
            ca_aligned, _ = _kabsch_align_np(ca, self.target_points)
            return ca_aligned
        return ca - ca.mean(axis=0)

    # ------------------------------------------------------------------ #
    # Visualization
    # ------------------------------------------------------------------ #
    def plot_overlay(
        self,
        *,
        save_path: Optional[str] = None,
        show: bool = True,
        get_best: bool = True,
        title: str = "Target vs predicted Cα",
    ) -> None:
        """Plot target point cloud vs predicted Cα coordinates."""
        pred_ca = self.get_ca_coords(get_best=get_best, aligned=True)
        plot_point_cloud(pred_ca, target=self.target_points, title=title, save_path=save_path, show=show)


__all__ = ["STLProteinDesigner"]

