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
                # "con": weights.get("con", 1.0),
                "con": 0.8,
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

    def debug_aux_structure(self) -> Dict[str, any]:
        """Debug helper to inspect aux data structure."""
        info = {}
        
        # Check _tmp["best"]
        best = self.model._tmp.get("best", {})  # pyright: ignore[reportPrivateUsage]
        info["has_best"] = bool(best)
        if best:
            info["best_keys"] = list(best.keys())
            best_aux = best.get("aux", {})
            info["best_aux_keys"] = list(best_aux.keys()) if best_aux else []
            if "log" in best_aux:
                info["best_log"] = dict(best_aux["log"])
            if "plddt" in best_aux:
                plddt = np.asarray(best_aux["plddt"])
                info["best_plddt_shape"] = plddt.shape
                info["best_plddt_mean"] = float(plddt.mean())
            if "atom_positions" in best_aux:
                pos = np.asarray(best_aux["atom_positions"])
                info["best_atom_positions_shape"] = pos.shape
        
        # Check current aux
        aux = self.model.aux
        info["has_aux"] = bool(aux)
        if aux:
            info["aux_keys"] = list(aux.keys())
            if "log" in aux:
                info["aux_log"] = dict(aux["log"])
            if "plddt" in aux:
                plddt = np.asarray(aux["plddt"])
                info["aux_plddt_shape"] = plddt.shape
                info["aux_plddt_mean"] = float(plddt.mean())
            if "atom_positions" in aux:
                pos = np.asarray(aux["atom_positions"])
                info["aux_atom_positions_shape"] = pos.shape
        
        return info

    def get_metrics(self) -> Dict[str, float]:
        """Return final (best if available) metrics.
        
        Returns metrics from the saved best checkpoint. Also recomputes
        chamfer_aligned from the actual CA coordinates for verification.
        """
        # Try to get log from best checkpoint, fall back to current aux
        best_aux = self.model._tmp.get("best", {}).get("aux", {})  # pyright: ignore[reportPrivateUsage]
        log = best_aux.get("log") if best_aux else None
        if not log:
            log = self.model.aux.get("log", {}) if self.model.aux else {}
        
        metrics = {
            "chamfer": float(log.get("chamfer", float("nan"))),
            "plddt": float(log.get("plddt", float("nan"))),
            "pae": float(log.get("pae", float("nan"))),
        }

        # Recompute chamfer from actual CA coordinates for verification
        try:
            ca = self.get_ca_coords(get_best=True, aligned=False)
            ca_aligned, target_centered = _kabsch_align_np(ca, self.target_points)
            metrics["chamfer_aligned"] = _chamfer_np(
                ca_aligned, target_centered, use_sqrt=self.use_sqrt
            )
            
            # Also compute pLDDT directly from coordinates if available
            aux = self.model._tmp.get("best", {}).get("aux") or self.model.aux  # pyright: ignore[reportPrivateUsage]
            if aux:
                plddt = aux.get("plddt")
                if plddt is not None:
                    plddt = np.asarray(plddt)
                    if plddt.ndim == 2:
                        # Multi-model: get best model's pLDDT
                        best_idx = self._get_best_model_index(aux)
                        metrics["plddt_recomputed"] = float(plddt[best_idx].mean())
                        metrics["_best_model_idx"] = best_idx
                    elif plddt.ndim == 1:
                        metrics["plddt_recomputed"] = float(plddt.mean())
        except Exception as e:
            metrics["chamfer_aligned"] = float("nan")
            metrics["_error"] = str(e)

        return metrics

    def _get_best_model_index(self, aux: dict) -> int:
        """Determine the best model index from aux data using pLDDT scores."""
        # Try to get pLDDT scores per model
        plddt = aux.get("plddt")
        if plddt is not None:
            plddt = np.asarray(plddt)
            if plddt.ndim == 2:  # (num_models, seq_len)
                # Average pLDDT per model, pick highest
                return int(np.argmax(plddt.mean(axis=1)))
            elif plddt.ndim == 1:
                return 0  # Single model
        
        # Fallback: check atom_positions shape and use model 0
        atom_positions = aux.get("atom_positions")
        if atom_positions is not None:
            atom_positions = np.asarray(atom_positions)
            if atom_positions.ndim == 4:
                # Try to use per-model pLDDT from "all" sub-dict
                all_aux = aux.get("all", {})
                if "plddt" in all_aux:
                    plddt_all = np.asarray(all_aux["plddt"])
                    if plddt_all.ndim == 2:
                        return int(np.argmax(plddt_all.mean(axis=1)))
        return 0

    def get_ca_coords(self, get_best: bool = True, aligned: bool = False) -> np.ndarray:
        """Return Cα coordinates (aligned via Kabsch if requested)."""
        aux = self.model._tmp.get("best", {}).get("aux") if get_best else self.model.aux  # pyright: ignore[reportPrivateUsage]
        if aux is None or not aux:
            # Fallback to current aux if best not available
            aux = self.model.aux
        if aux is None:
            raise RuntimeError("No auxiliary data available; run design() first.")
        
        atom_positions = aux.get("atom_positions")
        if atom_positions is None:
            all_aux = aux.get("all")
            if all_aux is not None and "atom_positions" in all_aux:
                atom_positions = all_aux["atom_positions"]
        if atom_positions is None:
            raise RuntimeError("Atom positions not found in model outputs.")
        
        atom_positions = np.asarray(atom_positions)
        
        # If stacked across models, select the best model (not just [0])
        if atom_positions.ndim == 4:
            best_idx = self._get_best_model_index(aux)
            atom_positions = atom_positions[best_idx]
        
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

