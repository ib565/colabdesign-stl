"""Differentiable loss utilities."""

from typing import Callable, Tuple

from jax import lax
import jax.numpy as jnp
import numpy as np


def _pairwise_sq_dists(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Compute pairwise squared distances between two point clouds.

    Args:
        pred: Array of shape (L, 3).
        target: Array of shape (N, 3).

    Returns:
        Array of shape (L, N) with squared distances.
    """
    diff = pred[:, None, :] - target[None, :, :]
    return jnp.sum(diff * diff, axis=-1)


def _validate_points(points: jnp.ndarray) -> Tuple[int, int]:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3); got {points.shape}.")
    if points.shape[0] == 0:
        raise ValueError("Points must be non-empty.")
    return points.shape


def chamfer_distance(
    pred: jnp.ndarray,
    target: jnp.ndarray,
    *,
    use_sqrt: bool = False,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Chamfer distance between two 3D point clouds.

    Uses squared distances by default for speed and smooth gradients. Set
    ``use_sqrt=True`` for more interpretable Å units (slightly slower).

    Args:
        pred: Predicted points, shape (L, 3).
        target: Target points, shape (N, 3).
        use_sqrt: If True, take sqrt of nearest distances (with epsilon).
        eps: Small constant for numerical stability when ``use_sqrt`` is True.

    Returns:
        Scalar jnp.ndarray loss.
    """
    pred = jnp.asarray(pred, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)

    _validate_points(pred)
    _validate_points(target)

    sq_dist = _pairwise_sq_dists(pred, target)

    if use_sqrt:
        loss_pred_to_target = jnp.mean(jnp.sqrt(jnp.min(sq_dist, axis=1) + eps))
        loss_target_to_pred = jnp.mean(jnp.sqrt(jnp.min(sq_dist, axis=0) + eps))
    else:
        loss_pred_to_target = jnp.mean(jnp.min(sq_dist, axis=1))
        loss_target_to_pred = jnp.mean(jnp.min(sq_dist, axis=0))

    return loss_pred_to_target + loss_target_to_pred


def _kabsch_align(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    """Align ``pred`` onto ``target`` via Kabsch (proper rotation only).

    Both inputs must be mean-centered before calling.
    """
    pred = jnp.asarray(pred, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    h = pred.T @ target
    u, _, vt = jnp.linalg.svd(h, full_matrices=False)
    r = vt.T @ u.T

    def _flip_last_row(_: None) -> jnp.ndarray:
        vt_fixed = vt.at[-1, :].set(-vt[-1, :])
        return vt_fixed.T @ u.T

    r = lax.cond(jnp.linalg.det(r) < 0, _flip_last_row, lambda _: r, operand=None)
    return pred @ r.T


def make_shape_loss(
    target_points: np.ndarray,
    *,
    use_sqrt: bool = False,
) -> Callable:
    """Create a Chamfer-based loss callback compatible with ColabDesign.

    The callback extracts Cα coordinates from the Alphafold structure module
    output, centers them, and computes Chamfer distance against the centered
    target point cloud. Intended for hallucination protocol via
    ``mk_afdesign_model(protocol="hallucination", loss_callback=...)``.

    Args:
        target_points: Target point cloud, shape (N, 3).
        use_sqrt: Whether to use the sqrt variant of Chamfer for Å units.

    Returns:
        A callable ``loss_fn(inputs, outputs, aux) -> dict`` that returns
        ``{"chamfer": loss}``.
    """

    target = jnp.asarray(target_points, dtype=jnp.float32)
    _validate_points(target)
    target_centered = target - target.mean(axis=0)

    # Atom order: N=0, CA=1, C=2, O=3, CB=4, ...
    CA_INDEX = 1

    def shape_loss(inputs, outputs, aux):
        positions = outputs["structure_module"]["final_atom_positions"]
        ca = positions[:, CA_INDEX, :]
        ca_centered = ca - ca.mean(axis=0)
        ca_aligned = _kabsch_align(ca_centered, target_centered)
        loss = chamfer_distance(ca_aligned, target_centered, use_sqrt=use_sqrt)
        return {"chamfer": loss}

    return shape_loss


def make_path_loss(
    target_points: np.ndarray,
) -> Callable:
    """Create a per-index MSE loss callback for ordered paths.

    Use when target_points has exactly L points matching the protein length,
    providing 1:1 correspondences. Applies Kabsch alignment before computing
    mean squared error per residue.

    This gives cleaner gradients than Chamfer for ordered targets and avoids
    "clumping" failure modes.

    Args:
        target_points: Ordered target path, shape (L, 3) where L = protein length.

    Returns:
        A callable ``loss_fn(inputs, outputs, aux) -> dict`` that returns
        ``{"path": loss}`` where loss is mean squared distance per residue.
    """
    target = jnp.asarray(target_points, dtype=jnp.float32)
    _validate_points(target)
    target_centered = target - target.mean(axis=0)

    CA_INDEX = 1

    def path_loss(inputs, outputs, aux):
        positions = outputs["structure_module"]["final_atom_positions"]
        ca = positions[:, CA_INDEX, :]
        ca_centered = ca - ca.mean(axis=0)
        ca_aligned = _kabsch_align(ca_centered, target_centered)
        # Per-index MSE: mean of squared distances at each position
        loss = jnp.mean(jnp.sum((ca_aligned - target_centered) ** 2, axis=-1))
        return {"path": loss}

    return path_loss

