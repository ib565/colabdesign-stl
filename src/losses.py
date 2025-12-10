"""Differentiable loss utilities."""

from typing import Tuple

import jax.numpy as jnp


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

