"""
Minimal demo for chamfer_distance: computes loss and gradient.

Run:
    python examples/chamfer_demo.py
"""

import jax
import jax.numpy as jnp

from src.losses import chamfer_distance


def main():
    target = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
    pred = jnp.array([[0.5, 0.2, 0.0], [1.5, -0.3, 0.1]], dtype=jnp.float32)

    loss_sq = chamfer_distance(pred, target, use_sqrt=False)
    loss_sqrt = chamfer_distance(pred, target, use_sqrt=True)

    grad_fn = jax.grad(lambda p: chamfer_distance(p, target).sum())
    grads = grad_fn(pred)

    print("Chamfer (squared):", float(loss_sq))
    print("Chamfer (sqrt):   ", float(loss_sqrt))
    print("Gradients shape:", grads.shape)
    print("Gradients:\n", grads)


if __name__ == "__main__":
    main()

