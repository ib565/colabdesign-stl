import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.losses import chamfer_distance


def test_zero_for_identical():
    pts = jnp.array([[0.0, 1.0, 2.0], [1.0, 0.5, -1.0]], dtype=jnp.float32)
    loss = chamfer_distance(pts, pts)
    assert loss == pytest.approx(0.0)


def test_positive_for_offset():
    p1 = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
    p2 = jnp.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=jnp.float32)
    loss = chamfer_distance(p1, p2)
    assert loss > 0.0


def test_gradients_exist():
    target = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
    pred = jnp.array([[0.5, 0.2, 0.0], [1.5, -0.3, 0.1]], dtype=jnp.float32)

    grad_fn = jax.grad(lambda p: chamfer_distance(p, target).sum())
    grads = grad_fn(pred)
    assert grads.shape == pred.shape
    assert jnp.isfinite(grads).all()


def test_use_sqrt_behaves():
    target = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    pred = jnp.array([[3.0, 0.0, 0.0]], dtype=jnp.float32)

    loss_sq = chamfer_distance(pred, target, use_sqrt=False)
    loss_sqrt = chamfer_distance(pred, target, use_sqrt=True)

    assert loss_sq > 0
    assert loss_sqrt > 0
    assert loss_sqrt < loss_sq  # sqrt reduces magnitude


@pytest.mark.parametrize(
    "bad_shape",
    [
        np.zeros((0, 3), dtype=np.float32),
        np.zeros((3,), dtype=np.float32),
        np.zeros((3, 2), dtype=np.float32),
    ],
)
def test_invalid_inputs_raise(bad_shape):
    with pytest.raises(ValueError):
        chamfer_distance(bad_shape, bad_shape)

