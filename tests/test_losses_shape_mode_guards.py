import numpy as np
import pytest

from src.losses import make_shape_loss


def test_make_shape_loss_raises_on_wrong_rank():
    target = np.zeros((0, 3), dtype=np.float32)
    with pytest.raises(ValueError):
        make_shape_loss(target)


def test_make_shape_loss_warns_length_mismatch():
    # Simulate mismatch by passing a target, then calling the inner fn
    target = np.zeros((5, 3), dtype=np.float32)
    loss_fn = make_shape_loss(target)
    # Fake inputs to hit validation quickly
    inputs = None
    outputs = {"structure_module": {"final_atom_positions": np.zeros((4, 3, 3), dtype=np.float32)}}
    aux = {}
    with pytest.raises(ValueError):
        loss_fn(inputs, outputs, aux)



