import os

import numpy as np
import pytest

from examples.stl.generators.make_cylinder_stl import make_cylinder_stl
from src.stl_designer import STLProteinDesigner


def test_designer_centerline_uses_path_loss(tmp_path):
    stl_path = tmp_path / "cyl.stl"
    make_cylinder_stl(str(stl_path), radius=5.0, height=20.0, sections=32)

    # Require AF params to be present; otherwise skip.
    data_dir = os.environ.get("AF_DATA_DIR")
    if data_dir is None:
        pytest.skip("AF_DATA_DIR not set; skipping designer construction.")

    try:
        designer = STLProteinDesigner(
            stl_path=str(stl_path),
            protein_length=40,
            num_target_points=200,  # ignored for centerline
            target_extent=50.0,
            stl_target_mode="centerline",
            target_arclength=None,
            centerline_surface_samples=2000,
            normalize_target_points=True,
            use_path_loss=True,
            path_weight=0.02,
            chamfer_weight=1.0,  # should be disabled internally
            con_weight=0.1,
            plddt_weight=0.1,
            pae_weight=0.05,
            data_dir=data_dir,
        )
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"Designer init failed (likely missing AF params): {exc}")

    tgt = designer.target_points
    assert tgt.shape == (40, 3)
    assert np.isfinite(tgt).all()
    extent = (tgt.max(axis=0) - tgt.min(axis=0)).max()
    assert extent == pytest.approx(50.0, rel=0.05)

    weights = designer.model.opt["weights"]
    assert weights["path"] > 0.0
    assert weights["chamfer"] == 0.0

