# ColabDesign STL Extension

Design proteins to approximate arbitrary STL shapes by integrating custom shape losses into ColabDesign. This repo adds STL → point-cloud preprocessing (Stage 1) and a differentiable Chamfer loss (Stage 2) for downstream integration.

## Status
- Stage 1 (STL processing) ✅
- Stage 2 (Chamfer loss in JAX) ✅
- Next: Stage 3 (ColabDesign integration)

## Requirements
- Python 3.13
- pip + venv (activate before running)
- JAX 0.4.34 (CPU wheel by default; use Colab GPU wheel if needed)

## Install
```pwsh
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Usage (Stage 1)
Generate a helix STL and sample a centered, scaled point cloud:
```pwsh
# From repo root, venv active
python examples/make_helix_stl.py --out examples/helix.stl
python -m examples.sample_points --plot   # add --plot to visualize
```
Expected logs:
- `points shape: (1000, 3)`
- `mean (should be ~0): ...`
- `bbox: [...] (longest=100.00, target=100.0)`

Notes:
- `target_extent` sets the longest dimension in Å; align with planned protein length (e.g., 100 Å for ~100 residues).
- Use `--seed` to make sampling deterministic.
- If running `python examples/sample_points.py` directly, ensure project root is on `PYTHONPATH` or add `sys.path` tweak.

## Tests (Stage 2)
Run Chamfer-loss unit tests (gradients, correctness, validation):
```pwsh
python -m pytest tests/test_losses.py
```
Default Chamfer uses squared distances for speed; pass `use_sqrt=True` for interpretable Å units at slight cost.

Quick demo (Stage 2):
```pwsh
python examples/chamfer_demo.py
```
Shows squared vs sqrt losses and gradient values.

## Files
- `src/stl_processing.py` — STL loader (`stl_to_points`) + `plot_point_cloud`.
- `src/losses.py` — JAX Chamfer distance with optional sqrt mode.
- `tests/test_losses.py` — Unit tests for Chamfer distance and gradients.
- `examples/chamfer_demo.py` — Minimal Chamfer usage + gradient printout.
- `examples/make_helix_stl.py` — procedural helix STL generator.
- `examples/sample_points.py` — sampling demo with simple assertions.
- `requirements.txt` — numpy, trimesh, matplotlib, jax, pytest.

## Next steps (roadmap)
- Stage 3: Wire into ColabDesign via `loss_callback` factory and add an end-to-end design script/notebook.

