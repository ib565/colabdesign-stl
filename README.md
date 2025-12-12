# ColabDesign STL Extension

Design proteins to approximate arbitrary STL shapes by integrating custom shape losses into ColabDesign. This repo adds STL → point-cloud preprocessing, a Kabsch-aligned Chamfer loss, and a small API/CLI for end-to-end runs.

## Status
- STL processing ✅
- Chamfer loss (JAX) ✅
- Per-index path loss ✅ (ordered targets; uses Kabsch)
- Kabsch fixes ✅
  - correct rotation application (pred @ R.T)
  - rank-deficient targets (collinear lines) regularized to avoid NaNs
- STLProteinDesigner + CLI ✅
- Alignment + Kabsch sanity checks ✅

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

## Tests
Run unit tests (Chamfer + path/Kabsch + alignment sanity):
```pwsh
python -m pytest
```
Default Chamfer uses squared distances for speed; pass `use_sqrt=True` for interpretable Å units.

Alignment sanity checks (Kabsch + centering):
```pwsh
python examples/alignment_sanity.py
```
Expected: Chamfer and RMSD ~2e-4 or below (float32 noise).

Quick demo (Stage 2):
```pwsh
python examples/chamfer_demo.py
```
Shows squared vs sqrt losses and gradient values.

## Stage 3a: Minimal ColabDesign integration
Smoke-test with a simple line target (uses per-index path loss):
```pwsh
# Ensure colabdesign is installed / on PYTHONPATH. Run with GPU for speed.
python examples/minimal_chamfer_hallucination.py \
  --soft-iters 20 --temp-iters 10 --hard-iters 5 \
  --length 50 --verbose 1
```
What to expect:
- JAX version/devices printed
- Model params loaded (5 models)
- Stage 1/2/3 banners with per-step logs showing `chamfer` loss
- Final log entry with `chamfer` key (decreases over iterations)
- Sequence printed (first 60 aa)

**Note:** Uses `design_3stage()` with separate soft/temp/hard iteration counts for better optimization. First JIT compile takes 1-3 minutes on CPU.

### Important: collinear targets (lines)
- Plain Kabsch SVD is ill-conditioned on rank-1 targets → NaNs.
- We regularize `H = pred.T @ target` with a small identity to keep SVD finite.
- Tests cover this (`test_kabsch_collinear_target_no_nan`); line targets now run without crashing.

### ColabDesign install & Alphafold weights
- Install ColabDesign into the active venv (sibling clone):
  ```pwsh
  .\venv\Scripts\Activate
  pip install -e ..\ColabDesign
  ```
- Download Alphafold params (≈3.5 GB) and extract so `.npz` live under `params/`:
  ```pwsh
  cd ..\ColabDesign
  mkdir params
  curl -L -o alphafold_params_2022-12-06.tar https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
  tar -xf alphafold_params_2022-12-06.tar -C params
  ```
- Point ColabDesign at the weights when running from `colabdesign-stl` (either env var or `data_dir` argument):
  ```pwsh
  # from colabdesign-stl root, same shell
  $env:AF_DATA_DIR = "..\ColabDesign"
  python examples\minimal_chamfer_hallucination.py --iters 10 --length 50
  ```
- If weights are missing, you’ll see warnings like `'model_*_ptm' not found` and `AssertionError: no model params defined`.

## Stage 3b: Full STL integration (STLProteinDesigner)
Run end-to-end from an STL file, saving sequence, PDB, and optional overlay plot:
```pwsh
# From repo root, venv active
python examples/design_from_stl.py --stl examples/helix.stl `
  --length 100 --num-points 1000 --target-extent 100 `
  --soft-iters 200 --temp-iters 100 --hard-iters 20 `
  --chamfer-weight 1.0 --plddt 0.1 --pae 0.05 `
  --sample-seed 0 --run-seed 0 --data-dir ..\ColabDesign `
  --out-dir outputs\helix --plot
```
Outputs:
- `outputs/helix/sequence.txt`
- `outputs/helix/structure.pdb`
- `outputs/helix/overlay.png` (target points vs predicted Cα)

Key flags:
- `--sample-seed`: controls STL surface sampling (default 0 for reproducibility; set to -1 for stochastic).
- `--run-seed`: controls the Gumbel restart in ColabDesign.
- `--use-sqrt`: switch to sqrt Chamfer if you want Å-scale values in logs (slightly slower).
- `--data-dir`: path to AlphaFold params (falls back to `AF_DATA_DIR` or `../ColabDesign` if present).

Programmatic use:
```python
from src import STLProteinDesigner

designer = STLProteinDesigner(
    stl_path="examples/helix.stl",
    protein_length=100,
    num_target_points=1000,
    target_extent=100.0,
    sample_seed=0,
    chamfer_weight=1.0,
    plddt_weight=0.1,
    pae_weight=0.05,
    use_sqrt=False,
    data_dir="../ColabDesign",
)
seq = designer.design(soft_iters=200, temp_iters=100, hard_iters=20, run_seed=0)
pdb_str = designer.get_structure()
metrics = designer.get_metrics()
designer.plot_overlay(save_path="overlay.png", show=False)
print(seq[:60], metrics)
```

Notes:
- First JIT compile still takes 1–3 minutes on CPU; GPU is recommended.
- Chamfer is squared by default for speed; use `--use-sqrt` for Å units.
- For ordered targets (line/helix/centerline), prefer per-index path loss (`use_path_loss=True` in `STLProteinDesigner`), and set `len(target_points)==protein_length`.
- `design_3stage` is used for better convergence on hallucination.
- Kabsch alignment is applied after centering both target and predicted Cα; scale is not normalized beyond your `target_extent`.

## Files
- `src/stl_processing.py` — STL loader (`stl_to_points`) + `plot_point_cloud`.
- `src/losses.py` — JAX Chamfer distance (`chamfer_distance`), `make_shape_loss`, Kabsch alignment.
- `src/stl_designer.py` — `STLProteinDesigner` orchestrator, overlay plotting.
- `tests/test_losses.py` — Chamfer unit tests and gradients.
- `tests/test_alignment_sanity.py` — Kabsch + centering sanity checks.
- `examples/alignment_sanity.py` — Quick CLI for alignment sanity.
- `examples/chamfer_demo.py` — Minimal Chamfer usage + gradient printout.
- `examples/make_helix_stl.py` — procedural helix STL generator.
- `examples/sample_points.py` — sampling demo with assertions.
- `examples/minimal_chamfer_hallucination.py` — Stage 3a smoke test.
- `requirements.txt` — numpy, trimesh, matplotlib, jax, pytest.

## Next steps (roadmap)
- Stage 3c: Verification (Cα coordinate matching, centering checks, multi-seed testing)
- Stage 4: Validation with real STL shapes (helix, ellipsoid, etc.)

