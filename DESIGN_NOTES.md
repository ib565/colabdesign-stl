# Design Notes: ColabDesign STL Extension

This document captures the key design decisions and technical details behind the ColabDesign STL extension.

## Architecture

### Integration Strategy

Use ColabDesign's built-in `loss_callback` system.

```python
# The integration pattern
af_model = mk_afdesign_model(protocol="hallucination", loss_callback=my_loss_fn)
af_model.opt["weights"]["my_loss_name"] = 1.0
```

This allows injecting custom losses without modifying ColabDesign core code. Reference: `colabdesign/af/examples/hallucination_custom_loss.ipynb`

### Pipeline Overview

```
STL File → Point Cloud → Centerline Extraction → Target Points
                                                      ↓
                                    Loss Callback (per-index MSE + Kabsch)
                                                      ↓
                                    ColabDesign Optimization Loop
                                                      ↓
                                    Protein Sequence + Structure
```

**Our Contribution:**
- STL → point cloud preprocessing (`src/stl_processing.py`)
- Centerline extraction for tube-like shapes (`stl_to_centerline_points`)
- Custom loss callbacks (`src/losses.py`)
- High-level orchestrator (`src/stl_designer.py`)

**ColabDesign (Unchanged):**
- AlphaFold forward pass
- Built-in losses (pLDDT, PAE, contact)
- Optimization loop (Gumbel → soft → temp → hard)
- Sequence updates

## Loss Functions

### Per-Index Path Loss (Primary)

**Design Decision:** Use per-index MSE loss for ordered targets (centerlines, lines, helices) rather than Chamfer distance.

**Why:**
- Cleaner gradients (1:1 correspondence between residues and target points)
- Avoids "clumping" failure modes where multiple residues map to the same target point
- More interpretable: loss directly measures deviation per residue

**Implementation:**
```python
def make_path_loss(target_points: np.ndarray) -> Callable:
    """Per-index MSE loss with Kabsch alignment."""
    target_centered = target - target.mean(axis=0)
    
    def path_loss(inputs, outputs, aux):
        ca = outputs["structure_module"]["final_atom_positions"][:, 1, :]  # CA index = 1
        ca_centered = ca - ca.mean(axis=0)
        ca_aligned = _kabsch_align(ca_centered, target_centered)
        loss = jnp.mean(jnp.sum((ca_aligned - target_centered) ** 2, axis=-1))
        return {"path": loss}
    
    return path_loss
```

**Requirements:**
- `len(target_points) == protein_length` (1:1 correspondence)
- Target points must be ordered (e.g., along a path)

### Chamfer Loss (Legacy)

**Status:** Implemented but not used in current workflow. Kept for potential surface-mode support.

**Why Chamfer:**
- Works with unordered point clouds
- Rotation/translation invariant (with Kabsch alignment)
- Suitable for surface sampling where correspondence is ambiguous

**Why Not Used:**
- Per-index path loss performs better for ordered targets
- Chamfer can cause clumping (multiple residues → same target point)

## Centerline Extraction

### Method: PCA + Binning + Smoothing

**Design Decision:** Use heuristic centerline extraction rather than skeletonization algorithms.

**Why:**
- Simpler, faster, and sufficient for tube-like shapes
- Works well for cylinders, helices, sine tubes
- No external dependencies beyond trimesh

**Algorithm:** Surface sampling → PCA canonicalization (with extent-based axis selection for stability) → binning along principal axis → smoothing → arclength resampling → optional scaling.


**Key Parameters:**
- `surface_samples`: Default 10,000 (10,000-12,000 recommended for typical shapes)
- `bins`: Default `4 * num_points` (automatic)
- `smooth_window`: Default 5

**Limitations:**
- Assumes roughly cylindrical cross-sections
- May fail on complex geometries (branches, sharp corners)
- PCA can be ambiguous for symmetric shapes

**Code:** `src/stl_processing.py::stl_to_centerline_points()`

## Kabsch Alignment

Apply optimal rotation (Kabsch algorithm) before computing loss for rotation-invariance.

**Critical Bug Fix:** Initial implementation had incorrect rotation application causing alignment failures:
```python
# After SVD: R = Vt.T @ U.T
# Apply rotation on the RIGHT: pred_aligned = pred_centered @ R.T
# NOT: pred_aligned = R @ pred_centered  (wrong!)
```

**Rank-Deficient Handling:**
- Collinear targets (lines) cause SVD to be ill-conditioned → NaNs
- **Fix:** Regularize `H = pred.T @ target` with small identity: `H + 1e-4 * I`
- Prevents NaN gradients while preserving optimization behavior

**Code:** `src/losses.py::_kabsch_align()`

## Cα Coordinate Extraction

Cα coordinates from AlphaFold structure module: `outputs["structure_module"]["final_atom_positions"][:, 1, :]` (atom order: N=0, CA=1, C=2, ...). Coordinates are centered before alignment/loss computation.

## Loss Callback Pattern

ColabDesign calls loss callbacks during optimization. Our callback extracts Cα from `outputs`, applies Kabsch alignment, and returns `{"path": loss_value}`. Weights are set via `af_model.opt["weights"]["path"] = 0.02`.

## Key Parameters

### Target Extent (`target_extent`)

Controls scale in Ångströms. Guidance: ~1.5Å per residue (helix), so 80 residues → ~120Å. Typical range: 50-200Å.

**Limitation:** `target_extent` must be manually tuned per STL shape. Different shapes require different values (e.g., cylinder: 100Å, sine_tube: 120Å) and there's no automatic way to determine optimal scaling. This makes the workflow less automated and requires trial-and-error for new shapes.

### Path Weight (`path_weight`)

Default: 0.02. Controls shape constraint vs. stability trade-off. Lower values favor stability (higher pLDDT), higher values favor shape matching (lower path loss).

### Centerline Surface Samples (`centerline_surface_samples`)

Default: 10,000.

## Failure Modes & Mitigations

### Rank-Deficient Kabsch (Collinear Targets)

**Problem:** Lines/collinear points cause SVD to be ill-conditioned → NaNs.

**Mitigation:** Regularize `H` matrix with `1e-4 * I` before SVD.

**Code:** `src/losses.py::_kabsch_align()` (line 84)

### Scale Mismatch

**Problem:** `target_extent` doesn't match protein length → poor convergence.

**Mitigation:** Use `target_extent ≈ protein_length * 1.5Å` (helix approximation) or `target_arclength` for exact control.

### Centerline Extraction Failures

**Problem:** Complex geometries (branches, sharp corners) break PCA + binning.

**Mitigation:** Stick to tube-like shapes. For complex shapes, consider skeletonization algorithms (future work).

### Low pLDDT

**Problem:** Shape constraint too strong → unphysical structures.

**Mitigation:** Reduce `path_weight` or increase `plddt_weight` to rebalance the trade-off.

## Code Organization

```
src/
├── stl_processing.py    # STL → points, centerline extraction, normalization
├── losses.py            # JAX loss functions, Kabsch alignment
└── stl_designer.py      # High-level orchestrator (STLProteinDesigner)

Key Functions:
- stl_to_centerline_points()  # Centerline extraction
- make_path_loss()             # Per-index MSE loss factory
- make_shape_loss()            # Chamfer loss factory (legacy)
- _kabsch_align()              # Rotation alignment
- STLProteinDesigner           # Main API
```

## Design Trade-offs

### Per-Index Path Loss vs. Chamfer

**Chosen:** Per-index path loss for ordered targets

**Trade-off:**
- ✅ Cleaner gradients, no clumping
- ❌ Requires 1:1 correspondence (`len(target) == protein_length`)
- ❌ Only works for ordered paths

**Alternative:** Chamfer loss (implemented but not used)
- ✅ Works with unordered point clouds
- ❌ Can cause clumping
- ❌ Less interpretable gradients

### Heuristic Centerline vs. Skeletonization

**Chosen:** PCA + binning heuristic

**Trade-off:**
- ✅ Simple, fast, no external dependencies
- ✅ Works well for tube-like shapes
- ❌ May fail on complex geometries
- ❌ Assumes roughly cylindrical cross-sections

**Alternative:** Skeletonization algorithm
- ✅ More robust for complex shapes
- ❌ More complex, slower
- ❌ Additional dependencies

### Kabsch Regularization

**Chosen:** Small identity regularization (`1e-4 * I`)

**Trade-off:**
- ✅ Prevents NaNs on collinear targets
- ✅ Minimal impact on optimization (regularization is tiny)

## Future Improvements

1. **Surface Mode:** Use Chamfer loss for non-tube shapes
2. **Better Centerline:** Skeletonization algorithms for complex geometries
3. **Auto-Tuning:** Automatic `target_extent` selection based on protein length
