# Design Notes: ColabDesign STL Extension

This document captures the key design decisions and technical details behind the ColabDesign STL extension.

## Architecture

### Integration Strategy: No Fork Required

**Key Decision:** Use ColabDesign's built-in `loss_callback` system rather than forking the codebase.

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

**Algorithm:**

1. **Surface Sampling:** Sample many points from STL surface (default: 10,000)
2. **PCA Canonicalization:** 
   - Center points
   - Compute PCA to find principal axes
   - Rotate into PCA frame (columns = principal directions)
   - Use extent-based axis selection (longest dimension) for stability
3. **Binning:**
   - Bin points along principal axis (default: `4 * num_points` bins)
   - Compute centroid per bin
   - Handle empty bins via linear interpolation
4. **Smoothing:** Apply moving average (default window: 5)
5. **Resampling:** Uniformly resample by arclength to exactly `num_points`
6. **Optional Scaling:** Scale to target arclength if specified

**Key Parameters:**
- `surface_samples`: More samples = more accurate centerline (but slower)
- `bins`: More bins = finer detail (default: `4 * num_points`)
- `smooth_window`: Larger window = smoother centerline (default: 5)

**Limitations:**
- Assumes roughly cylindrical cross-sections
- May fail on complex geometries (branches, sharp corners)
- PCA can be ambiguous for symmetric shapes (handled via extent-based selection)

**Code:** `src/stl_processing.py::stl_to_centerline_points()`

## Kabsch Alignment

### Why Alignment?

**Problem:** Protein structures can be rotated/translated arbitrarily, but we want rotation-invariant loss.

**Solution:** Apply optimal rotation (Kabsch algorithm) before computing loss.

### Implementation Details

**Key Fix:** Correct rotation application
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

**Key Detail:** Cα coordinates come from AlphaFold's structure module output.

```python
positions = outputs["structure_module"]["final_atom_positions"]
ca = positions[:, 1, :]  # Index 1 = Cα (atom order: N=0, CA=1, C=2, O=3, ...)
```

**Shape:** `[L, 3]` where `L` = protein length

**Important:** Coordinates are in AlphaFold's internal frame. We center them before alignment/loss computation.

## Loss Callback Pattern

### How It Works

ColabDesign calls loss callbacks during the optimization loop:

```python
# From colabdesign/af/model.py (simplified)
for fn in self._callbacks["model"]["loss"]:
    fn_args = {"inputs": inputs, "outputs": outputs, "opt": opt, "aux": aux, ...}
    sub_args = {k: fn_args.get(k, None) for k in signature(fn).parameters}
    aux["losses"].update(fn(**sub_args))
```

**Our Callback:**
- Receives: `inputs`, `outputs`, `aux`
- Extracts Cα coordinates from `outputs`
- Computes loss (with Kabsch alignment)
- Returns: `{"path": loss_value}` (or `{"chamfer": loss_value}`)

**Weight Setting:**
```python
af_model.opt["weights"]["path"] = 0.02  # Our custom loss
af_model.opt["weights"]["plddt"] = 2.0  # Built-in losses
af_model.opt["weights"]["con"] = 0.2
af_model.opt["weights"]["pae"] = 0.2
```

Total loss = weighted sum of all losses.

## Key Parameters

### Target Extent (`target_extent`)

**Purpose:** Controls the scale of the target shape in Ångströms.

**Guidance:**
- Alpha helix: ~1.5Å rise per residue
- 80 residues → ~120Å axial length
- Typical range: 30-150Å

**Tuning:**
- Too small: Protein can't fit the shape
- Too large: Protein becomes too extended
- Optimal: Match expected protein length (e.g., 80 residues → 30-50Å for compact shapes)

### Path Weight (`path_weight`)

**Purpose:** Controls strength of shape constraint vs. protein stability.

**Default:** 0.02

**Tuning:**
- Too low: Shape doesn't match (stability dominates)
- Too high: Low pLDDT (shape constraint forces unphysical structures)
- Sweet spot: Balance shape matching with pLDDT > 50

### Centerline Surface Samples (`centerline_surface_samples`)

**Purpose:** Number of surface points used for centerline extraction.

**Default:** 10,000

**Tuning:**
- More samples: More accurate centerline (but slower)
- Fewer samples: Faster but may miss details
- Recommended: 10,000-12,000 for typical shapes

## Failure Modes & Mitigations

### Rank-Deficient Kabsch (Collinear Targets)

**Problem:** Lines/collinear points cause SVD to be ill-conditioned → NaNs.

**Mitigation:** Regularize `H` matrix with `1e-4 * I` before SVD.

**Code:** `src/losses.py::_kabsch_align()` (line 84)

### Scale Mismatch

**Problem:** `target_extent` doesn't match protein length → poor convergence.

**Mitigation:** 
- Use `target_extent ≈ protein_length * 1.5Å` (helix approximation)
- Or use `target_arclength` to set exact arclength

### Centerline Extraction Failures

**Problem:** Complex geometries (branches, sharp corners) break PCA + binning.

**Mitigation:**
- Stick to tube-like shapes (cylinders, helices, smooth curves)
- Increase `surface_samples` for better coverage
- Consider skeletonization algorithms for complex shapes (future work)

### Low pLDDT

**Problem:** Shape constraint too strong → unphysical structures.

**Mitigation:**
- Reduce `path_weight` (try 0.01)
- Increase `plddt_weight` (try 2.0-5.0)
- Try simpler shape or longer protein

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

**Alternative:** Skeletonization algorithms (e.g., medial axis)
- ✅ More robust for complex shapes
- ❌ More complex, slower
- ❌ Additional dependencies

### Kabsch Regularization

**Chosen:** Small identity regularization (`1e-4 * I`)

**Trade-off:**
- ✅ Prevents NaNs on collinear targets
- ✅ Minimal impact on optimization (regularization is tiny)
- ❌ Slightly modifies loss landscape (negligible in practice)

## Future Improvements

1. **Surface Mode:** Use Chamfer loss for non-tube shapes
2. **Better Centerline:** Skeletonization algorithms for complex geometries
3. **Multi-Chain:** Extend to multi-chain protein design
4. **Auto-Tuning:** Automatic `target_extent` selection based on protein length
5. **Interactive Visualization:** Real-time overlay plots during optimization

