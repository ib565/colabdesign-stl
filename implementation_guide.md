# ColabDesign STL Extension: Implementation Guide

## Project Goal

Extend ColabDesign AF to accept an STL file as input and design a protein sequence whose predicted structure approximates the STL shape.

---

## Key Discovery: No Fork Required

ColabDesign has a built-in `loss_callback` system that allows injecting custom losses without modifying core code. This is the integration path we will use.

```python
# The pattern
af_model = mk_afdesign_model(protocol="hallucination", loss_callback=my_loss_fn)
af_model.opt["weights"]["my_loss_name"] = 1.0
```

Reference: `colabdesign/af/examples/hallucination_custom_loss.ipynb`

---

## Architecture Overview

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUR CONTRIBUTION                               │
│                                                                             │
│  ┌──────────┐    ┌─────────────────┐    ┌─────────────────────────────────┐ │
│  │  STL     │──▶│  Point Cloud    │───▶│  shape_loss callback            │ │
│  │  File    │    │  (N, 3) array   │    │  returns {"chamfer": value}     │ │
│  └──────────┘    └─────────────────┘    └─────────────────────────────────┘ │
│                                                       │                     │
└───────────────────────────────────────────────────────┼─────────────────────┘
                                                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COLABDESIGN (UNCHANGED)                             │
│                                                                             │
│  mk_afdesign_model(protocol="hallucination", loss_callback=shape_loss)      │
│                              │                                              │
│                              ▼                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Optimization Loop                                                     │ │
│  │                                                                        │ │
│  │  1. Forward pass through AlphaFold                                     │ │
│  │  2. Compute built-in losses (plddt, pae, con)                          │ │
│  │  3. Call loss_callback → adds "chamfer" to aux["losses"]               │ │
│  │  4. Weight all losses via opt["weights"]                               │ │
│  │  5. Backprop, update sequence                                          │ │
│  │  6. Repeat                                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                              │                                              │
│                              ▼                                              │
│                      Final Sequence + Structure                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Technical Reference

### How to Access Cα Coordinates

```python
ca_coords = outputs["structure_module"]["final_atom_positions"][:, 1, :]
# Shape: [L, 3] where L is protein length
# Index 1 = Cα (atom order is N, CA, C, CB, O, ...)
```

### How Loss Callbacks Work

From `colabdesign/af/model.py`:

```python
# Lines 228-234
for c in ["loss","post"]:
    for fn in self._callbacks["model"][c]:
        fn_args = {"inputs":inputs, "outputs":outputs, "opt":opt,
                   "aux":aux, "seq":seq, "key":key(), "params":params}
        sub_args = {k:fn_args.get(k,None) for k in signature(fn).parameters}
        if c == "loss": aux["losses"].update(fn(**sub_args))
```

Your callback receives these possible arguments: `inputs`, `outputs`, `opt`, `aux`, `seq`, `key`, `params`. Use only what you need.

### How Weights Work

```python
af_model.opt["weights"]["chamfer"] = 1.0  # Your custom loss
af_model.opt["weights"]["plddt"] = 0.1    # Built-in losses
af_model.opt["weights"]["con"] = 1.0
```

Total loss = Σ (weight × loss_value) for all losses.

---

## Success Criteria

### MVP (Must Have)

- [ ] Load a helix STL file
- [ ] Convert to scaled/centered point cloud
- [ ] Chamfer loss implemented and working
- [ ] Integration with ColabDesign via loss_callback
- [ ] Produces a sequence
- [ ] Predicted structure visually resembles helix
- [ ] pLDDT > 50
- [ ] Basic README with usage instructions

### Post-MVP (Nice to Have)

- [ ] Multiple test shapes with results
- [ ] Parameter tuning documentation
- [ ] Polished API with factory functions
- [ ] Colab notebook demo

---

## Stage 0: Environment Setup and Verification

### Objective

Verify ColabDesign works and understand the callback system hands-on.

### Tasks

1. **Set up environment**
   - Use Google Colab (recommended) or local with GPU
   - Install ColabDesign: `pip install git+https://github.com/sokrypton/ColabDesign.git`
   - Install trimesh: `pip install trimesh`

2. **Run existing hallucination example**
   - Open any hallucination notebook from ColabDesign examples
   - Run end-to-end
   - Note: how long it takes, what outputs look like

3. **Run the custom loss example**
   - Open `af/examples/hallucination_custom_loss.ipynb`
   - Study how the RG loss is added
   - Verify you understand the pattern

4. **Test the callback system manually**
   - Add a dummy loss that just returns a constant
   - Verify it appears in the logs
   - Verify changing the weight changes the total loss

### Checkpoint

- [ ] ColabDesign runs without errors
- [ ] Can run hallucination end-to-end
- [ ] Understand loss_callback pattern from example
- [ ] Verified callback system works with dummy loss

### Time Estimate: 0.5 days

---

## Stage 1: STL Processing Module

### Objective

Build a module that converts STL files to protein-appropriate point clouds.

### Tasks

1. **Implement STL loader** ✅ `src/stl_processing.py`

   ```python
   import os
   import numpy as np
   import trimesh
   from typing import Optional

   def stl_to_points(
       stl_path: str,
       num_points: int = 1000,
       target_extent: float = 100.0,
       center: bool = True,
       seed: Optional[int] = None,
   ) -> np.ndarray:
       """Load STL, sample points, center, scale."""
       if not os.path.exists(stl_path):
           raise FileNotFoundError(f"STL file not found: {stl_path}")
       mesh = trimesh.load_mesh(stl_path, force="mesh")
       if mesh.is_empty or len(mesh.vertices) == 0:
           raise ValueError(f"STL file is empty or invalid: {stl_path}")
       if seed is not None:
           np.random.seed(seed)
       points, _ = trimesh.sample.sample_surface(mesh, num_points)
       if center:
           points = points - points.mean(axis=0)
       extent = (points.max(axis=0) - points.min(axis=0)).max()
       if extent <= 0:
           raise ValueError("Mesh has zero extent; cannot scale.")
       points = points * (target_extent / extent)
       return points.astype(np.float32)
   ```

2. **Handle edge cases** ✅
   - Missing file → `FileNotFoundError`
   - Empty/invalid mesh → `ValueError`
   - Zero extent → `ValueError`

3. **Create visualization helper** ✅ `plot_point_cloud(...)` in `stl_processing.py`
   - Matplotlib 3D scatter for quick inspection

### Critical Note: Units and Scale

```text
STL files: typically millimeters or arbitrary units
Proteins: measured in Ångströms (Å)

Scale guidance:
- Alpha helix: ~1.5Å rise per residue
- 100 residues → ~150Å axial length
- Target extent should be 50-150Å for typical designs
- Your STL's longest dimension becomes target_extent
```

### Checkpoint

- [x] Can load any valid STL file
- [x] Point cloud visually matches STL shape (helix STL generator + sample)
- [x] Output is centered at origin
- [x] Output scale is in Ångströms (50-150Å range)
- [x] Works with at least 2 different STL files (generated helix; can swap params/seeds)
- [x] Visualization confirms correctness (plot helper)

### Deliverables

- `src/stl_processing.py` (loader + plot helper)
- `examples/make_helix_stl.py` (helix STL generator)
- `examples/sample_points.py` (sampling demo + assertions)

### How to verify (Stage 1)

```pwsh
# from repo root, venv active
python examples/make_helix_stl.py --out examples/helix.stl
python -m examples.sample_points --plot  # or python examples/sample_points.py --plot (after adding project root to PYTHONPATH)
```

Expected:
- `points shape: (NUM_POINTS, 3)`
- mean ~0 on all axes
- bbox longest edge == `target_extent` (e.g., 100.00)

### Time Estimate: 0.5 days

---

## Stage 2: Loss Function Implementation

### Objective

Implement a differentiable Chamfer distance loss in JAX.

### Tasks

1. **Implement Chamfer distance**

   ```python
   import jax.numpy as jnp

   def chamfer_distance(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
       """
       Compute Chamfer distance between two point clouds.
       
       Args:
           pred: Predicted points, shape (L, 3)
           target: Target points, shape (N, 3)
           
       Returns:
           Scalar loss value
       """
       # Pairwise squared distances: (L, N)
       diff = pred[:, None, :] - target[None, :, :]
       sq_dist = jnp.sum(diff ** 2, axis=-1)
       
       # Chamfer: mean of min distances in both directions
       loss_pred_to_target = jnp.mean(jnp.min(sq_dist, axis=1))
       loss_target_to_pred = jnp.mean(jnp.min(sq_dist, axis=0))
       
       return loss_pred_to_target + loss_target_to_pred
   ```

2. **Test differentiability**

   ```python
   import jax

   # Must not error
   grad_fn = jax.grad(lambda p: chamfer_distance(p, target))
   gradients = grad_fn(pred)
   ```

3. **Test expected behavior**
   - Loss = 0 when pred == target
   - Loss > 0 when pred != target
   - Loss decreases as points move closer

### Important Notes

**Squared distances:** The implementation above returns squared Ångströms for efficiency. This is fine for optimization but means:
- Loss of 100 ≈ 10Å average error
- For interpretable logging, optionally use sqrt version:

```python
# More interpretable (actual Ångströms) but slightly slower
loss_pred_to_target = jnp.mean(jnp.sqrt(jnp.min(sq_dist, axis=1) + 1e-8))
loss_target_to_pred = jnp.mean(jnp.sqrt(jnp.min(sq_dist, axis=0) + 1e-8))
```

**Memory:** Chamfer distance is O(L × N) in memory. For L=100, N=1000, this is 100K floats (~400KB) — fine. For very large clouds, consider subsampling.

### Checkpoint

- [ ] Loss function runs without errors
- [ ] `jax.grad()` works on it
- [ ] Loss = 0 for identical point clouds
- [ ] Loss > 0 for different point clouds
- [ ] Loss decreases as points approach target (gradient check)

### Deliverables

- `src/losses.py`
- Test script demonstrating behavior

### Time Estimate: 0.25 days

---

## Stage 3: ColabDesign Integration

### Objective

Wire the STL processing and loss function into ColabDesign's hallucination protocol.

**Weight defaults:** For hallucination, only `con` (and `i_con` for multichain) is nonzero by default. `plddt`, `pae`, `exp_res`, and `helix` start at 0, so set them explicitly alongside your custom `chamfer` term.

### Stage 3a: Minimal Integration (Hardcoded Test)

#### Tasks

1. **Create the loss callback using factory pattern**

   ```python
   import jax.numpy as jnp

   def make_shape_loss(target_points: np.ndarray):
       """
       Factory function that creates a shape loss callback.
       
       Args:
           target_points: Target point cloud, shape (N, 3)
           
       Returns:
           Loss callback function compatible with ColabDesign
       """
       target = jnp.asarray(target_points, dtype=jnp.float32)
       target_centered = target - target.mean(axis=0)
       
       def shape_loss(inputs, outputs, aux):
           # Extract Cα coordinates
           # Atom order: N=0, CA=1, C=2, O=3, CB=4, ...
           ca = outputs["structure_module"]["final_atom_positions"][:, 1, :]
           
           # Center the predicted coordinates
           ca_centered = ca - ca.mean(axis=0)
           
           # Compute Chamfer distance
           loss = chamfer_distance(ca_centered, target_centered)
           
           return {"chamfer": loss}
       
       return shape_loss
   ```

2. **Test with hardcoded simple target**

   ```python
   # Simple test: line of points
   test_target = np.linspace([0, 0, 0], [100, 0, 0], 50).astype(np.float32)
   
   from colabdesign import mk_afdesign_model
   
   loss_fn = make_shape_loss(test_target)
   af_model = mk_afdesign_model(protocol="hallucination", loss_callback=loss_fn)
   # Default AF hallucination weights set only "con"; plddt/pae/exp_res/helix start at 0.
   af_model.opt["weights"].update({
       "chamfer": 1.0,
       "plddt": 0.1,
       "pae": 0.05,
       "exp_res": 0.0
   })
   
   af_model.prep_inputs(length=50)
   af_model.restart(mode="gumbel", seed=0)
   
   # Run just a few iterations to verify it works
   af_model.design(10)
   
   # Check that chamfer loss is being tracked
   print(af_model._tmp["log"][-1])
   ```

3. **Verify the loss is being optimized**
   - Chamfer loss should appear in logs
   - Loss should decrease over iterations

#### Checkpoint

- [ ] No crashes during optimization
- [ ] "chamfer" appears in loss logs
- [ ] Chamfer loss decreases over iterations
- [ ] Can extract final sequence with `af_model.get_seqs()`

#### Time Estimate: 0.5 days

---

### Stage 3b: Full STL Integration

#### Tasks

1. **Create main designer class**

   ```python
   from colabdesign import mk_afdesign_model
   import numpy as np

   class STLProteinDesigner:
       """Design proteins to match STL shapes."""
       
       def __init__(
           self,
           stl_path: str,
           protein_length: int = 100,
           num_target_points: int = 1000,
           target_extent: float = 100.0,
           chamfer_weight: float = 1.0
       ):
           # Load and process STL
           self.target_points = stl_to_points(
               stl_path,
               num_points=num_target_points,
               target_extent=target_extent
           )
           
           # Create model with custom loss
           loss_fn = make_shape_loss(self.target_points)
           self.model = mk_afdesign_model(
               protocol="hallucination",
               loss_callback=loss_fn
           )
           # Set weights explicitly (defaults are zero for plddt/pae/exp_res/helix)
           self.model.opt["weights"].update({
               "chamfer": chamfer_weight,
               "con": 1.0,
               "i_con": self.model.opt["weights"].get("i_con", 0.0),
               "plddt": 0.1,
               "pae": 0.05,
               "exp_res": 0.0
           })
           
           # Prepare inputs
           self.model.prep_inputs(length=protein_length)
       
       def design(
           self,
           soft_iters: int = 200,
           temp_iters: int = 100,
           hard_iters: int = 20,
           seed: int = 0,
           verbose: int = 10
       ) -> str:
           """Run design optimization and return best sequence."""
           self.model.restart(mode="gumbel", seed=seed)
           self.model.design_3stage(
               soft_iters=soft_iters,
               temp_iters=temp_iters,
               hard_iters=hard_iters,
               save_best=True,
               verbose=verbose
           )
           return self.model.get_seqs()[0]
       
       def get_structure(self) -> str:
           """Get PDB string of designed structure."""
           return self.model.save_pdb()
       
       def get_metrics(self) -> dict:
           """Get final metrics."""
           log = self.model._tmp.get("best", {}).get("aux", {}).get("log", self.model.aux["log"])
           return {
               "chamfer": float(log.get("chamfer", float("nan"))),
               "plddt": float(log.get("plddt", float("nan"))),
           }
   ```

2. **Test with real STL file**
   - Use a helix STL
   - Run full optimization
   - Extract sequence and structure

3. **Add visualization**
   - Plot target points vs predicted Cα positions
   - Use matplotlib 3D scatter or py3Dmol

#### Checkpoint

- [ ] `STLProteinDesigner("helix.stl").design()` runs end-to-end
- [ ] Returns valid amino acid sequence
- [ ] Can save PDB file
- [ ] Visualization shows predicted structure vs target

#### Time Estimate: 0.5 days

---

### Stage 3c: Verification

#### Tasks

1. **Verify Cα extraction is correct**

   ```python
   # After running design:
   pdb_string = designer.get_structure()
   
   # Parse PDB and extract CA coordinates manually
   # Compare with what we used in the loss function
   # They should match
   ```

2. **Verify centering is working**
   - Both target and predicted should be centered
   - Visualize to confirm overlap makes sense

3. **Test different random seeds**
   - Run same STL with different seeds
   - Results should vary but all be reasonable

#### Checkpoint

- [ ] Cα coordinates match between loss function and saved PDB
- [ ] Centering visually confirmed
- [ ] Multiple seeds produce varied but valid results

#### Time Estimate: 0.25 days

---

## Stage 4: Validation and Testing

### Objective

Verify the system works and characterize its behavior.

### MVP Validation (Required)

#### Tasks

1. **Create or obtain helix STL**
   - Option A: Download from 3D model site
   - Option B: Generate programmatically:

   ```python
   import numpy as np
   from stl import mesh  # numpy-stl library
   
   def create_helix_stl(output_path, n_points=100, radius=10, pitch=5, height=100):
       """Create a simple helix tube STL."""
       # Generate helix centerline
       t = np.linspace(0, height/pitch * 2 * np.pi, n_points)
       x = radius * np.cos(t)
       y = radius * np.sin(t)
       z = pitch * t / (2 * np.pi)
       # ... create tube mesh around centerline
       # ... save as STL
   ```

2. **Run full design on helix**

   ```python
   designer = STLProteinDesigner(
       "helix.stl",
       protein_length=100,
       target_extent=100.0,
       chamfer_weight=1.0
   )
   sequence = designer.design()
   metrics = designer.get_metrics()
   print(f"Chamfer: {metrics['chamfer']:.2f}, pLDDT: {metrics['plddt']:.2f}")
   ```

3. **Evaluate results**
   - Final Chamfer loss (lower is better, <50 is good)
   - pLDDT score (higher is better, >50 is okay, >70 is good)
   - Visual inspection of overlay

4. **Create result visualization**
   - 3D plot showing target points (one color) and Cα positions (another color)
   - Save as PNG for README

#### Checkpoint

- [ ] Helix design completes successfully
- [ ] Final Chamfer loss < 100 (squared Å) or < 10Å if using sqrt version
- [ ] pLDDT > 50
- [ ] Visual inspection shows reasonable shape match
- [ ] Result saved as image

### Post-MVP Validation (If Time Permits)

#### Tasks

1. **Test additional shapes**

   | Shape | Expected Difficulty | Notes |
   |-------|---------------------|-------|
   | Ellipsoid | Easy | Globular, protein-friendly |
   | S-curve | Medium | Tests flexibility |
   | Torus | Hard | May struggle |

2. **Parameter sensitivity testing**
   - Different protein lengths for same shape
   - Different chamfer weights
   - Different number of target points

3. **Document findings**
   - Which shapes work?
   - What parameters work best?
   - Failure modes?

#### Checkpoint

- [ ] At least 2 additional shapes tested
- [ ] Results documented with metrics
- [ ] Parameter recommendations established

### Time Estimate: 1 day (MVP), +1 day (post-MVP)

---

## Stage 5: Documentation and Polish

### Objective

Make the project presentable and usable.

### Tasks

1. **Code cleanup**
   - Consistent style (run black/ruff)
   - Type hints on all public functions
   - Docstrings on all public functions
   - Remove debug prints

2. **Error handling**

   ```python
   def stl_to_points(stl_path, ...):
       if not os.path.exists(stl_path):
           raise FileNotFoundError(f"STL file not found: {stl_path}")
       
       mesh = trimesh.load_mesh(stl_path)
       if mesh.is_empty:
           raise ValueError(f"STL file is empty or invalid: {stl_path}")
   ```

3. **README.md**

   ```markdown
   # ColabDesign STL Extension
   
   Design proteins that fold into arbitrary 3D shapes.
   
   ## Installation
   
   pip install git+https://github.com/sokrypton/ColabDesign.git
   pip install trimesh numpy
   
   ## Quick Start
   
   from stl_designer import STLProteinDesigner
   
   designer = STLProteinDesigner("helix.stl", protein_length=100)
   sequence = designer.design()
   print(sequence)
   
   ## Example Result
   
   [Image showing helix target vs designed protein]
   
   ## Limitations
   
   - Works best with elongated, continuous shapes
   - Sharp corners are difficult (proteins don't do sharp corners)
   - Hollow structures are challenging
   - Very small or very large shapes may fail
   
   ## API Reference
   
   [Brief documentation of main class/functions]
   ```

4. **Example script**

   ```python
   # examples/design_helix.py
   """Example: Design a protein matching a helix shape."""
   
   from src.stl_designer import STLProteinDesigner
   
   def main():
       designer = STLProteinDesigner(
           stl_path="examples/helix.stl",
           protein_length=100,
           target_extent=100.0
       )
       
       sequence = designer.design(verbose=10)
       
       print(f"\nDesigned sequence:\n{sequence}")
       print(f"\nMetrics: {designer.get_metrics()}")
       
       # Save structure
       with open("designed_protein.pdb", "w") as f:
           f.write(designer.get_structure())
   
   if __name__ == "__main__":
       main()
   ```

5. **Optional: Colab notebook**
   - Self-contained demo
   - Includes installation
   - Runs end-to-end

### Checkpoint

- [ ] Code passes linting
- [ ] All public functions have docstrings
- [ ] README includes installation, usage, example, limitations
- [ ] At least one example script works
- [ ] Someone unfamiliar can run it from README

### Deliverables

- Clean codebase
- README.md
- Example script(s)
- Result images

### Time Estimate: 0.5 days

---

## Repository Structure

```text
colabdesign-stl/
├── README.md
├── requirements.txt
│
├── src/
│   ├── __init__.py
│   ├── stl_processing.py      # Stage 1: STL → point cloud
│   ├── losses.py              # Stage 2: Chamfer distance
│   └── stl_designer.py        # Stage 3: Main integration
│
├── examples/
│   ├── helix.stl              # Test shape
│   ├── design_helix.py        # Example script
│   └── demo.ipynb             # Optional: Colab notebook
│
├── results/
│   └── helix_result.png       # Visualization
│
└── tests/                     # Optional
    ├── test_stl_processing.py
    └── test_losses.py
```

---

## Timeline Summary

| Stage | Task | Time |
|-------|------|------|
| 0 | Environment setup, verify ColabDesign works | 0.5 days |
| 1 | STL processing module | 0.5 days |
| 2 | Chamfer loss function | 0.25 days |
| 3a | Minimal integration (hardcoded test) | 0.5 days |
| 3b | Full STL integration | 0.5 days |
| 3c | Verification | 0.25 days |
| 4 | Validation with helix | 1 day |
| 5 | Documentation | 0.5 days |
| **Total MVP** | | **4 days** |
| Buffer | Unexpected issues | 2 days |
| Post-MVP | Additional shapes, polish | Remaining |

---

## Parameter Reference

### Recommended Starting Values

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `protein_length` | 100 | Adjust based on shape complexity |
| `num_target_points` | 1000 | More = finer detail, slower |
| `target_extent` | 100.0 Å | Longest dimension of target |
| `chamfer_weight` | 1.0 | Increase if shape not matching |
| `soft_iters` | 200 | Initial soft optimization |
| `temp_iters` | 100 | Temperature annealing |
| `hard_iters` | 20 | Final discrete optimization |

### Protein Length Guidelines

```text
Alpha helix: ~1.5Å rise per residue
Beta sheet: ~3.5Å per residue (extended)

For a shape with longest dimension D (in Å):
- Minimum length ≈ D / 3.5
- Recommended length ≈ D / 1.5 to D / 2.0
- Maximum useful length ≈ D / 1.0

Example: 100Å helix → 67-100 residues
```

### Loss Weight Tuning

```text
If pLDDT is low (<50):
  → Decrease chamfer_weight (try 0.1)
  → The shape constraint is fighting protein stability

If shape doesn't match but pLDDT is high:
  → Increase chamfer_weight (try 10.0)
  → Stability is dominating over shape

If both are bad:
  → Shape may be impossible for a protein
  → Try simpler shape or different protein length
```

---

## Common Issues and Solutions

### Issue: "chamfer" not appearing in logs

**Cause:** Callback not registered properly.

**Solution:** Verify callback is passed to `mk_afdesign_model()`:
```python
af_model = mk_afdesign_model(protocol="hallucination", loss_callback=loss_fn)
# NOT: af_model.set_callback(loss_fn)  # This might not exist
```

### Issue: Loss is NaN or Inf

**Cause:** Scale mismatch or empty point cloud.

**Solution:**
- Check target points are not empty
- Check target scale is reasonable (10-200Å)
- Add small epsilon to sqrt if using sqrt version: `jnp.sqrt(x + 1e-8)`

### Issue: pLDDT is very low (<30)

**Cause:** Shape constraint is too strong, forcing unphysical structures.

**Solution:**
- Reduce chamfer_weight
- Try a more protein-compatible shape
- Increase protein length (more flexibility)

### Issue: Shape doesn't match at all

**Cause:** Shape may be incompatible with protein geometry.

**Solution:**
- Verify target points are centered
- Try longer protein
- Try simpler shape first
- Check that scale is appropriate

### Issue: Memory error

**Cause:** Point cloud too large.

**Solution:**
- Reduce `num_target_points` (try 500)
- Reduce protein length
- Use Colab with GPU

---

## Verification Checklist

Run these checks during development:

### STL Processing
```python
points = stl_to_points("helix.stl", num_points=1000, target_extent=100)
assert points.shape == (1000, 3)
assert np.abs(points.mean(axis=0)).max() < 1.0  # Centered
assert np.abs(points).max() <= 100.0  # Scaled correctly
```

### Chamfer Loss
```python
import jax.numpy as jnp
import jax

p1 = jnp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=jnp.float32)
p2 = jnp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=jnp.float32)
assert chamfer_distance(p1, p2) == 0.0  # Identical

p3 = jnp.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=jnp.float32)
assert chamfer_distance(p1, p3) > 0.0  # Different

# Gradient check
grad_fn = jax.grad(lambda p: chamfer_distance(p, p2))
grads = grad_fn(p3)
assert grads.shape == p3.shape  # Gradients exist
```

### Cα Extraction
```python
# After running a design:
pdb_str = model.save_pdb()

# Parse CA coordinates from PDB
ca_from_pdb = []
for line in pdb_str.split('\n'):
    if line.startswith('ATOM') and ' CA ' in line:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
        ca_from_pdb.append([x, y, z])

# Compare with what loss function sees
ca_from_outputs = outputs["structure_module"]["final_atom_positions"][:, 1, :]

# Should be very close (within floating point tolerance)
```

---

## Notes Space

Use this section to record findings during implementation:

### Environment Notes
- 
- 

### Parameter Findings
- 
- 

### What Worked
- 
- 

### What Didn't Work
- 
- 

### Ideas for Improvement
- 
- 
