# Results

This directory contains curated, reproducible results from the ColabDesign STL extension.

## Contents

- **`overlays/`** - Overlay plots showing target centerlines (red) vs predicted Cα coordinates (blue) for each shape
- **`metrics.csv`** - Quantitative metrics table for all designs

## How Results Were Generated

All results were generated using `examples/colab_stl_notebook.ipynb` with the following presets:
- `stl_centerline_cylinder` - Cylinder tube
- `stl_centerline_sine` - Sine-wave tube  
- `stl_centerline_helix1turn` - Helical tube (1 turn)

**Common settings:**
- `PROTEIN_LENGTH = 80`
- `TARGET_EXTENT = 30.0`
- `PATH_WEIGHT = 0.02`
- `CON_WEIGHT = 0.2`
- `PLDDT_WEIGHT = 2.0`
- `SAMPLE_SEED = 0`
- `RUN_SEED = 0`

**Iterations:** `SOFT_ITERS = 300`, `TEMP_ITERS = 150`, `HARD_ITERS = 20`

For exact reproducibility, see the commit hash and notebook version used.

