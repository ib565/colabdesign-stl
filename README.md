# ColabDesign STL Extension

Design proteins to approximate arbitrary STL shapes by integrating custom shape losses into [ColabDesign](https://github.com/sokrypton/ColabDesign). This extension adds STL → point-cloud preprocessing, centerline extraction for tube-like shapes, and a per-index path loss for ordered targets.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ib565/colabdesign-stl/blob/main/examples/colab_stl_notebook.ipynb)

## Quick Start

### Colab (Recommended)

1. **Open the notebook**: Click the "Open In Colab" badge above or [open directly](https://colab.research.google.com/github/ib565/colabdesign-stl/blob/main/examples/colab_stl_notebook.ipynb)
2. **Choose a preset**: Set `PRESET = "stl_centerline_cylinder"` (or `sine_tube`, `helix_tube_1turn`)
3. **Run all cells**: The notebook will clone the repo, install dependencies, and run protein design

**Note:** First run downloads AlphaFold parameters (~3.5GB) and JIT compilation takes 30-90 seconds.

### Local (Advanced)

**Requirements:**
- Python 3.10+ (tested on 3.13)
- GPU recommended (CPU is very slow)
- AlphaFold parameters (see below)

**Installation:**
```bash
git clone https://github.com/ib565/colabdesign-stl.git
cd colabdesign-stl
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install git+https://github.com/sokrypton/ColabDesign.git
```

**AlphaFold Parameters:**
Download and extract AlphaFold parameters (~3.5GB):
```bash
mkdir params
curl -L -o alphafold_params_2022-12-06.tar https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar
tar -xf alphafold_params_2022-12-06.tar -C params
export AF_DATA_DIR=$(pwd)  # Point to directory containing params/
```

**Run design:**
```bash
python examples/design_from_stl.py \
  --stl cylinder \
  --target-mode centerline \
  --length 80 \
  --target-extent 30.0 \
  --soft-iters 300 --temp-iters 150 --hard-iters 20 \
  --path-weight 0.02 --con 0.2 --plddt 2.0 --pae 0.2 \
  --data-dir . \
  --out-dir outputs/cylinder --plot
```

## Examples

We demonstrate protein design for three tube-like STL shapes:

### Cylinder
A straight cylindrical tube (simplest case for centerline extraction).

![Cylinder overlay](results/overlays/cylinder.png)

### Sine Tube
A tube following a sine-wave path (tests centerline extraction on curved shapes).

![Sine tube overlay](results/overlays/sine.png)

### Helix Tube
A helical tube with one complete turn (tests 3D helical centerline extraction).

![Helix tube overlay](results/overlays/helix.png)

**Metrics:** See `results/metrics.csv` for quantitative results.

## How It Works

1. **STL Processing**: Load STL mesh → sample surface points → extract centerline via PCA + binning
2. **Target Normalization**: Center and scale to target extent (Å)
3. **Path Loss**: Per-index MSE loss between predicted Cα coordinates and target centerline (with Kabsch alignment)
4. **Optimization**: ColabDesign hallucination protocol with custom loss callback

**Key Features:**
- **Centerline extraction**: PCA canonicalization + binning + smoothing for tube-like shapes
- **Per-index path loss**: Cleaner gradients than Chamfer for ordered targets
- **Kabsch alignment**: Rotation-invariant loss computation
- **No fork required**: Uses ColabDesign's `loss_callback` system

## Usage

### Notebook (Recommended)
Use `examples/colab_stl_notebook.ipynb` with presets:
- `stl_centerline_cylinder`
- `stl_centerline_sine`
- `stl_centerline_helix1turn`

### Command Line
```bash
python examples/design_from_stl.py --stl <name> --target-mode centerline [options]
```

### Programmatic
```python
from src import STLProteinDesigner

designer = STLProteinDesigner(
    stl_path="examples/stl/cylinder.stl",
    protein_length=80,
    target_extent=30.0,
    stl_target_mode="centerline",
    path_weight=0.02,
    data_dir="path/to/alphafold/params",
)
seq = designer.design(soft_iters=300, temp_iters=150, hard_iters=20)
pdb_str = designer.get_structure(save_path="structure.pdb")
designer.plot_overlay(save_path="overlay.png")
```

## Inspection Tools

**Inspect STL files:**
```bash
python scripts/inspect_stl.py cylinder --mode centerline
```

**Build target points independently:**
```bash
python scripts/build_target_points.py --mode stl_centerline --stl-path examples/stl/cylinder.stl
```

**Generate example STLs:**
```bash
python scripts/generate_stls.py
```

## Limitations

- **Tube-like shapes only**: Centerline extraction assumes roughly cylindrical cross-sections
- **Scale ambiguity**: `target_extent` controls scaling but optimal values depend on protein length
- **GPU recommended**: CPU runs are very slow (minutes per iteration)
- **Heuristic centerline**: PCA + binning works well but may fail on complex geometries
- **Single-chain only**: Current implementation designs single protein chains

## Future Work

- [ ] Surface mode with Chamfer loss (for non-tube shapes)
- [ ] Multi-chain design
- [ ] Automatic `target_extent` tuning
- [ ] Improved centerline extraction (skeletonization methods)
- [ ] Interactive visualization tools

## Project Structure

```
colabdesign-stl/
├── src/                    # Core library
│   ├── stl_processing.py  # STL → points, centerline extraction
│   ├── losses.py          # JAX Chamfer + path loss + Kabsch
│   └── stl_designer.py    # High-level orchestrator
├── examples/
│   ├── colab_stl_notebook.ipynb  # Main notebook
│   ├── design_from_stl.py        # CLI entrypoint
│   └── stl/                       # Example STL files
│       ├── *.stl                  # Mesh files
│       └── generators/            # STL generator scripts
├── scripts/                # Utility scripts
│   ├── generate_stls.py   # Generate example STLs
│   ├── inspect_stl.py     # Inspect STL files
│   └── build_target_points.py  # Build target points
├── results/                # Curated outputs
│   ├── overlays/          # Overlay plots
│   └── metrics.csv        # Quantitative metrics
└── tests/                  # Unit tests
```

## Citation

If you use this code, please cite:
- ColabDesign: [sokrypton/ColabDesign](https://github.com/sokrypton/ColabDesign)
- AlphaFold: [Jumper et al., Nature 2021](https://www.nature.com/articles/s41586-021-03819-2)

## License

See `LICENSE` file. This project extends ColabDesign, which has its own license.
