# %% [markdown]
# # ColabDesign STL: STL → Protein Design (Colab-ready)
#
# This script is structured as a notebook (Jupytext style). Each `# %%` is a cell.
# You can open it directly in Colab (via the Colab extension) or convert to `.ipynb`.
# GPU recommended; CPU is only for smoke tests.

# %%
# Configuration: edit these for your run
USE_HELIX_PATH = True                  # True → use parametric helix points instead of STL
STL_PATH = "examples/helix.stl"        # Path to your STL (upload or generate below)
OUT_DIR = "outputs/colab_helix"        # Where to save sequence/PDB/plot
PROTEIN_LENGTH = 100
NUM_TARGET_POINTS = 200                # For helix path: points along path; for STL: samples from surface
TARGET_EXTENT = 100.0                  # Å; longest dimension after scaling
SAMPLE_SEED = 0                        # -1 or None for stochastic sampling (STL only)
RUN_SEED = 0                           # Restart seed for ColabDesign
SOFT_ITERS = 200
TEMP_ITERS = 100
HARD_ITERS = 20
CHAMFER_WEIGHT = 1.0
PLDDT_WEIGHT = 0.1
PAE_WEIGHT = 0.05
USE_SQRT = False                       # True → Chamfer in Å (slightly slower)
# Chamfer is computed after Kabsch alignment (rotation/translation-invariant).
DATA_DIR = None                        # Path to AlphaFold params (set below); if None, will try AF_DATA_DIR or ../ColabDesign
AUTO_DOWNLOAD_PARAMS = False           # Set True to auto-download AF params (~3.5 GB) into DATA_DIR if missing

# Helix path parameters (used when USE_HELIX_PATH=True)
HELIX_RADIUS = 10.0
HELIX_PITCH = 5.0
HELIX_TURNS = 3.0

# %%
# Install dependencies (Colab GPU runtime already has CUDA-enabled JAX)
import sys
import subprocess


def pip_install(*packages):
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


# Core deps
pip_install("git+https://github.com/sokrypton/ColabDesign.git")
pip_install("trimesh", "py3Dmol", "matplotlib")

# %%
# Imports and path setup
import os
from pathlib import Path
import subprocess

import numpy as np

# Repo URL (public)
REPO_URL = "https://github.com/ib565/colabdesign-stl"

# Resolve ROOT robustly (notebook-safe: __file__ may be undefined in Colab)
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

# If src/ is not present (e.g., fresh Colab), clone the repo
if not (ROOT / "src").exists():
    clone_dir = Path("/content/colabdesign-stl")
    if not clone_dir.exists():
        print(f"src/ not found; cloning {REPO_URL} into {clone_dir} ...")
        subprocess.check_call(["git", "clone", REPO_URL, str(clone_dir)])
    else:
        print(f"Using existing clone at {clone_dir}")
    ROOT = clone_dir

# Add to PYTHONPATH
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    STLProteinDesigner,
    make_helix_path,
    normalize_points,
    plot_point_cloud,
    stl_to_points,
)  # noqa: E402

# %%
# Optional: Download AlphaFold params if missing (≈3.5 GB)
AF_TAR_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"


def ensure_af_params(data_dir: Path, auto_download: bool = False):
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / "params"
    if marker.exists() and any(marker.iterdir()):
        print(f"AlphaFold params found at: {marker}")
        return marker
    if not auto_download:
        raise FileNotFoundError(
            f"AlphaFold params not found at {marker}. "
            f"Set AUTO_DOWNLOAD_PARAMS=True to fetch (~3.5 GB) or pre-populate manually."
        )
    tar_path = data_dir / "alphafold_params_2022-12-06.tar"
    if not tar_path.exists():
        cmd = ["curl", "-L", "-o", str(tar_path), AF_TAR_URL]
        print("Downloading AF params (this may take several minutes)...")
        subprocess.check_call(cmd)
    print("Extracting params...")
    subprocess.check_call(["tar", "-xf", str(tar_path), "-C", str(data_dir)])
    marker = data_dir / "params"
    print("Params ready at:", marker)
    return marker


# Resolve data_dir (priority: DATA_DIR → AF_DATA_DIR → ../ColabDesign)
resolved_data_dir = None
if DATA_DIR:
    resolved_data_dir = Path(DATA_DIR)
elif os.environ.get("AF_DATA_DIR"):
    resolved_data_dir = Path(os.environ["AF_DATA_DIR"])
else:
    candidate = ROOT.parent / "ColabDesign"
    resolved_data_dir = candidate if candidate.exists() else None

if resolved_data_dir is None:
    print("No AlphaFold params directory found. Set DATA_DIR or AF_DATA_DIR, or enable AUTO_DOWNLOAD_PARAMS.")
else:
    print("Using data_dir:", resolved_data_dir)
    if AUTO_DOWNLOAD_PARAMS:
        resolved_data_dir = ensure_af_params(resolved_data_dir, auto_download=True)

# %%
# Build target points (helix path or STL surface samples)
target_points = None

if USE_HELIX_PATH:
    target_points = normalize_points(
        make_helix_path(
            num_points=NUM_TARGET_POINTS,
            radius=HELIX_RADIUS,
            pitch=HELIX_PITCH,
            turns=HELIX_TURNS,
        ),
        target_extent=TARGET_EXTENT,
        center=True,
    )
    print("Using helical path target points (no STL file).")
else:
    GENERATE_STL = not Path(STL_PATH).exists()

    if GENERATE_STL:
        from examples.make_helix_stl import make_helix_stl  # noqa: E402

        helix_out = Path(STL_PATH)
        helix_out.parent.mkdir(parents=True, exist_ok=True)
        make_helix_stl(
            output_path=str(helix_out),
            turns=HELIX_TURNS,
            radius=HELIX_RADIUS,
            pitch=HELIX_PITCH,
        )
        print("Generated helix STL at", helix_out)
    else:
        print("Using existing STL:", STL_PATH)

# %%
# Quick visualization of sampled target points (optional)
DO_POINT_PLOT = False

if DO_POINT_PLOT:
    if USE_HELIX_PATH:
        pts = target_points
    else:
        pts = stl_to_points(
            STL_PATH,
            num_points=NUM_TARGET_POINTS,
            target_extent=TARGET_EXTENT,
            seed=SAMPLE_SEED,
        )
    plot_point_cloud(pts, title="Sampled target points", show=True, save_path=None)

# %%
# Run the design
out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

print("Initializing designer...")
designer = STLProteinDesigner(
    stl_path=None if USE_HELIX_PATH else STL_PATH,
    target_points=target_points if USE_HELIX_PATH else None,
    protein_length=PROTEIN_LENGTH,
    num_target_points=NUM_TARGET_POINTS,
    target_extent=TARGET_EXTENT,
    center=True,
    sample_seed=None if SAMPLE_SEED in (-1, None) else SAMPLE_SEED,
    chamfer_weight=CHAMFER_WEIGHT,
    plddt_weight=PLDDT_WEIGHT,
    pae_weight=PAE_WEIGHT,
    use_sqrt=USE_SQRT,
    data_dir=str(resolved_data_dir) if resolved_data_dir else None,
    verbose=max(1, SOFT_ITERS // 20),
)

print("Running design... (first JIT can take 30–90s on Colab GPU)")
seq = designer.design(
    soft_iters=SOFT_ITERS,
    temp_iters=TEMP_ITERS,
    hard_iters=HARD_ITERS,
    run_seed=RUN_SEED,
    save_best=True,
)

(out_dir / "sequence.txt").write_text(seq)
pdb_path = out_dir / "structure.pdb"
designer.get_structure(save_path=str(pdb_path), get_best=True)
metrics = designer.get_metrics()

print("Done.")
print(f"Sequence length: {len(seq)}")
unit = "Å" if USE_SQRT else "squared Å"
print(f"Chamfer: {metrics['chamfer']:.3f} ({unit})")
print(f"pLDDT:   {metrics['plddt']:.3f}")
print(f"PAE:     {metrics['pae']:.3f}")
print(f"PDB:     {pdb_path}")
print(f"Seq:     {out_dir / 'sequence.txt'}")

# %%
# Overlay plot (target points vs predicted Cα)
PLOT_OVERLAY = True
if PLOT_OVERLAY:
    plot_path = out_dir / "overlay.png"
    designer.plot_overlay(save_path=str(plot_path), show=False)
    print("Overlay saved to", plot_path)

# %%
# Inspect first 80 residues and path to files
print("First 80 aa:", seq[:80])
print("Outputs in:", out_dir.resolve())

