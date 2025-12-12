# %% [markdown]
# # ColabDesign STL (centerline + path loss) — Script/Notebook
#
# Jupytext-friendly: convert to .ipynb if you prefer. This follows the
# original `colab_stl_notebook.ipynb` structure but adds:
# - `PATH_MODE="stl_centerline"` for tube-like STLs
# - Presets to avoid constant knob-twiddling
# - Optional target_arclength diagnostics; bbox scaling remains default

# %% [markdown]
# ## 1) Config (edit here)

# %%
from pathlib import Path

# Choose a preset or set to None to fully manual-tune.
PRESET = "stl_centerline_cylinder"
OVERRIDES = {}  # e.g., {"TARGET_EXTENT": 40.0}

# Fallback/manual values (used if PRESET is None or not overriding)
PATH_MODE = "helix"                     # "line", "helix", "hairpin", "stl", "stl_centerline"
STL_PATH = "examples/stl/cylinder.stl" # Used when PATH_MODE startswith("stl")
OUT_DIR = "outputs/line_smoke"
PROTEIN_LENGTH = 80
NUM_TARGET_POINTS = 80                 # For ordered paths must equal PROTEIN_LENGTH
TARGET_EXTENT = 30.0                   # Bbox max-dimension scaling (kept for consistency)
TARGET_ARCLENGTH = None                # Optional: only for centerline extraction (diagnostic)
CENTERLINE_SURFACE_SAMPLES = 10000
CENTERLINE_BINS = None
CENTERLINE_SMOOTH_WINDOW = 5
LINE_LENGTH = 40.0
SAMPLE_SEED = 0
RUN_SEED = 0
SOFT_ITERS = 300
TEMP_ITERS = 150
HARD_ITERS = 20
CHAMFER_WEIGHT = 0.02
PATH_WEIGHT = 0.02
PLDDT_WEIGHT = 2.0
PAE_WEIGHT = 0.2
CON_WEIGHT = 0.5
USE_PATH_LOSS = True                  # For ordered paths; centerline forces True
USE_SQRT = False
NORMALIZE_TARGET = True               # If False, skip bbox scaling (center only)
DATA_DIR = "/content/data_dir"
AUTO_DOWNLOAD_PARAMS = True

# Helix path parameters
HELIX_RADIUS = 5.0
HELIX_PITCH = 5.0
HELIX_TURNS = 1.0

FORCE_RECLONE = True

# %% [markdown]
# ## 2) Presets (minimal, editable)

# %%
PRESETS = {
    "line_long": {
        "PATH_MODE": "line",
        "OUT_DIR": "outputs/line_long",
        "PROTEIN_LENGTH": 80,
        "NUM_TARGET_POINTS": 80,
        "TARGET_EXTENT": 150.0,
        "LINE_LENGTH": 150.0,
        "PATH_WEIGHT": 0.02,
        "CON_WEIGHT": 0.2,
        "PLDDT_WEIGHT": 2.0,
    },
    "helix_easy": {
        "PATH_MODE": "helix",
        "OUT_DIR": "outputs/helix_easy",
        "PROTEIN_LENGTH": 80,
        "NUM_TARGET_POINTS": 80,
        "TARGET_EXTENT": 30.0,
        "HELIX_RADIUS": 5.0,
        "HELIX_PITCH": 10.0,
        "HELIX_TURNS": 1.0,
        "PATH_WEIGHT": 0.02,
    },
    "stl_centerline_cylinder": {
        "PATH_MODE": "stl_centerline",
        "STL_PATH": "examples/stl/cylinder.stl",
        "OUT_DIR": "outputs/stl_cylinder",
        "PROTEIN_LENGTH": 80,
        "NUM_TARGET_POINTS": 80,
        "TARGET_EXTENT": 30.0,
        "CENTERLINE_SURFACE_SAMPLES": 10000,
        "PATH_WEIGHT": 0.02,
        "CON_WEIGHT": 0.2,
        "PLDDT_WEIGHT": 2.0,
    },
    "stl_centerline_sine": {
        "PATH_MODE": "stl_centerline",
        "STL_PATH": "examples/stl/sine_tube.stl",
        "OUT_DIR": "outputs/stl_sine",
        "PROTEIN_LENGTH": 80,
        "NUM_TARGET_POINTS": 80,
        "TARGET_EXTENT": 30.0,
        "CENTERLINE_SURFACE_SAMPLES": 12000,
        "PATH_WEIGHT": 0.02,
        "CON_WEIGHT": 0.2,
        "PLDDT_WEIGHT": 2.0,
    },
    "stl_centerline_helix1turn": {
        "PATH_MODE": "stl_centerline",
        "STL_PATH": "examples/stl/helix_tube_1turn.stl",
        "OUT_DIR": "outputs/stl_helix1",
        "PROTEIN_LENGTH": 80,
        "NUM_TARGET_POINTS": 80,
        "TARGET_EXTENT": 30.0,
        "CENTERLINE_SURFACE_SAMPLES": 12000,
        "PATH_WEIGHT": 0.02,
        "CON_WEIGHT": 0.2,
        "PLDDT_WEIGHT": 2.0,
    },
}

# Apply preset
if PRESET is not None and PRESET in PRESETS:
    locals().update(PRESETS[PRESET])
if OVERRIDES:
    locals().update(OVERRIDES)

# %% [markdown]
# ## 3) Setup (clone, deps, data_dir)

# %%
import sys
import subprocess
import shutil
import os
import numpy as np

def pip_install(*packages):
    cmd = [sys.executable, "-m", "pip", "install", "--quiet", *packages]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)

# Core deps
pip_install("git+https://github.com/sokrypton/ColabDesign.git")
pip_install("trimesh", "py3Dmol", "matplotlib")

# Resolve ROOT (clone if missing)
REPO_URL = "https://github.com/ib565/colabdesign-stl"
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
if not (ROOT / "src").exists():
    clone_dir = Path("/content/colabdesign-stl")
    if FORCE_RECLONE and clone_dir.exists():
        print("Forcing reclone. Deleting existing repo dir")
        shutil.rmtree(clone_dir)
    if not clone_dir.exists():
        print(f"src/ not found; cloning {REPO_URL} into {clone_dir} ...")
        subprocess.check_call(["git", "clone", REPO_URL, str(clone_dir)])
    ROOT = clone_dir
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import (
    STLProteinDesigner,
    make_helix_path,
    normalize_points,
    plot_point_cloud,
    stl_to_points,
    stl_to_centerline_points,
)
from examples.stl.resolve_stl import resolve_or_generate_stl

# Resolve data_dir
def ensure_af_params(data_dir: Path, auto_download: bool = False):
    AF_TAR_URL = "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / "params"
    if marker.exists() and any(marker.iterdir()):
        print(f"AlphaFold params found at: {marker}")
        return marker
    if not auto_download:
        raise FileNotFoundError(f"AlphaFold params not found at {marker}.")
    tar_path = data_dir / "alphafold_params_2022-12-06.tar"
    if not tar_path.exists():
        cmd = ["curl", "-L", "-o", str(tar_path), AF_TAR_URL]
        print("Downloading AF params (several minutes)...")
        subprocess.check_call(cmd)
    print("Extracting params...")
    marker.mkdir(exist_ok=True)
    subprocess.check_call(["tar", "-xf", str(tar_path), "-C", str(marker)])
    print("Params ready at:", marker)
    return marker

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

# %% [markdown]
# ## 4) Build target points (line/helix/hairpin/stl_centerline)

# %%
def make_line_path(num_points: int, length: float) -> np.ndarray:
    z = np.linspace(-length / 2.0, length / 2.0, num_points, dtype=np.float32)
    zeros = np.zeros_like(z)
    return np.stack([zeros, zeros, z], axis=1)

target_points = None
USING_TARGET_POINTS = PATH_MODE in ("line", "helix", "hairpin", "stl_centerline")

if PATH_MODE == "line":
    target_points = normalize_points(
        make_line_path(num_points=NUM_TARGET_POINTS, length=LINE_LENGTH),
        target_extent=TARGET_EXTENT,
        center=True,
    )
    print("Using straight-line target points.")
elif PATH_MODE == "helix":
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
    print("Using helical path target points.")
elif PATH_MODE == "hairpin":
    from src import make_hairpin_path

    target_points = normalize_points(
        make_hairpin_path(num_points=NUM_TARGET_POINTS),
        target_extent=TARGET_EXTENT,
        center=True,
    )
    print("Using hairpin path target points.")
elif PATH_MODE == "stl_centerline":
    stl_resolved = resolve_or_generate_stl(STL_PATH)
    print("Using STL centerline from", stl_resolved)
    pts = stl_to_centerline_points(
        str(stl_resolved),
        num_points=PROTEIN_LENGTH,
        surface_samples=CENTERLINE_SURFACE_SAMPLES,
        bins=CENTERLINE_BINS,
        smooth_window=CENTERLINE_SMOOTH_WINDOW,
        seed=SAMPLE_SEED if SAMPLE_SEED >= 0 else None,
        target_arclength=TARGET_ARCLENGTH,
    )
    if NORMALIZE_TARGET:
        target_points = normalize_points(pts, target_extent=TARGET_EXTENT, center=True)
    else:
        target_points = (pts - pts.mean(axis=0)).astype(np.float32)
else:
    raise ValueError(f"Unknown PATH_MODE: {PATH_MODE}")

# Diagnostics: report arclength / avg step
def _polyline_arclength(poly):
    if len(poly) < 2:
        return 0.0
    seg = np.diff(poly, axis=0)
    return float(np.sum(np.linalg.norm(seg, axis=1)))

arc = _polyline_arclength(target_points)
avg_step = arc / max(len(target_points) - 1, 1)
print(f"Target arclength (post-scaling): {arc:.2f} Å; avg step: {avg_step:.2f} Å")

# Quick visualization (optional)
plot_point_cloud(target_points, title="Sampled target points", show=True, save_path=None)

# %% [markdown]
# ## 5) Run design

# %%
from pathlib import Path

out_dir = Path(OUT_DIR)
out_dir.mkdir(parents=True, exist_ok=True)

print("Initializing designer...")
designer = STLProteinDesigner(
    stl_path=None if USING_TARGET_POINTS else STL_PATH,
    target_points=target_points if USING_TARGET_POINTS else None,
    protein_length=PROTEIN_LENGTH,
    num_target_points=NUM_TARGET_POINTS,
    target_extent=TARGET_EXTENT,
    center=True,
    sample_seed=None if SAMPLE_SEED in (-1, None) else SAMPLE_SEED,
    chamfer_weight=CHAMFER_WEIGHT,
    path_weight=PATH_WEIGHT,
    use_path_loss=USE_PATH_LOSS or (PATH_MODE == "stl_centerline"),
    con_weight=CON_WEIGHT,
    plddt_weight=PLDDT_WEIGHT,
    pae_weight=PAE_WEIGHT,
    use_sqrt=USE_SQRT,
    data_dir=str(resolved_data_dir) if resolved_data_dir else None,
    verbose=max(1, SOFT_ITERS // 20),
    stl_target_mode="centerline" if PATH_MODE == "stl_centerline" else "surface",
    target_arclength=TARGET_ARCLENGTH,
    centerline_surface_samples=CENTERLINE_SURFACE_SAMPLES,
    centerline_bins=CENTERLINE_BINS,
    centerline_smooth_window=CENTERLINE_SMOOTH_WINDOW,
    normalize_target_points=NORMALIZE_TARGET,
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
if designer.use_path_loss:
    print(f"Path: {metrics['path']:.3f} (squared Å)")
    print(f"Path Aligned: {metrics.get('path_aligned', float('nan')):.3f} (squared Å)")
else:
    unit = "Å" if USE_SQRT else "squared Å"
    print(f"Chamfer: {metrics['chamfer']:.3f} ({unit})")
    print(f"Chamfer Aligned: {metrics.get('chamfer_aligned', float('nan')):.3f} ({unit})")
print(f"pLDDT:   {metrics['plddt']:.3f}")
print(f"PAE:     {metrics['pae']:.3f}")
print(f"PDB:     {pdb_path}")
print(f"Seq:     {out_dir / 'sequence.txt'}")

# Debug info (optional)
debug_info = designer.debug_aux_structure()
print("\n=== Debug Info ===")
for k, v in debug_info.items():
    print(f"  {k}: {v}")
print("==================\n")

# Extent diagnostics for CA vs target
ca = designer.get_ca_coords(get_best=True, aligned=False)
ca_extent = (ca.max(0) - ca.min(0)).max()
tgt_extent = (designer.target_points.max(0) - designer.target_points.min(0)).max()
print(f"CA extent: {ca_extent:.3f}  Target extent: {tgt_extent:.3f}")

# %% [markdown]
# ## 6) Overlay plot

# %%
try:
    from IPython.display import Image, display  # type: ignore
    plot_path = out_dir / "overlay.png"
    designer.plot_overlay(save_path=str(plot_path), show=False)
    print("Overlay saved to", plot_path)
    display(Image(filename=str(plot_path)))
except Exception:
    plot_path = out_dir / "overlay.png"
    designer.plot_overlay(save_path=str(plot_path), show=False)
    print("Overlay saved to", plot_path)
    print("IPython display not available in this environment.")

# %% [markdown]
# ## 7) 3D overlay (py3Dmol, optional)

# %%
try:
    import py3Dmol  # type: ignore

    pred_ca = designer.get_ca_coords(get_best=True)
    tgt = target_points if PATH_MODE in ["line", "helix", "hairpin", "stl_centerline"] else designer.target_points
    pdb_str = Path(pdb_path).read_text()

    view = py3Dmol.view(width=720, height=720)
    view.addModel(pdb_str, "pdb")
    view.setStyle({"cartoon": {"color": "skyblue", "opacity": 0.55}})
    view.addStyle({"atom": "CA"}, {"sphere": {"color": "deepskyblue", "radius": 0.7}})

    tgt = np.asarray(tgt, dtype=float)
    xyz_body = "\n".join(f"C {x:.3f} {y:.3f} {z:.3f}" for x, y, z in tgt)
    xyz_text = f"{len(tgt)}\npoints\n{xyz_body}\n"
    view.addModel(xyz_text, "xyz")
    view.setStyle({"model": 1}, {"sphere": {"color": "red", "radius": 0.8}})

    view.zoomTo()
    view.show()
except Exception as e:
    print("py3Dmol overlay not available:", e)

# %% [markdown]
# ## 8) Inspect outputs

# %%
print("First 80 aa:", seq[:80])
print("Outputs in:", out_dir.resolve())