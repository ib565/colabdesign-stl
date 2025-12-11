# %% [markdown]
# # Stage 3a: Chamfer Shape Loss - Colab Test
# 
# This notebook tests the custom Chamfer loss integration with ColabDesign hallucination.
# Run cells in order.

# %% [markdown]
# ## 1. Setup (run once)

# %%
# Install dependencies
import os
if not os.path.isdir("params"):
    # Install ColabDesign
    os.system("pip -q install git+https://github.com/sokrypton/ColabDesign.git@v1.1.1")
    # Symlink for debugging
    os.system("ln -s /usr/local/lib/python3.*/dist-packages/colabdesign colabdesign")
    # Download AlphaFold params (~3.5GB)
    os.system("mkdir params")
    os.system("apt-get install aria2 -qq")
    os.system("aria2c -q -x 16 https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar")
    os.system("tar -xf alphafold_params_2022-12-06.tar -C params")

# %%
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import jax
import jax.numpy as jnp
import numpy as np
print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")
print(f"JAX default backend: {jax.default_backend()}")

# %%
from colabdesign import mk_afdesign_model, clear_mem
from IPython.display import HTML

# %% [markdown]
# ## 2. Define Chamfer Loss (our custom loss)

# %%
def chamfer_distance(pred, target, use_sqrt=False, eps=1e-8):
    """Chamfer distance between two point clouds."""
    pred = jnp.asarray(pred, dtype=jnp.float32)
    target = jnp.asarray(target, dtype=jnp.float32)
    
    # Pairwise squared distances: (L, N)
    diff = pred[:, None, :] - target[None, :, :]
    sq_dist = jnp.sum(diff * diff, axis=-1)
    
    if use_sqrt:
        loss_pred_to_target = jnp.mean(jnp.sqrt(jnp.min(sq_dist, axis=1) + eps))
        loss_target_to_pred = jnp.mean(jnp.sqrt(jnp.min(sq_dist, axis=0) + eps))
    else:
        loss_pred_to_target = jnp.mean(jnp.min(sq_dist, axis=1))
        loss_target_to_pred = jnp.mean(jnp.min(sq_dist, axis=0))
    
    return loss_pred_to_target + loss_target_to_pred


def make_shape_loss(target_points, use_sqrt=False):
    """Create a shape loss callback for ColabDesign."""
    target = jnp.asarray(target_points, dtype=jnp.float32)
    target_centered = target - target.mean(axis=0)
    
    CA_INDEX = 1  # Atom order: N=0, CA=1, C=2, O=3, CB=4, ...
    
    def shape_loss(inputs, outputs):
        positions = outputs["structure_module"]["final_atom_positions"]
        ca = positions[:, CA_INDEX, :]
        ca_centered = ca - ca.mean(axis=0)
        loss = chamfer_distance(ca_centered, target_centered, use_sqrt=use_sqrt)
        return {"chamfer": loss}
    
    return shape_loss

print("Chamfer loss functions defined ✓")

# %% [markdown]
# ## 3. Quick test: Chamfer loss standalone

# %%
# Test chamfer_distance works and is differentiable
p1 = jnp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=jnp.float32)
p2 = jnp.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=jnp.float32)
p3 = jnp.array([[0, 1, 0], [1, 1, 0], [2, 1, 0]], dtype=jnp.float32)

loss_same = chamfer_distance(p1, p2)
loss_diff = chamfer_distance(p1, p3)
print(f"Chamfer (identical): {loss_same:.4f} (should be 0)")
print(f"Chamfer (offset by 1): {loss_diff:.4f} (should be 2.0 = 1^2 * 2)")

# Test gradient
grad_fn = jax.grad(lambda p: chamfer_distance(p, p2))
grads = grad_fn(p3)
print(f"Gradient shape: {grads.shape} ✓")
print(f"Gradients:\n{grads}")

# %% [markdown]
# ## 4. Create target point cloud (simple line)

# %%
# Simple 1D line target along x-axis
LENGTH = 50
target_points = np.linspace([0, 0, 0], [100, 0, 0], LENGTH).astype(np.float32)
print(f"Target shape: {target_points.shape}")
print(f"Target extent: {target_points.max(axis=0) - target_points.min(axis=0)}")

# %% [markdown]
# ## 5. Create model with custom loss

# %%
clear_mem()

# Create loss callback
loss_fn = make_shape_loss(target_points, use_sqrt=False)

# Create model
af_model = mk_afdesign_model(
    protocol="hallucination",
    loss_callback=loss_fn,
    data_dir="."  # params folder is in current dir
)

print(f"Model params loaded: {af_model._model_names}")

# Set weights explicitly
af_model.opt["weights"].update({
    "chamfer": 1.0,
    "con": 1.0,
    "plddt": 0.1,
    "pae": 0.05,
    "exp_res": 0.0,
    "helix": 0.0,
})
print(f"Weights: {af_model.opt['weights']}")

# Prepare inputs
af_model.prep_inputs(length=LENGTH)
af_model.restart(mode="gumbel", seed=0)
print("Model ready ✓")

# %% [markdown]
# ## 6. Run design (short test)

# %%
# Run a short design to verify everything works
print("Starting design (this may take a few minutes on first run due to JIT)...")
af_model.design_3stage(
    soft_iters=20,
    temp_iters=10,
    hard_iters=5,
    verbose=1
)
print("Design complete!")

# %% [markdown]
# ## 7. Results

# %%
# Get final metrics
logs = af_model._tmp.get("log", [])
final_log = logs[-1] if logs else {}
print("Final log entry:")
for k, v in final_log.items():
    if isinstance(v, (int, float)):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

# %%
# Get sequence
seqs = af_model.get_seqs()
if seqs:
    print(f"\nDesigned sequence ({len(seqs[0])} aa):")
    print(seqs[0])

# %%
# Check if chamfer is in the logs
if "chamfer" in final_log:
    print(f"\n✓ SUCCESS: 'chamfer' appears in logs with value {final_log['chamfer']:.4f}")
else:
    print("\n✗ ISSUE: 'chamfer' not found in logs")
    print("Available keys:", list(final_log.keys()))

# %% [markdown]
# ## 8. Visualize (optional)

# %%
# Save and visualize structure
af_model.save_pdb("test_design.pdb")
af_model.plot_pdb()

# %%
# Animation of design trajectory
HTML(af_model.animate())

