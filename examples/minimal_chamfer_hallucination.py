"""
Stage 3a minimal integration: custom Chamfer shape loss with ColabDesign
hallucination protocol. Uses a simple line target to verify the loss plugs in
and is tracked during optimization.

Run (few iters for smoke test):
    python examples/minimal_chamfer_hallucination.py --iters 20 --length 50
"""

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Ensure project root is on PYTHONPATH when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.losses import make_shape_loss  # noqa: E402

DATA_DIR = Path(__file__).resolve().parents[1] / ".." / "ColabDesign"

def _check_jax():
    """Print JAX platform info for debugging."""
    import jax
    print(f"JAX version: {jax.__version__}")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX default backend: {jax.default_backend()}", flush=True)

def _mk_model(
    length: int,
    chamfer_weight: float,
    plddt_weight: float,
    pae_weight: float,
    use_sqrt: bool,
    seed: int,
):
    try:
        from colabdesign import mk_afdesign_model
    except ImportError as exc:
        print(
            "colabdesign is required for this demo. Install it or run in Colab.",
            file=sys.stderr,
        )
        raise SystemExit(1) from exc

    # Simple 1D line target along x-axis.
    target = np.linspace([0, 0, 0], [100, 0, 0], length).astype(np.float32)
    loss_fn = make_shape_loss(target, use_sqrt=use_sqrt)

    af_model = mk_afdesign_model(protocol="hallucination", loss_callback=loss_fn, data_dir=DATA_DIR)
    print("Loaded model params:", getattr(af_model, "_model_names", []))
    if not getattr(af_model, "_model_names", []):
        raise RuntimeError(f"No model params loaded. Check data_dir={DATA_DIR} or AF_DATA_DIR.")

    # Explicitly set weights; ColabDesign defaults set many to zero.
    weights = af_model.opt["weights"]
    weights.update(
        {
            "chamfer": chamfer_weight,
            "con": weights.get("con", 1.0),
            "i_con": weights.get("i_con", 0.0),
            "plddt": plddt_weight,
            "pae": pae_weight,
            "exp_res": weights.get("exp_res", 0.0),
            "helix": weights.get("helix", 0.0),
        }
    )

    af_model.prep_inputs(length=length)
    af_model.restart(mode="gumbel", seed=seed)
    return af_model


def main():
    parser = argparse.ArgumentParser(description="Minimal Chamfer loss demo.")
    parser.add_argument("--length", type=int, default=50, help="Protein length.")
    parser.add_argument("--soft-iters", type=int, default=5, help="Soft design iterations.")
    parser.add_argument("--temp-iters", type=int, default=3, help="Temp annealing iterations.")
    parser.add_argument("--hard-iters", type=int, default=2, help="Hard design iterations.")
    parser.add_argument(
        "--chamfer-weight", type=float, default=1.0, help="Weight for Chamfer loss."
    )
    parser.add_argument("--plddt", type=float, default=0.1, help="pLDDT weight.")
    parser.add_argument("--pae", type=float, default=0.05, help="PAE weight.")
    parser.add_argument(
        "--use-sqrt",
        action="store_true",
        help="Use sqrt Chamfer (Å units, slightly slower).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--verbose", type=int, default=1, help="Print every N steps.")

    args = parser.parse_args()

    _check_jax()

    af_model = _mk_model(
        length=args.length,
        chamfer_weight=args.chamfer_weight,
        plddt_weight=args.plddt,
        pae_weight=args.pae,
        use_sqrt=args.use_sqrt,
        seed=args.seed,
    )

    print("Starting design... (first JIT can take 1–3 minutes)", flush=True)
    print(f"  soft_iters={args.soft_iters}, temp_iters={args.temp_iters}, hard_iters={args.hard_iters}", flush=True)
    try:
        af_model.design_3stage(
            soft_iters=args.soft_iters,
            temp_iters=args.temp_iters,
            hard_iters=args.hard_iters,
            verbose=max(1, args.verbose),
        )
    except Exception as exc:
        print(f"ERROR during design: {exc}", flush=True)
        import traceback
        traceback.print_exc()
        raise
    print("Design finished.", flush=True)

    logs = af_model._tmp.get("log", [])
    final_log: Dict[str, Any] = logs[-1] if logs else {}
    print("Final log entry:", final_log, flush=True)

    seqs = af_model.get_seqs()
    if seqs:
        print("Designed sequence (first 60 aa):", seqs[0][:60], flush=True)


if __name__ == "__main__":
    main()

