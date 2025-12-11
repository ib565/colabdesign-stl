"""
End-to-end STL-driven protein design.

Example:
    python examples/design_from_stl.py --stl examples/helix.stl \
      --length 100 --num-points 1000 --target-extent 100 \
      --soft-iters 200 --temp-iters 100 --hard-iters 20 \
      --chamfer-weight 1.0 --plddt 0.1 --pae 0.05 \
      --sample-seed 0 --run-seed 0 --data-dir ../ColabDesign \
      --out-dir outputs/helix --plot
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure project root on PYTHONPATH when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import STLProteinDesigner  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design a protein to match an STL shape.")
    parser.add_argument("--stl", required=True, help="Path to STL file.")
    parser.add_argument("--length", type=int, default=100, help="Protein length (residues).")
    parser.add_argument("--num-points", type=int, default=1000, help="Number of target points to sample.")
    parser.add_argument("--target-extent", type=float, default=100.0, help="Longest target dimension after scaling (Å).")
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True, help="Center target point cloud.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for STL sampling (None for stochastic).")
    parser.add_argument("--run-seed", type=int, default=0, help="Seed for design restart (Gumbel init).")
    parser.add_argument("--chamfer-weight", type=float, default=1.0, help="Chamfer loss weight.")
    parser.add_argument("--plddt", type=float, default=0.1, help="pLDDT loss weight.")
    parser.add_argument("--pae", type=float, default=0.05, help="PAE loss weight.")
    parser.add_argument("--use-sqrt", action="store_true", help="Use sqrt Chamfer (Å units, slower).")
    parser.add_argument("--soft-iters", type=int, default=200, help="Soft stage iterations.")
    parser.add_argument("--temp-iters", type=int, default=100, help="Temp annealing iterations.")
    parser.add_argument("--hard-iters", type=int, default=20, help="Hard stage iterations.")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to AlphaFold params (falls back to AF_DATA_DIR or ../ColabDesign).")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for results.")
    parser.add_argument("--plot", action="store_true", help="Save overlay plot (target vs predicted CA).")
    return parser.parse_args()


def _ensure_out_dir(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = _parse_args()

    stl_path = Path(args.stl)
    if not stl_path.exists():
        raise FileNotFoundError(f"STL not found: {stl_path}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / stl_path.stem
    _ensure_out_dir(out_dir)

    print("Initializing designer...")
    designer = STLProteinDesigner(
        stl_path=str(stl_path),
        protein_length=args.length,
        num_target_points=args.num_points,
        target_extent=args.target_extent,
        center=args.center,
        sample_seed=None if args.sample_seed < 0 else args.sample_seed,
        chamfer_weight=args.chamfer_weight,
        plddt_weight=args.plddt,
        pae_weight=args.pae,
        use_sqrt=args.use_sqrt,
        data_dir=args.data_dir,
        verbose=max(1, args.soft_iters // 20),
    )

    print("Running design... (first JIT can take 1–3 minutes on CPU)")
    seq = designer.design(
        soft_iters=args.soft_iters,
        temp_iters=args.temp_iters,
        hard_iters=args.hard_iters,
        run_seed=args.run_seed,
        save_best=True,
    )

    # Save outputs
    (out_dir / "sequence.txt").write_text(seq)
    pdb_path = out_dir / "structure.pdb"
    designer.get_structure(save_path=str(pdb_path), get_best=True)
    metrics = designer.get_metrics()

    print("Done.")
    print(f"Sequence length: {len(seq)}")
    print(f"Chamfer: {metrics['chamfer']:.3f} (squared Å)" if not args.use_sqrt else f"Chamfer: {metrics['chamfer']:.3f} (Å)")
    print(f"pLDDT:  {metrics['plddt']:.3f}")
    print(f"PAE:    {metrics['pae']:.3f}")
    print(f"PDB:    {pdb_path}")
    print(f"Seq:    {(out_dir / 'sequence.txt')}")

    if args.plot:
        plot_path = out_dir / "overlay.png"
        designer.plot_overlay(save_path=str(plot_path), show=False)
        print(f"Plot:   {plot_path}")


if __name__ == "__main__":
    main()

