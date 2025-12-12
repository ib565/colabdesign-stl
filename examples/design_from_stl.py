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
import sys
from pathlib import Path

# Ensure project root on PYTHONPATH when run as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import STLProteinDesigner  # noqa: E402
from examples.stl.resolve_stl import resolve_or_generate_stl  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Design a protein to match an STL shape.")
    parser.add_argument("--stl", required=True, help="Path to STL file.")
    parser.add_argument("--length", type=int, default=100, help="Protein length (residues).")
    parser.add_argument("--num-points", type=int, default=1000, help="Number of target points to sample (surface mode only).")
    parser.add_argument("--target-extent", type=float, default=100.0, help="Longest target dimension after scaling (Å).")
    parser.add_argument("--center", action=argparse.BooleanOptionalAction, default=True, help="Center target point cloud.")
    parser.add_argument("--normalize-target", action=argparse.BooleanOptionalAction, default=True, help="Apply bbox normalization to target points.")
    parser.add_argument("--target-mode", choices=["surface", "centerline"], default="centerline", help="Use STL surface sampling or extracted centerline.")
    parser.add_argument("--target-arclength", type=float, default=None, help="Optional target arclength for centerline rescaling.")
    parser.add_argument("--centerline-surface-samples", type=int, default=10_000, help="Surface samples for centerline extraction.")
    parser.add_argument("--centerline-bins", type=int, default=None, help="Bins along PCA axis for centerline extraction (default 4*L).")
    parser.add_argument("--centerline-smooth-window", type=int, default=5, help="Moving-average window for centerline smoothing.")
    parser.add_argument("--sample-seed", type=int, default=0, help="Seed for STL sampling (None for stochastic).")
    parser.add_argument("--run-seed", type=int, default=0, help="Seed for design restart (Gumbel init).")
    parser.add_argument("--chamfer-weight", type=float, default=1.0, help="Chamfer loss weight (surface mode).")
    parser.add_argument("--path-weight", type=float, default=0.02, help="Per-index path loss weight (centerline mode).")
    parser.add_argument("--plddt", type=float, default=0.1, help="pLDDT loss weight.")
    parser.add_argument("--pae", type=float, default=0.05, help="PAE loss weight.")
    parser.add_argument("--con", type=float, default=0.5, help="Contact loss weight.")
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

    stl_path = resolve_or_generate_stl(args.stl)

    out_dir = Path(args.out_dir) if args.out_dir else Path("outputs") / stl_path.stem
    _ensure_out_dir(out_dir)

    print("Initializing designer...")
    use_path_loss = args.target_mode == "centerline"
    designer = STLProteinDesigner(
        stl_path=str(stl_path),
        protein_length=args.length,
        num_target_points=args.num_points,
        target_extent=args.target_extent,
        center=args.center,
        sample_seed=None if args.sample_seed < 0 else args.sample_seed,
        chamfer_weight=args.chamfer_weight,
        path_weight=args.path_weight,
        use_path_loss=use_path_loss,
        con_weight=args.con,
        plddt_weight=args.plddt,
        pae_weight=args.pae,
        use_sqrt=args.use_sqrt,
        stl_target_mode=args.target_mode,
        target_arclength=args.target_arclength,
        centerline_surface_samples=args.centerline_surface_samples,
        centerline_bins=args.centerline_bins,
        centerline_smooth_window=args.centerline_smooth_window,
        normalize_target_points=args.normalize_target,
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
    if use_path_loss:
        print(f"Path:   {metrics['path']:.3f} (squared Å)")
    else:
        print(
            f"Chamfer: {metrics['chamfer']:.3f} (squared Å)"
            if not args.use_sqrt
            else f"Chamfer: {metrics['chamfer']:.3f} (Å)"
        )
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

