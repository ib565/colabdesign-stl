#!/usr/bin/env python3
"""
Generate all example STL files for the ColabDesign STL extension.

This script generates the three example STLs (cylinder, sine_tube, helix_tube_1turn)
into examples/stl/ using the default parameters.
"""

import sys
from pathlib import Path

# Ensure project root is on PYTHONPATH
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.stl.generators.make_cylinder_stl import make_cylinder_stl
from examples.stl.generators.make_sine_tube_stl import make_sine_tube_stl
from examples.stl.generators.make_helix_stl import make_helix_stl


def main():
    stl_dir = ROOT / "examples" / "stl"
    stl_dir.mkdir(parents=True, exist_ok=True)

    print("Generating example STL files...")
    
    # Cylinder
    print("\n1. Generating cylinder.stl...")
    make_cylinder_stl(
        output_path=str(stl_dir / "cylinder.stl"),
        radius=5.0,
        height=30.0,
        sections=64,
    )
    
    # Sine tube
    print("\n2. Generating sine_tube.stl...")
    make_sine_tube_stl(
        output_path=str(stl_dir / "sine_tube.stl"),
        length=30.0,
        amplitude=6.0,
        tube_radius=1.5,
        samples=300,
        tube_segments=24,
    )
    
    # Helix tube (1 turn)
    print("\n3. Generating helix_tube_1turn.stl...")
    make_helix_stl(
        output_path=str(stl_dir / "helix_tube_1turn.stl"),
        turns=1,
        radius=5.0,
        pitch=8.0,
        tube_radius=1.5,
        samples_per_turn=240,
        tube_segments=24,
    )
    
    print("\n[OK] All STL files generated successfully!")
    print(f"Output directory: {stl_dir}")


if __name__ == "__main__":
    main()

