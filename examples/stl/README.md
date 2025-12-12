# Example STL Files

This directory contains example STL mesh files for testing the ColabDesign STL extension.

## Available STLs

### `cylinder.stl`
A simple straight cylindrical tube. Good for testing centerline extraction on a trivial case.

**Parameters:**
- Radius: 5.0
- Height: 30.0
- Sections: 64

**Generate:** `python examples/stl/generators/make_cylinder_stl.py --out examples/stl/cylinder.stl`

### `sine_tube.stl`
A tube following a sine-wave path (one oscillation). Tests centerline extraction on curved shapes.

**Parameters:**
- Length: 30.0
- Amplitude: 6.0
- Tube radius: 1.5
- Samples: 300
- Tube segments: 24

**Generate:** `python examples/stl/generators/make_sine_tube_stl.py --out examples/stl/sine_tube.stl`

### `helix_tube_1turn.stl`
A helical tube with one complete turn. Tests centerline extraction on 3D helical paths.

**Parameters:**
- Turns: 1
- Radius: 5.0
- Pitch: 8.0
- Tube radius: 1.5
- Samples per turn: 240
- Tube segments: 24

**Generate:** `python examples/stl/generators/make_helix_stl.py --out examples/stl/helix_tube_1turn.stl --turns 1 --radius 5.0 --pitch 8.0 --tube_radius 1.5 --samples_per_turn 240 --tube_segments 24`

## Regenerating All STLs

Use the convenience script:
```bash
python scripts/generate_stls.py
```

This will generate all three STLs with the default parameters listed above.

