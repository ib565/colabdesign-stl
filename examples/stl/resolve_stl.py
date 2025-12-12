from pathlib import Path
from typing import Callable, Dict, Tuple

from .make_cylinder_stl import make_cylinder_stl
from .make_sine_tube_stl import make_sine_tube_stl
from ..make_helix_stl import make_helix_stl


Generator = Tuple[Callable[..., None], Dict]

DEFAULT_GENERATORS: Dict[str, Generator] = {
    "cylinder": (make_cylinder_stl, {}),
    "sine_tube": (make_sine_tube_stl, {}),
    "helix_tube_1turn": (
        make_helix_stl,
        {
            "turns": 1,
            "radius": 5.0,
            "pitch": 8.0,
            "tube_radius": 1.5,
            "samples_per_turn": 240,
            "tube_segments": 24,
        },
    ),
}


def resolve_or_generate_stl(name_or_path: str, generators: Dict[str, Generator] = None) -> Path:
    """
    Resolve an STL path, generating it if a known stem is provided.

    Search order:
      1) If name_or_path exists, return it.
      2) If basename (with/without .stl) exists under examples/stl/, return it.
      3) If stem matches a known generator, generate into examples/stl/<stem>.stl.
    """
    gen_map = generators or DEFAULT_GENERATORS
    p = Path(name_or_path)
    if p.suffix.lower() != ".stl":
        p = p.with_suffix(".stl")

    # Direct path
    if p.exists():
        return p

    # Look under examples/stl/
    root = Path(__file__).resolve().parents[2]
    stl_dir = root / "examples" / "stl"
    candidate = stl_dir / p.name
    if candidate.exists():
        return candidate

    stem = p.stem
    if stem not in gen_map:
        raise FileNotFoundError(f"STL not found and no generator for stem: {stem}")

    stl_dir.mkdir(parents=True, exist_ok=True)
    gen_fn, kwargs = gen_map[stem]
    gen_fn(str(candidate), **kwargs)
    return candidate


__all__ = ["resolve_or_generate_stl", "DEFAULT_GENERATORS"]

