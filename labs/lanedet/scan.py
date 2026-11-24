"""Block 2 - scanlines: collapse a band of rows to a 1D profile and find the line blobs in it."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Line:
    u: float  # centroid, continuous pixel coordinate (pixel c spans [c, c+1))
    mass: float  # integrated profile, roughly the line width in pixels
    clipped: bool  # touches the image border, may be partly out of view


def band_profile(mask: np.ndarray, rows: tuple[int, int]) -> np.ndarray:
    """Mean of mask rows r0..r1-1 -> (W,) profile."""
    r0, r1 = rows
    return mask[r0:r1].mean(axis=0)


def find_lines(profile: np.ndarray, level: float = 0.1, min_mass: float = 1.0) -> list[Line]:
    """Runs of the profile above `level` -> one Line per run, left to right."""
    on = np.concatenate([[False], profile > level, [False]])
    edges = np.flatnonzero(on[1:] != on[:-1])
    lines = []
    for start, end in zip(edges[::2], edges[1::2]):
        p = profile[start:end]
        mass = float(p.sum())
        if mass < min_mass:
            continue
        u = float((p * (np.arange(start, end) + 0.5)).sum() / mass)
        lines.append(Line(u, mass, clipped=start == 0 or end == len(profile)))
    return lines
