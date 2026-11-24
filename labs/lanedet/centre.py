"""Block 4 - track centre on one band: decide which blobs are the boundaries, return the midpoint."""

from dataclasses import dataclass, field

from .scan import Line


@dataclass
class BandEstimate:
    u: float | None  # track centre column, px (None = nothing to go on)
    case: str  # "pair", "left", "right", "none"
    confidence: float  # 0..1
    used: list = field(default_factory=list)  # the Line(s) the estimate is based on


def refuted_sides(line: Line, half_px: float, profile, level: float = 0.1, margin: float = 3.0, window: int = 4) -> set[str]:
    """Sides a lone line cannot be: the other boundary would be in view, but the profile is empty there.

    "left" means "this line is the left boundary" (the other one would sit at u + 2*half_px).
    """
    refuted = set()
    width = len(profile)
    for side, other in (("left", line.u + 2 * half_px), ("right", line.u - 2 * half_px)):
        if margin <= other <= width - margin:
            lo, hi = max(0, int(other) - window), min(width, int(other) + window + 1)
            if profile[lo:hi].max() < level:
                refuted.add(side)
    return refuted


def side_from_refuted(refuted: set[str]) -> str | None:
    """The one side left standing, or None if both or neither are refuted."""
    if refuted == {"right"}:
        return "left"
    if refuted == {"left"}:
        return "right"
    return None


def estimate_centre(
    lines: list[Line],
    half_px: float,
    prior_u: float,
    tolerance: float = 0.35,
    single_confidence: float = 0.5,
    side: str | None = None,
) -> BandEstimate:
    """
    lines      : blobs found on this band
    half_px    : expected track half-width on this band, px (from the camera model)
    prior_u    : where we expect the centre (last frame's estimate, or the image centre)
    tolerance  : accepted relative error between a pair's spacing and the expected width
    side       : if known, which boundary a lone line is ("left"/"right"); otherwise the prior decides
    """
    best = None
    for i, a in enumerate(lines):
        for b in lines[i + 1 :]:
            err = abs((b.u - a.u) - 2 * half_px) / (2 * half_px)
            if err < tolerance and (best is None or err < best[0]):
                best = (err, a, b)
    if best is not None:
        err, a, b = best
        conf = 1.0 - err / tolerance
        if a.clipped or b.clipped:
            conf *= 0.8
        return BandEstimate((a.u + b.u) / 2, "pair", conf, [a, b])

    candidates = []
    for line in lines:
        candidates.append((abs(line.u + half_px - prior_u), line.u + half_px, "left", line))
        candidates.append((abs(line.u - half_px - prior_u), line.u - half_px, "right", line))
    if side is not None and len(lines) == 1:
        candidates = [c for c in candidates if c[2] == side]
    if candidates:
        _, u, side, line = min(candidates, key=lambda c: c[0])
        conf = single_confidence * (0.8 if line.clipped else 1.0)
        return BandEstimate(u, side, conf, [line])

    return BandEstimate(None, "none", 0.0, [])
