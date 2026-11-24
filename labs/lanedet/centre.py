"""Block 4 - track centre on one band: decide which blobs are the boundaries, return the midpoint."""

from dataclasses import dataclass, field

from .scan import Line


@dataclass
class BandEstimate:
    u: float | None  # track centre column, px (None = nothing to go on)
    case: str  # "pair", "left", "right", "none"
    confidence: float  # 0..1
    used: list = field(default_factory=list)  # the Line(s) the estimate is based on


def estimate_centre(
    lines: list[Line],
    half_px: float,
    prior_u: float,
    tolerance: float = 0.35,
    single_confidence: float = 0.5,
) -> BandEstimate:
    """
    lines      : blobs found on this band
    half_px    : expected track half-width on this band, px (from the camera model)
    prior_u    : where we expect the centre (last frame's estimate, or the image centre)
    tolerance  : accepted relative error between a pair's spacing and the expected width
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
    if candidates:
        _, u, side, line = min(candidates, key=lambda c: c[0])
        conf = single_confidence * (0.8 if line.clipped else 1.0)
        return BandEstimate(u, side, conf, [line])

    return BandEstimate(None, "none", 0.0, [])
