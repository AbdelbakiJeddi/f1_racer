"""Block 5 - the detector: connects mask -> scan -> centre -> camera and outputs control errors."""

import math
from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .centre import BandEstimate, estimate_centre, refuted_sides, side_from_refuted
from .mask import to_float, yellow_mask
from .scan import band_profile, find_lines


@dataclass
class Detection:
    e_lat: float  # track centre lateral offset at the near band / track half-width (+ = centre is right)
    e_heading: float  # track direction relative to the car, rad (+ = track heads right)
    confidence: float  # 0..1
    x_near: float  # m
    x_far: float  # m
    near: BandEstimate
    far: BandEstimate
    mask: np.ndarray


class LaneDetector:
    def __init__(
        self,
        camera: Camera,
        half_width: float = 0.30,
        near_rows: tuple[int, int] = (68, 75),
        far_rows: tuple[int, int] = (52, 58),
        mask_low: float = 0.15,
        mask_high: float = 0.40,
        line_level: float = 0.1,
        min_mass: float = 1.0,
        tolerance: float = 0.35,
    ):
        self.camera = camera
        self.half_width = half_width
        self.near_rows = near_rows
        self.far_rows = far_rows
        self.mask_low = mask_low
        self.mask_high = mask_high
        self.line_level = line_level
        self.min_mass = min_mass
        self.tolerance = tolerance
        self.reset()

    def reset(self) -> None:
        self.prev_x = {"near": 0.0, "far": 0.0}  # last centre estimates, m
        self.prev_valid = False  # prev_x holds a real estimate (not just the reset value)

    @staticmethod
    def band_row(rows: tuple[int, int]) -> float:
        return (rows[0] + rows[1]) / 2  # centre of rows r0..r1-1 in continuous v

    def _scan(self, mask: np.ndarray, rows: tuple[int, int]) -> tuple[list, float, str | None]:
        """Lines on the band, half-width in px, and the side a lone line must be (None = undecided)."""
        v = self.band_row(rows)
        profile = band_profile(mask, rows)
        lines = find_lines(profile, self.line_level, self.min_mass)
        half_px = float(self.camera.width_px(self.half_width, v))
        side = side_from_refuted(refuted_sides(lines[0], half_px, profile, self.line_level)) if len(lines) == 1 else None
        return lines, half_px, side

    def __call__(self, frame: np.ndarray) -> Detection:
        mask = yellow_mask(to_float(frame), self.mask_low, self.mask_high)
        scans = {"near": self._scan(mask, self.near_rows), "far": self._scan(mask, self.far_rows)}
        v = {"near": self.band_row(self.near_rows), "far": self.band_row(self.far_rows)}
        def prior(name: str, x: float) -> float:
            return float(self.camera.to_pixel_u(x, v[name]))

        bands = {n: estimate_centre(*scans[n][:2], prior(n, self.prev_x[n]), self.tolerance) for n in v}
        # A lone line is placed by, in order: a pair on the other band, last frame's centre, the missing-line test.
        # The test comes last because worn or shadowed paint also looks like "no line there".
        for name, other in (("near", "far"), ("far", "near")):
            lines, half_px, side = scans[name]
            if bands[name].case not in ("left", "right"):
                continue
            if bands[other].case == "pair":
                x_other = float(self.camera.lateral(bands[other].u, v[other]))
                bands[name] = estimate_centre(lines, half_px, prior(name, x_other), self.tolerance)
            elif not self.prev_valid:
                # a lone line in both bands is the same boundary, so an undecided band borrows the other's answer
                if side is None and len(scans[other][0]) == 1:
                    side = scans[other][2]
                bands[name] = estimate_centre(lines, half_px, prior(name, self.prev_x[name]), self.tolerance, side=side)
        near, far = bands["near"], bands["far"]

        v_near, v_far = self.band_row(self.near_rows), self.band_row(self.far_rows)
        x_near = self.prev_x["near"] if near.u is None else float(self.camera.lateral(near.u, v_near))
        x_far = self.prev_x["far"] if far.u is None else float(self.camera.lateral(far.u, v_far))
        self.prev_x = {"near": x_near, "far": x_far}
        self.prev_valid = self.prev_valid or near.u is not None or far.u is not None

        dz = float(self.camera.distance(v_far) - self.camera.distance(v_near))
        return Detection(
            e_lat=float(np.clip(x_near / self.half_width, -1.0, 1.0)),
            e_heading=math.atan2(x_far - x_near, dz),
            confidence=(near.confidence + far.confidence) / 2,
            x_near=x_near,
            x_far=x_far,
            near=near,
            far=far,
            mask=mask,
        )
