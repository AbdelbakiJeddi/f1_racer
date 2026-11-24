"""Block 5 - the detector: connects mask -> scan -> centre -> camera and outputs control errors."""

import math
from dataclasses import dataclass

import numpy as np

from .camera import Camera
from .centre import BandEstimate, estimate_centre
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

    @staticmethod
    def band_row(rows: tuple[int, int]) -> float:
        return (rows[0] + rows[1]) / 2  # centre of rows r0..r1-1 in continuous v

    def _band(self, mask: np.ndarray, name: str, rows: tuple[int, int]) -> BandEstimate:
        v = self.band_row(rows)
        lines = find_lines(band_profile(mask, rows), self.line_level, self.min_mass)
        half_px = float(self.camera.width_px(self.half_width, v))
        prior_u = float(self.camera.to_pixel_u(self.prev_x[name], v))
        return estimate_centre(lines, half_px, prior_u, self.tolerance)

    def __call__(self, frame: np.ndarray) -> Detection:
        mask = yellow_mask(to_float(frame), self.mask_low, self.mask_high)
        near = self._band(mask, "near", self.near_rows)
        far = self._band(mask, "far", self.far_rows)

        v_near, v_far = self.band_row(self.near_rows), self.band_row(self.far_rows)
        x_near = self.prev_x["near"] if near.u is None else float(self.camera.lateral(near.u, v_near))
        x_far = self.prev_x["far"] if far.u is None else float(self.camera.lateral(far.u, v_far))
        self.prev_x = {"near": x_near, "far": x_far}

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
