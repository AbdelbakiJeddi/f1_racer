"""Block 3 - camera model: image pixels <-> metres on a flat ground plane (level pinhole camera)."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Camera:
    width: int  # px
    height: int  # px
    focal: float  # px
    cam_height: float  # m above the ground
    v_horizon: float  # row of the horizon, px

    @property
    def cx(self) -> float:
        return self.width / 2

    def distance(self, v):
        """Forward distance z (m) of the ground seen at row v."""
        return self.focal * self.cam_height / (np.asarray(v, dtype=float) - self.v_horizon)

    def lateral(self, u, v):
        """Lateral position x (m, + right) of the ground point at pixel (u, v)."""
        return self.cam_height * (np.asarray(u, dtype=float) - self.cx) / (np.asarray(v, dtype=float) - self.v_horizon)

    def to_pixel_u(self, x, v):
        """Column u where a ground point at lateral x appears on row v."""
        return self.cx + np.asarray(x, dtype=float) * (np.asarray(v, dtype=float) - self.v_horizon) / self.cam_height

    def width_px(self, metres, v):
        """How many pixels a ground length of `metres` (across the image) spans on row v."""
        return np.asarray(metres, dtype=float) * (np.asarray(v, dtype=float) - self.v_horizon) / self.cam_height


def calibrate(rows, left_u, right_u, width: int, height: int, focal: float, cam_height: float) -> Camera:
    """Recover the horizon row from left/right line positions on a straight track.

    On flat ground both lines are straight in the image and meet at the horizon.
    """
    rows = np.asarray(rows, dtype=float)
    spread = np.asarray(right_u, dtype=float) - np.asarray(left_u, dtype=float)
    slope, intercept = np.polyfit(rows, spread, 1)  # spread = slope * (v - v_horizon)
    return Camera(width, height, focal, cam_height, v_horizon=-intercept / slope)
