"""Block 1 - colour mask: how much each pixel looks like a yellow line, 0..1."""

import numpy as np


def to_float(frame: np.ndarray) -> np.ndarray:
    """RGB frame (H, W, 3[+alpha]) as float32 in [0, 1]."""
    frame = np.asarray(frame)[..., :3]
    if frame.dtype == np.uint8:
        return frame.astype(np.float32) / 255.0
    return frame.astype(np.float32)


def yellowness(rgb: np.ndarray) -> np.ndarray:
    """min(R, G) - B: high only when red AND green both exceed blue."""
    return np.minimum(rgb[..., 0], rgb[..., 1]) - rgb[..., 2]


def yellow_mask(rgb: np.ndarray, low: float = 0.15, high: float = 0.40) -> np.ndarray:
    """Soft mask: 0 below `low`, linear ramp, 1 above `high`."""
    return np.clip((yellowness(rgb) - low) / (high - low), 0.0, 1.0)
