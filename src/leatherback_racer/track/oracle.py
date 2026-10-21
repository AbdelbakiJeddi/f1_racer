"""Mathematical Track Oracle for privileged state queries and analytical track geometry.

Provides fast, vectorized GPU lookups for nearest centerline points, lateral distance,
heading error, tangents, and normals.
"""

from __future__ import annotations

import os
from typing import Optional, Dict, Any
import numpy as np
import torch

from isaaclab.utils.math import wrap_to_pi


def get_default_track_npz_path() -> str:
    """Finds the default track_data.npz path in the project asset directory."""
    # Look in f1_racer_rl assets
    try:
        from f1_racer_rl.assets import F1_RACER_RL_ASSETS_DIR
        npz_path = str(F1_RACER_RL_ASSETS_DIR / "track_data.npz")
        if os.path.exists(npz_path):
            return npz_path
    except ImportError:
        pass

    # Fallback to relative path search
    candidates = [
        "f1_racer_rl/src/f1_racer_rl/assets/data/track_data.npz",
        "assets/data/track_data.npz",
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    return "track_data.npz"


class TrackOracle:
    """Pre-loaded GPU cache of mathematical track geometry for batched error computations.
    
    Provides analytical ground-truth track queries used by privileged Critics,
    reward engines, and reset event terms.
    """

    _instance: Optional[TrackOracle] = None

    def __init__(self, npz_path: Optional[str] = None, device: str = "cuda"):
        if npz_path is None:
            npz_path = get_default_track_npz_path()

        if not os.path.exists(npz_path):
            raise FileNotFoundError(f"Track metadata NPZ not found at: {npz_path}")

        data = np.load(npz_path)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Load arrays to torch tensor on target device
        self.centerline = torch.tensor(data["centerline"], dtype=torch.float32, device=self.device)  # (M, 2)
        self.tangents = torch.tensor(data["tangents"], dtype=torch.float32, device=self.device)      # (M, 2)
        self.normals = torch.tensor(data["normals"], dtype=torch.float32, device=self.device)        # (M, 2)
        self.left_line = torch.tensor(data["left_line"], dtype=torch.float32, device=self.device)    # (M, 2)
        self.right_line = torch.tensor(data["right_line"], dtype=torch.float32, device=self.device)  # (M, 2)
        self.track_width = float(data["track_width"])
        self.is_closed = bool(data["is_closed"])
        self.num_points = self.centerline.shape[0]

        # Precompute tangent orientations in radians
        self.tangent_yaws = torch.atan2(self.tangents[:, 1], self.tangents[:, 0])  # (M,)

    @classmethod
    def get_instance(cls, device: str = "cuda", npz_path: Optional[str] = None) -> TrackOracle:
        """Singleton accessor for TrackOracle."""
        target_device_type = "cuda" if (torch.cuda.is_available() and "cuda" in device) else "cpu"
        if cls._instance is None or cls._instance.device.type != target_device_type:
            cls._instance = cls(npz_path=npz_path, device=device)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Resets the singleton instance (useful when track changes dynamically)."""
        cls._instance = None

    def query(self, pos_xy: torch.Tensor, yaw: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Vectorized computation of vehicle errors relative to the track.

        Args:
            pos_xy: Vehicle positions in local env frame (N, 2).
            yaw: Vehicle heading in radians (N,).

        Returns:
            Dictionary containing:
                - lateral_error: (N, 1) signed lateral distance to centerline (+left, -right)
                - heading_error: (N, 1) signed angular error relative to track tangent
                - closest_idx: (N,) index of closest centerline point
                - tangent: (N, 2) unit tangent vector at closest point
                - normal: (N, 2) unit normal vector at closest point
                - closest_point: (N, 2) coordinates of closest centerline point
        """
        if pos_xy.device != self.device:
            pos_xy = pos_xy.to(self.device)
        if yaw.device != self.device:
            yaw = yaw.to(self.device)

        # Pairwise distance to all centerline points: (N, 1, 2) - (1, M, 2) -> (N, M, 2)
        diff = pos_xy.unsqueeze(1) - self.centerline.unsqueeze(0)
        dists_sq = torch.sum(diff ** 2, dim=-1)  # (N, M)
        closest_idx = torch.argmin(dists_sq, dim=-1)  # (N,)

        # Gather closest reference geometry
        c_pt = self.centerline[closest_idx]      # (N, 2)
        c_tan = self.tangents[closest_idx]        # (N, 2)
        c_norm = self.normals[closest_idx]        # (N, 2)
        c_yaw = self.tangent_yaws[closest_idx]    # (N,)

        # Lateral error: displacement vector projected onto track normal
        disp = pos_xy - c_pt                      # (N, 2)
        lat_error = torch.sum(disp * c_norm, dim=-1, keepdim=True)  # (N, 1)

        # Heading error: vehicle yaw minus track tangent direction wrapped to [-pi, pi]
        heading_error = wrap_to_pi(yaw - c_yaw).unsqueeze(-1)       # (N, 1)

        return {
            "lateral_error": lat_error,
            "heading_error": heading_error,
            "closest_idx": closest_idx,
            "tangent": c_tan,
            "normal": c_norm,
            "closest_point": c_pt,
        }
