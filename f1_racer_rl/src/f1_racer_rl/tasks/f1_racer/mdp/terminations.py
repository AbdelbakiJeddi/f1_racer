"""Termination terms for the Leatherback racing car grouped by class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from leatherback_racer.track import TrackOracle

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


# Outer wheel corners in the car frame (x forward, y left), m. From leatherback.usd wheel bounds:
# front wheels reach x = +0.202, rear wheels x = -0.222, outer faces sit at y = +-0.14.
WHEEL_CORNERS = ((0.202, 0.14), (0.202, -0.14), (-0.222, 0.14), (-0.222, -0.14))


def wheel_lateral_extent(env: ManagerBasedRLEnv, asset: Articulation) -> torch.Tensor:
    """Largest |lateral distance to the centerline| over the four outer wheel corners, (N,)."""
    oracle = TrackOracle.get_instance(device=str(asset.device))
    pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    extent = torch.zeros_like(yaw)
    for x, y in WHEEL_CORNERS:
        corner = pos_2d + torch.stack([cos * x - sin * y, sin * x + cos * y], dim=-1)
        lat = oracle.query(corner, yaw)["lateral_error"].squeeze(-1)
        extent = torch.maximum(extent, lat.abs())
    return extent


class CarTerminations:
    """Encapsulates termination conditions for the vehicle."""

    @staticmethod
    def off_track(
        env: ManagerBasedRLEnv,
        line_thickness: float = 0.08,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Terminates as soon as any wheel's outer edge reaches the inner edge of a boundary line."""
        asset: Articulation = env.scene[asset_cfg.name]
        oracle = TrackOracle.get_instance(device=str(asset.device))
        limit = oracle.track_width / 2.0 - line_thickness / 2.0
        return wheel_lateral_extent(env, asset) >= limit

    @staticmethod
    def rollover(
        env: ManagerBasedRLEnv,
        max_tilt_rad: float = 0.785,  # ~45 degrees
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Terminates episode if the car rolls or pitches excessively (crashed/flipped)."""
        asset: Articulation = env.scene[asset_cfg.name]
        roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)

        return (torch.abs(roll) > max_tilt_rad) | (torch.abs(pitch) > max_tilt_rad)

    @staticmethod
    def lap_completed(env: ManagerBasedRLEnv) -> torch.Tensor:
        """Terminates when the car successfully completes the full lap / turn."""
        if hasattr(env, "_lap_completed"):
            return env._lap_completed
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


# Direct functional aliases for compatibility
off_track = CarTerminations.off_track
vehicle_rollover = CarTerminations.rollover
lap_completed = CarTerminations.lap_completed
