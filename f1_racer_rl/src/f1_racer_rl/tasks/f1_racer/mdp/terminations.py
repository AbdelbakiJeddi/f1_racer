"""Termination terms for the Leatherback racing car grouped by class."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat
from leatherback_racer.track import TrackOracle

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


class CarTerminations:
    """Encapsulates termination conditions for the vehicle."""

    @staticmethod
    def off_track(
        env: ManagerBasedRLEnv,
        max_lateral_distance: float = 0.75,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Terminates episode if the car drifts past the outer track boundary ribbons."""
        asset: Articulation = env.scene[asset_cfg.name]
        oracle = TrackOracle.get_instance(device=str(asset.device))

        pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

        query = oracle.query(pos_2d, yaw)
        lat_err = torch.abs(query["lateral_error"].squeeze(-1))

        return lat_err > max_lateral_distance

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


# Direct functional aliases for compatibility
off_track = CarTerminations.off_track
vehicle_rollover = CarTerminations.rollover
