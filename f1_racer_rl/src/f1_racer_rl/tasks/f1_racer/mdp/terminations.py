# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Termination terms for the Leatherback racing car."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from .track_oracle import TrackOracle

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


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
    lat_err = torch.abs(query["lateral_error"].squeeze(-1))  # (N,)

    return lat_err > max_lateral_distance


def vehicle_rollover(
    env: ManagerBasedRLEnv,
    max_tilt_rad: float = 0.785,  # ~45 degrees
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminates episode if the car rolls or pitches excessively (crashed/flipped)."""
    asset: Articulation = env.scene[asset_cfg.name]
    roll, pitch, _ = euler_xyz_from_quat(asset.data.root_quat_w.torch)

    tilted = (torch.abs(roll) > max_tilt_rad) | (torch.abs(pitch) > max_tilt_rad)
    return tilted
