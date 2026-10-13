# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reward terms for the Leatherback racing car."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from .track_oracle import TrackOracle

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def track_centering(env: ManagerBasedRLEnv, std: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gaussian reward maximizing time spent on the track centerline.

    Returns 1.0 when perfectly centered, decaying smoothly as vehicle approaches boundaries.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    oracle = TrackOracle.get_instance(device=str(asset.device))

    pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

    query = oracle.query(pos_2d, yaw)
    lat_err = query["lateral_error"].squeeze(-1)  # (N,)

    return torch.exp(-torch.square(lat_err) / (2.0 * std * std))


def track_forward_progress(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward for forward velocity projected along the track tangent."""
    asset: Articulation = env.scene[asset_cfg.name]
    oracle = TrackOracle.get_instance(device=str(asset.device))

    pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

    query = oracle.query(pos_2d, yaw)
    tan_vec = query["tangent"]  # (N, 2)

    vel_world_2d = asset.data.root_lin_vel_w.torch[:, :2]
    progress = torch.sum(vel_world_2d * tan_vec, dim=-1)  # (N,)
    return progress


def track_heading_alignment(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward cosine of heading error relative to track tangent direction."""
    asset: Articulation = env.scene[asset_cfg.name]
    oracle = TrackOracle.get_instance(device=str(asset.device))

    pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

    query = oracle.query(pos_2d, yaw)
    head_err = query["heading_error"].squeeze(-1)  # (N,)

    return torch.cos(head_err)


def boundary_violation_penalty(
    env: ManagerBasedRLEnv,
    margin_threshold: float = 0.45,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalizes vehicle when it strays beyond the boundary safety margin."""
    asset: Articulation = env.scene[asset_cfg.name]
    oracle = TrackOracle.get_instance(device=str(asset.device))

    pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

    query = oracle.query(pos_2d, yaw)
    lat_err = torch.abs(query["lateral_error"].squeeze(-1))  # (N,)

    violation = torch.clamp(lat_err - margin_threshold, min=0.0)
    return torch.square(violation)


def steering_smoothness_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Steering_Knuckle_.*"]),
) -> torch.Tensor:
    """Penalizes excessive steering angle magnitude to encourage smooth trajectories."""
    asset: Articulation = env.scene[asset_cfg.name]
    steer = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(steer), dim=-1)
