# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Observation functions for the Leatherback racing car."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from .track_oracle import TrackOracle

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


def car_steering_angle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns normalized front knuckle steering angle (N, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]  # (N, num_joints)
    return torch.mean(joint_pos, dim=-1, keepdim=True)


def car_forward_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns car forward velocity in chassis local coordinate frame (N, 1)."""
    asset: Articulation = env.scene[asset_cfg.name]
    vx = asset.data.root_lin_vel_b.torch[:, 0:1]
    return vx


def car_proprioception(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Returns combined proprioceptive features: [steering_angle, forward_speed] (N, 2)."""
    asset: Articulation = env.scene[asset_cfg.name]
    steer = torch.mean(asset.data.joint_pos.torch[:, asset_cfg.joint_ids], dim=-1, keepdim=True)
    speed = asset.data.root_lin_vel_b.torch[:, 0:1]
    return torch.cat([steer, speed], dim=-1)


def privileged_track_state(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Computes privileged geometric errors relative to the mathematical track centerline.

    Returns:
        Tensor of shape (N, 8):
        - lateral_error: (N, 1) signed lateral distance to centerline (+left, -right)
        - heading_error: (N, 1) angular error relative to tangent in [-pi, pi]
        - vx: (N, 1) longitudinal velocity in body frame
        - vy: (N, 1) lateral velocity in body frame
        - yaw_rate: (N, 1) angular velocity around z-axis
        - steering_angle: (N, 1) knuckle joint position
        - rel_pos_tangent: (N, 1) projection of velocity along track tangent
        - norm_lat_error: (N, 1) lateral error normalized by half-track width
    """
    asset: Articulation = env.scene[asset_cfg.name]
    oracle = TrackOracle.get_instance(device=str(asset.device))

    # Car position in local environment frame
    pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
    # Car yaw
    _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

    query = oracle.query(pos_2d, yaw)
    lat_err = query["lateral_error"]   # (N, 1)
    head_err = query["heading_error"] # (N, 1)
    tan_vec = query["tangent"]        # (N, 2)

    vx = asset.data.root_lin_vel_b.torch[:, 0:1]
    vy = asset.data.root_lin_vel_b.torch[:, 1:2]
    wz = asset.data.root_ang_vel_b.torch[:, 2:3]

    steer = torch.mean(asset.data.joint_pos.torch[:, asset_cfg.joint_ids], dim=-1, keepdim=True)

    # Progress: world velocity dot tangent
    vel_world_2d = asset.data.root_lin_vel_w.torch[:, :2]
    v_tangent = torch.sum(vel_world_2d * tan_vec, dim=-1, keepdim=True)

    half_width = oracle.track_width / 2.0
    norm_lat_err = lat_err / half_width

    return torch.cat([lat_err, head_err, vx, vy, wz, steer, v_tangent, norm_lat_err], dim=-1)
