"""Reward terms for the Leatherback racing car grouped by class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from leatherback_racer.track import TrackOracle

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

from .terminations import wheel_lateral_extent

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


class CarRewards:
    """Encapsulates all reward and penalty evaluation terms for the vehicle."""

    @staticmethod
    def centering(
        env: ManagerBasedRLEnv,
        std: float = 0.25,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
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

    @staticmethod
    def forward_progress(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Reward for forward velocity projected along the track tangent."""
        asset: Articulation = env.scene[asset_cfg.name]
        oracle = TrackOracle.get_instance(device=str(asset.device))

        pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

        query = oracle.query(pos_2d, yaw)
        tan_vec = query["tangent"]  # (N, 2)

        vel_world_2d = asset.data.root_lin_vel_w.torch[:, :2]
        return torch.sum(vel_world_2d * tan_vec, dim=-1)  # (N,)

    @staticmethod
    def heading_alignment(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Reward cosine of heading error relative to track tangent direction."""
        asset: Articulation = env.scene[asset_cfg.name]
        oracle = TrackOracle.get_instance(device=str(asset.device))

        pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

        query = oracle.query(pos_2d, yaw)
        head_err = query["heading_error"].squeeze(-1)  # (N,)

        return torch.cos(head_err)

    @staticmethod
    def boundary_penalty(
        env: ManagerBasedRLEnv,
        margin_threshold: float = 0.50,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalizes the outermost wheel once it gets closer to the boundary line than the safety margin."""
        asset: Articulation = env.scene[asset_cfg.name]
        violation = torch.clamp(wheel_lateral_extent(env, asset) - margin_threshold, min=0.0)
        return torch.square(violation)

    @staticmethod
    def steering_smoothness(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["Knuckle__Upright__Front_.*"]),
    ) -> torch.Tensor:
        """Penalizes knuckle angular velocity to ensure smooth physical steering."""
        asset: Articulation = env.scene[asset_cfg.name]
        knuckle_vel = asset.data.joint_vel.torch[:, asset_cfg.joint_ids]
        return torch.sum(torch.square(knuckle_vel), dim=-1)

    @staticmethod
    def steering_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
        """Penalizes high-frequency steering action changes between consecutive steps."""
        action_term = env.action_manager.get_term("drive")
        delta_steer = action_term.raw_actions[:, 0] - action_term.prev_raw_actions[:, 0]
        return torch.square(delta_steer)

    @staticmethod
    def slow_speed_penalty(
        env: ManagerBasedRLEnv,
        min_speed: float = 1.5,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Penalizes vehicle when forward speed along track tangent is below min_speed."""
        asset: Articulation = env.scene[asset_cfg.name]
        oracle = TrackOracle.get_instance(device=str(asset.device))

        pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

        query = oracle.query(pos_2d, yaw)
        tan_vec = query["tangent"]

        vel_world_2d = asset.data.root_lin_vel_w.torch[:, :2]
        fwd_speed = torch.sum(vel_world_2d * tan_vec, dim=-1)

        return torch.clamp(min_speed - fwd_speed, min=0.0)

    @staticmethod
    def lap_completion_bonus(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Awards a large one-time reward when completing a full turn / lap around the track."""
        if not hasattr(env, "_checkpoint_progress"):
            env._checkpoint_progress = torch.zeros(env.num_envs, dtype=torch.int64, device=env.device)
            env._lap_completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        asset: Articulation = env.scene[asset_cfg.name]
        oracle = TrackOracle.get_instance(device=str(asset.device))

        pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

        query = oracle.query(pos_2d, yaw)
        closest_idx = query["closest_idx"]
        num_points = oracle.num_points

        # Handle episode resets
        reset_ids = env.episode_length_buf == 0
        env._checkpoint_progress[reset_ids] = 0
        env._lap_completed[reset_ids] = False

        # Progress checkpoints (1/4, 1/2, 3/4)
        q1 = num_points // 4
        half = num_points // 2
        q3 = (3 * num_points) // 4

        env._checkpoint_progress = torch.where(
            (env._checkpoint_progress == 0) & (closest_idx >= q1) & (closest_idx < half),
            1,
            env._checkpoint_progress,
        )
        env._checkpoint_progress = torch.where(
            (env._checkpoint_progress == 1) & (closest_idx >= half) & (closest_idx < q3),
            2,
            env._checkpoint_progress,
        )
        env._checkpoint_progress = torch.where(
            (env._checkpoint_progress == 2) & (closest_idx >= q3),
            3,
            env._checkpoint_progress,
        )

        completed = (env._checkpoint_progress == 3) & (closest_idx < max(1, q1 // 2)) & (~env._lap_completed)
        env._lap_completed = env._lap_completed | completed

        return completed.float()


# Direct functional aliases for compatibility
track_centering = CarRewards.centering
track_forward_progress = CarRewards.forward_progress
track_heading_alignment = CarRewards.heading_alignment
boundary_violation_penalty = CarRewards.boundary_penalty
steering_smoothness_penalty = CarRewards.steering_smoothness
steering_rate_penalty = CarRewards.steering_rate
slow_speed_penalty = CarRewards.slow_speed_penalty
lap_completion_bonus = CarRewards.lap_completion_bonus
