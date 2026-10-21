# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Action terms for the Leatherback racing car."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class LeatherbackDriveAction(ActionTerm):
    """Processes normalized steering and throttle commands for the Leatherback robot."""

    cfg: LeatherbackDriveActionCfg
    _asset: Articulation

    def __init__(self, cfg: LeatherbackDriveActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        # Find steering knuckle joint indices
        self._steer_joint_ids, self._steer_joint_names = self._asset.find_joints("Knuckle__Upright__Front_.*")
        # Find left drive wheel joint indices
        self._left_wheel_ids, self._left_wheel_names = self._asset.find_joints("Wheel__.*_Left")
        # Find right drive wheel joint indices
        self._right_wheel_ids, self._right_wheel_names = self._asset.find_joints("Wheel__.*_Right")

        # Buffers for target commands
        self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._current_steer = torch.zeros(self.num_envs, 1, device=self.device)

        self._steer_targets = torch.zeros(self.num_envs, len(self._steer_joint_ids), device=self.device)
        self._left_wheel_targets = torch.zeros(self.num_envs, len(self._left_wheel_ids), device=self.device)
        self._right_wheel_targets = torch.zeros(self.num_envs, len(self._right_wheel_ids), device=self.device)

    @property
    def action_dim(self) -> int:
        return 2  # [steering, throttle]

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = torch.clamp(actions, -1.0, 1.0)

        steer_action = self._processed_actions[:, 0:1]    # [-1.0, 1.0]
        throttle_action = self._processed_actions[:, 1:2] # [-1.0, 1.0]

        # Steering angle target in radians with servo rate limiting to prevent high-frequency wiggling
        target_steer = steer_action * self.cfg.max_steer_angle
        max_delta = self.cfg.max_steer_rate * self._env.step_dt
        steer_delta = torch.clamp(target_steer - self._current_steer, -max_delta, max_delta)
        self._current_steer = self._current_steer + steer_delta
        self._steer_targets = self._current_steer.repeat(1, len(self._steer_joint_ids))

        # Forward speed mapping: [-1, 1] -> [min_speed, max_speed]
        speed = 0.5 * (throttle_action + 1.0) * (self.cfg.max_speed - self.cfg.min_speed) + self.cfg.min_speed
        target_omega = speed / self.cfg.wheel_radius  # (num_envs, 1) rad/s

        # In official NVIDIA Leatherback, both left and right wheels rotate with positive velocity for forward motion
        self._left_wheel_targets = target_omega.repeat(1, len(self._left_wheel_ids))
        self._right_wheel_targets = target_omega.repeat(1, len(self._right_wheel_ids))

    def apply_actions(self):
        # Apply front wheel steering angles (position control)
        self._asset.set_joint_position_target_index(target=self._steer_targets, joint_ids=self._steer_joint_ids)
        # Apply 4-wheel drive velocities (velocity control)
        self._asset.set_joint_velocity_target_index(target=self._left_wheel_targets, joint_ids=self._left_wheel_ids)
        self._asset.set_joint_velocity_target_index(target=self._right_wheel_targets, joint_ids=self._right_wheel_ids)

    def reset(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._current_steer.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            self._current_steer[env_ids] = 0.0


@configclass
class LeatherbackDriveActionCfg(ActionTermCfg):
    """Configuration for Leatherback continuous Ackermann steering and 4-wheel drive."""

    class_type: type[ActionTerm] = LeatherbackDriveAction

    max_steer_angle: float = 0.45
    """Maximum front knuckle steering angle in radians (default: ~25.8 deg)."""

    max_steer_rate: float = 6.0
    """Maximum steering slew rate in rad/s (~340 deg/s, eliminates servo jitter/wobble)."""

    max_speed: float = 2.5
    """Maximum driving speed in m/s (calibrated for stable 4-wheel ground contact without pitch-up)."""

    min_speed: float = 0.0
    """Minimum driving speed in m/s (0.0 prevents moving backwards)."""

    wheel_radius: float = 0.052
    """Effective wheel radius in meters."""
