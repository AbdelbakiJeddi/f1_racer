"""Observation terms for the Leatherback racing car grouped by class."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from leatherback_racer.track import TrackOracle
from leatherback_racer.vision import BatchedVisionTrackDetector

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv


class CarObservations:
    """Encapsulates all observation computation methods for the vehicle."""

    _detector_cache: dict[str, BatchedVisionTrackDetector] = {}

    @classmethod
    def get_detector(cls, device: str, img_h: int = 128, img_w: int = 128) -> BatchedVisionTrackDetector:
        """Returns a cached GPU instance of BatchedVisionTrackDetector."""
        key = f"{device}_{img_h}_{img_w}"
        if key not in cls._detector_cache:
            cls._detector_cache[key] = BatchedVisionTrackDetector(img_h=img_h, img_w=img_w).to(device)
        return cls._detector_cache[key]

    @staticmethod
    def steering_angle(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Returns front knuckle steering angle (N, 1)."""
        asset: Articulation = env.scene[asset_cfg.name]
        joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        return torch.mean(joint_pos, dim=-1, keepdim=True)

    @staticmethod
    def forward_speed(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Returns car forward velocity in chassis local coordinate frame (N, 1)."""
        asset: Articulation = env.scene[asset_cfg.name]
        return asset.data.root_lin_vel_b.torch[:, 0:1]

    @staticmethod
    def proprioception(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        """Returns combined proprioceptive features: [steering_angle, forward_speed] (N, 2)."""
        asset: Articulation = env.scene[asset_cfg.name]
        steer = torch.mean(asset.data.joint_pos.torch[:, asset_cfg.joint_ids], dim=-1, keepdim=True)
        speed = asset.data.root_lin_vel_b.torch[:, 0:1]
        return torch.cat([steer, speed], dim=-1)

    @staticmethod
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

        pos_2d = asset.data.root_pos_w.torch[:, :2] - env.scene.env_origins[:, :2]
        _, _, yaw = euler_xyz_from_quat(asset.data.root_quat_w.torch)

        query = oracle.query(pos_2d, yaw)
        lat_err = query["lateral_error"]  # (N, 1)
        head_err = query["heading_error"]  # (N, 1)
        tan_vec = query["tangent"]  # (N, 2)

        vx = asset.data.root_lin_vel_b.torch[:, 0:1]
        vy = asset.data.root_lin_vel_b.torch[:, 1:2]
        wz = asset.data.root_ang_vel_b.torch[:, 2:3]

        steer = torch.mean(asset.data.joint_pos.torch[:, asset_cfg.joint_ids], dim=-1, keepdim=True)

        vel_world_2d = asset.data.root_lin_vel_w.torch[:, :2]
        v_tangent = torch.sum(vel_world_2d * tan_vec, dim=-1, keepdim=True)

        half_width = oracle.track_width / 2.0
        norm_lat_err = lat_err / half_width

        return torch.cat([lat_err, head_err, vx, vy, wz, steer, v_tangent, norm_lat_err], dim=-1)

    @staticmethod
    def vision_track_state(
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Computes vision-derived track errors and combines with vehicle kinematics.

        Returns:
            Tensor of shape (N, 7):
            - e_lat: normalized lateral displacement [-1, 1]
            - e_heading: heading angular error in radians
            - confidence: line detection confidence [0, 1]
            - vx: longitudinal velocity in body frame
            - vy: lateral velocity in body frame
            - wz: yaw rate in body frame
            - steer: knuckle joint position
        """
        sensor = env.scene[sensor_cfg.name]
        rgb_images = sensor.data.output["rgb"]

        # Support H, W from image shape
        img_h = rgb_images.shape[1] if rgb_images.dim() == 4 and rgb_images.shape[-1] in (3, 4) else rgb_images.shape[2]
        img_w = rgb_images.shape[2] if rgb_images.dim() == 4 and rgb_images.shape[-1] in (3, 4) else rgb_images.shape[3]

        detector = CarObservations.get_detector(device=str(rgb_images.device), img_h=img_h, img_w=img_w)
        features, _ = detector(rgb_images)  # (N, 3): [e_lat, e_heading, confidence]

        asset: Articulation = env.scene[asset_cfg.name]
        vx = asset.data.root_lin_vel_b.torch[:, 0:1]
        vy = asset.data.root_lin_vel_b.torch[:, 1:2]
        wz = asset.data.root_ang_vel_b.torch[:, 2:3]
        steer = torch.mean(asset.data.joint_pos.torch[:, asset_cfg.joint_ids], dim=-1, keepdim=True)

        return torch.cat([features, vx, vy, wz, steer], dim=-1)


# Direct functional aliases for compatibility
car_steering_angle = CarObservations.steering_angle
car_forward_speed = CarObservations.forward_speed
car_proprioception = CarObservations.proprioception
privileged_track_state = CarObservations.privileged_track_state
vision_track_state = CarObservations.vision_track_state
get_vision_detector = CarObservations.get_detector
