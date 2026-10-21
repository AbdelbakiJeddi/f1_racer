"""Event terms for the Leatherback racing car grouped by class."""

from __future__ import annotations

from typing import TYPE_CHECKING
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz
from leatherback_racer.track import TrackOracle

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedEnv


class CarEvents:
    """Encapsulates reset and randomization event terms for the vehicle."""

    @staticmethod
    def reset_pose_on_track(
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | slice,
        random_start_point: bool = False,
        lateral_range: float = 0.0,
        yaw_noise_range: float = 0.0,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> None:
        """Resets car root position and orientation onto the track starting point or random waypoint."""
        asset: Articulation = env.scene[asset_cfg.name]
        oracle = TrackOracle.get_instance(device=str(asset.device))

        if isinstance(env_ids, slice):
            num_resets = env.num_envs
            actual_env_ids = torch.arange(num_resets, device=asset.device)
        else:
            num_resets = env_ids.shape[0]
            actual_env_ids = env_ids

        if num_resets == 0:
            return

        # Choose track point index (index 0 is the start grid line)
        if random_start_point:
            idx = torch.randint(0, oracle.num_points, (num_resets,), device=asset.device)
        else:
            idx = torch.zeros(num_resets, dtype=torch.long, device=asset.device)

        # Reference centerline position and normal
        c_pt = oracle.centerline[idx]    # (num_resets, 2)
        c_norm = oracle.normals[idx]     # (num_resets, 2)
        c_yaw = oracle.tangent_yaws[idx] # (num_resets,)

        # Lateral offset along normal vector
        if lateral_range > 0.0:
            lat_offset = torch.empty(num_resets, 1, device=asset.device).uniform_(-lateral_range, lateral_range)
        else:
            lat_offset = torch.zeros((num_resets, 1), device=asset.device)
        car_pos_2d = c_pt + lat_offset * c_norm  # (num_resets, 2)

        # World position: add local env origins + chassis road clearance z=0.035m
        env_origins_2d = env.scene.env_origins[actual_env_ids, :2]
        pos_x = car_pos_2d[:, 0] + env_origins_2d[:, 0]
        pos_y = car_pos_2d[:, 1] + env_origins_2d[:, 1]
        pos_z = torch.full((num_resets,), 0.035, device=asset.device)
        positions = torch.stack([pos_x, pos_y, pos_z], dim=-1)

        # Heading alignment
        if yaw_noise_range > 0.0:
            yaw_noise = torch.empty(num_resets, device=asset.device).uniform_(-yaw_noise_range, yaw_noise_range)
            total_yaw = c_yaw + yaw_noise
        else:
            total_yaw = c_yaw

        # Quaternion (roll=0, pitch=0, yaw=total_yaw)
        zeros = torch.zeros(num_resets, device=asset.device)
        orientations = quat_from_euler_xyz(zeros, zeros, total_yaw)

        # Write root pose to physics simulation
        root_pose = torch.cat([positions, orientations], dim=-1)
        asset.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=actual_env_ids)

        # Zero root linear and angular velocities
        zero_vel = torch.zeros((num_resets, 6), device=asset.device)
        asset.write_root_velocity_to_sim_index(root_velocity=zero_vel, env_ids=actual_env_ids)


# Direct functional aliases for compatibility
reset_car_pose_on_track = CarEvents.reset_pose_on_track
