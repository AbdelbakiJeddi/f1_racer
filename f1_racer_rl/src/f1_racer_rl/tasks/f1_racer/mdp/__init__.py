"""MDP terms for the F1 Racer task grouped into domain classes."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Core MDP Classes
from .actions import LeatherbackDriveAction, LeatherbackDriveActionCfg
from .events import (  # noqa: F401
    CarEvents,
    reset_car_pose_on_track,
)

# Functional aliases for direct callable dispatch
from .observations import (  # noqa: F401
    CarObservations,
    car_forward_speed,
    car_proprioception,
    car_steering_angle,
    get_vision_detector,
    privileged_track_state,
    vision_track_state,
)
from .rewards import (  # noqa: F401
    CarRewards,
    boundary_violation_penalty,
    lap_completion_bonus,
    slow_speed_penalty,
    steering_rate_penalty,
    steering_smoothness_penalty,
    track_centering,
    track_forward_progress,
    track_heading_alignment,
)
from .terminations import (  # noqa: F401
    CarTerminations,
    lap_completed,
    off_track,
    vehicle_rollover,
)
from .track_oracle import TrackOracle

__all__ = [
    # Classes
    "LeatherbackDriveAction",
    "LeatherbackDriveActionCfg",
    "CarObservations",
    "CarRewards",
    "CarTerminations",
    "CarEvents",
    "TrackOracle",
    # Functions
    "car_steering_angle",
    "car_forward_speed",
    "car_proprioception",
    "privileged_track_state",
    "vision_track_state",
    "get_vision_detector",
    "track_centering",
    "track_forward_progress",
    "track_heading_alignment",
    "boundary_violation_penalty",
    "steering_smoothness_penalty",
    "steering_rate_penalty",
    "slow_speed_penalty",
    "lap_completion_bonus",
    "off_track",
    "vehicle_rollover",
    "lap_completed",
    "reset_car_pose_on_track",
]
