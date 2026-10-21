"""MDP terms for the F1 Racer task grouped into domain classes."""

from isaaclab.envs.mdp import *  # noqa: F401, F403

# Core MDP Classes
from .actions import LeatherbackDriveAction, LeatherbackDriveActionCfg
from .observations import CarObservations
from .rewards import CarRewards
from .terminations import CarTerminations
from .events import CarEvents
from .track_oracle import TrackOracle

# Functional aliases for direct callable dispatch
from .observations import (  # noqa: F401
    car_steering_angle,
    car_forward_speed,
    car_proprioception,
    privileged_track_state,
    vision_track_state,
    get_vision_detector,
)
from .rewards import (  # noqa: F401
    track_centering,
    track_forward_progress,
    track_heading_alignment,
    boundary_violation_penalty,
    steering_smoothness_penalty,
    steering_rate_penalty,
)
from .terminations import (  # noqa: F401
    off_track,
    vehicle_rollover,
)
from .events import (  # noqa: F401
    reset_car_pose_on_track,
)

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
    "off_track",
    "vehicle_rollover",
    "reset_car_pose_on_track",
]
