# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Environment configuration for the Leatherback F1 racing car task."""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_tasks.utils import PresetCfg
from isaaclab_visualizers.kit import KitVisualizerCfg

from f1_racer_rl.assets import F1_RACER_RL_ASSETS_DIR, LEATHERBACK_CFG
from ... import mdp

##
# Physics Presets
##


@configclass
class F1RacerPhysicsCfg(PresetCfg):
    """Physics presets for the F1 Racer environment."""

    isaacsim_physx: PhysxCfg = PhysxCfg()
    ovphysx: OvPhysxCfg = OvPhysxCfg()
    physx: PhysxAutoCfg = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    default: PhysxAutoCfg = physx


##
# Scene Definition
##


@configclass
class F1RacerSceneCfg(InteractiveSceneCfg):
    """Configuration for the F1 Racer racing arena scene."""

    # Dark tarmac ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(500.0, 500.0),
            color=(0.12, 0.12, 0.12),
        ),
    )

    # Standalone racing corridor boundary ribbons
    track = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Track",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(F1_RACER_RL_ASSETS_DIR / "track.usd"),
            visible=True,
        ),
    )

    # Leatherback racing car articulation
    robot: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # Lighting: balanced dome ambient + distant sunlight
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.85, 0.9, 1.0), intensity=2500.0),
    )

    sun_light = AssetBaseCfg(
        prim_path="/World/SunLight",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 0.98, 0.92),
            intensity=3000.0,
            angle=0.53,
        ),
    )


##
# MDP Settings
##


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # Continuous Ackermann steering + 4-wheel drive velocity targets
    drive = mdp.LeatherbackDriveActionCfg(
        asset_name="robot",
        max_steer_angle=0.45,
        max_speed=3.5,
        min_speed=0.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for state-based RL."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy (Actor: state vector containing dynamics and track errors)."""

        state = ObsTerm(
            func=mdp.privileged_track_state,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["Knuckle__Upright__Front_.*"]
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic."""

        privileged_state = ObsTerm(
            func=mdp.privileged_track_state,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=["Knuckle__Upright__Front_.*"]
                )
            },
        )

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for reset events and domain randomization."""

    # Reset car root pose onto the track centerline
    reset_pose = EventTerm(
        func=mdp.reset_car_pose_on_track,
        mode="reset",
        params={
            "random_start_point": False,
            "lateral_range": 0.0,
            "yaw_noise_range": 0.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    # Reset knuckle steering and wheel rotational velocities
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for path centering and forward racing."""

    # (1) Primary task: Gaussian reward for remaining centered between boundary lines
    centering = RewTerm(func=mdp.track_centering, weight=1.5, params={"std": 0.25})

    # (2) Progress: forward velocity along track tangent
    progress = RewTerm(func=mdp.track_forward_progress, weight=1.0)

    # (3) Heading: alignment of car with track tangent direction
    heading = RewTerm(func=mdp.track_heading_alignment, weight=0.5)

    # (4) Penalty for straying close to or outside the boundary ribbons
    boundary = RewTerm(
        func=mdp.boundary_violation_penalty,
        weight=-2.0,
        params={"margin_threshold": 0.45},
    )

    # (5) Action smoothness: penalize large steering deflections
    steering_smoothness = RewTerm(
        func=mdp.steering_smoothness_penalty,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=["Knuckle__Upright__Front_.*"]
            )
        },
    )

    # (6) Alive bonus: reward for surviving without crashing
    alive = RewTerm(func=mdp.is_alive, weight=0.2)


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    # (1) Episode timeout
    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # (2) Boundary crash / off-track termination
    off_track = DoneTerm(func=mdp.off_track, params={"max_lateral_distance": 0.75})

    # (3) Vehicle rollover or flip
    rollover = DoneTerm(func=mdp.vehicle_rollover, params={"max_tilt_rad": 0.785})


##
# Environment Configuration
##


@configclass
class F1RacerEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Leatherback F1 Racer environment."""

    # Simulation settings
    sim: sim_utils.SimulationCfg = sim_utils.SimulationCfg(
        dt=1.0 / 120.0,
        render_interval=4,
        physics=F1RacerPhysicsCfg(),
    )

    # Scene settings (isolated parallel environments)
    scene: F1RacerSceneCfg = F1RacerSceneCfg(num_envs=32, env_spacing=40.0)

    # MDP specifications
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self) -> None:
        """Post initialization."""
        # 120 Hz physics with decimation=4 -> 30 Hz control
        self.decimation = 4
        self.episode_length_s = 25.0

        # Free overview camera letting the user look at all environments and freely navigate the viewport
        self.sim.default_visualizer_cfg = KitVisualizerCfg(
            eye=(25.0, 15.0, 12.0),
            lookat=(15.0, 5.0, 0.0),
            origin_type="world",
        )
        self.viewer.origin_type = "world"
        self.viewer.eye = (25.0, 15.0, 12.0)
        self.viewer.lookat = (15.0, 5.0, 0.0)
