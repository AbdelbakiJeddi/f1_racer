# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Leatherback racing car articulation."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from . import F1_RACER_RL_ASSETS_DIR

LEATHERBACK_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=str(F1_RACER_RL_ASSETS_DIR / "leatherback.usd"),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            max_linear_velocity=20.0,
            max_angular_velocity=20.0,
            max_depenetration_velocity=1.0,
            enable_gyroscopic_forces=True,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(17.031, 3.923, 0.035),
        rot=(0.0, 0.0, 0.6656, 0.7463),
        joint_pos={
            "Shock__Rear_Right": -0.01,
            "Shock__Rear_Left": -0.01,
            "Shock__Front_Right": 0.01,
            "Shock__Front_Left": 0.01,
            "Wheel__.*": 0.0,
            "Knuckle__.*": 0.0,
            "Chassis__.*": 0.0,
            "Upright__.*": 0.0,
            "Shock__Arm__.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "suspension": ImplicitActuatorCfg(
            joint_names_expr=["Shock__Rear_.*", "Shock__Front_.*"],
            stiffness=4000.0,
            damping=300.0,
        ),
        "steering": ImplicitActuatorCfg(
            joint_names_expr=["Knuckle__Upright__Front_.*"],
            effort_limit=6.0,
            velocity_limit=8.0,
            stiffness=600.0,
            damping=40.0,
        ),
        "throttle": ImplicitActuatorCfg(
            joint_names_expr=["Wheel__.*"],
            effort_limit=2.5,
            velocity_limit=100.0,
            stiffness=0.0,
            damping=0.5,
        ),
        "linkages": ImplicitActuatorCfg(
            joint_names_expr=["(Chassis__.*|Upright__.*|Shock__Arm__.*)"],
            stiffness=0.0,
            damping=5.0,
        ),
    },
)
