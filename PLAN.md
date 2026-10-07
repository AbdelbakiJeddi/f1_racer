# Leatherback Path Tracking Project Plan (Vision-Only RL in Isaac Lab 3.0)

## 1. Executive Summary

This project implements an autonomous path tracking and lane-keeping system for the **Leatherback** Ackermann-steering robot in **NVIDIA Isaac Lab 3.0**. 

The task requires the vehicle to navigate through a corridor defined by **two parallel boundary lines drawn on the ground**. The robot must stay strictly between the lines while maximizing forward speed. 

Crucially, the control policy operates under a **vision-only** observation paradigm:
- **No LiDAR**.
- **No ground-truth GPS, odometry or world-coordinate error inputs given to the policy.**
- The actor policy observes **only onboard camera images** (front/downward-facing RGB or grayscale) plus vehicle proprioception (forward velocity, steering angle).
- Training leverages **Asymmetric Actor-Critic (PPO)** in Isaac Lab, where privileged simulator ground-truth states are used only by the Critic and Reward Oracle during simulation, then discarded for real-time deployment.

---

## 2. Problem Formulation & Vision-Based Sensing

### 2.1 The Two-Line Track Boundary
- The corridor is defined by a reference centerline `C(s)` and track width `W` (e.g., 1.0 m to 1.5 m wide).
- **Left boundary line:** High-contrast line drawn on ground at distance `+W/2` from centerline.
- **Right boundary line:** High-contrast line drawn on ground at distance `-W/2` from centerline.
- The boundary lines are rendered as high-contrast visual features on the ground plane (e.g., bright white or yellow stripes against dark tarmac/surface).

### 2.2 How Lateral & Heading Errors are Encoded in the Camera Frame
The policy does not need explicit mathematical coordinates `(e_lat, e_heading)` because these physical states are directly manifested in the camera pixel plane:
- **Lateral Offset (Off-Center):**
  - Perfectly centered: Left line appears on the left half of the image; right line appears symmetrically on the right half.
  - Car drifts right: Both lines translate leftward across the pixel matrix.
  - Car drifts left: Both lines translate rightward across the pixel matrix.
- **Heading Error (Yaw Misalignment):**
  - Car aligned with path: The two boundary lines converge toward the vertical midline vanishing point.
  - Car yawed diagonally: The lines tilt diagonally across the frame.
- **Look-Ahead Curvature:**
  - Approaching left/right turns appear as curves in the lines in the upper rows of the image.

---

## 3. Asymmetric Actor-Critic RL Architecture

```
+------------------------------------------------------------------------------------+
|                         SIMULATION PHASE (DURING TRAINING)                         |
|                                                                                    |
|  [Simulation Physics Engine & Ground Truth Oracle]                                 |
|  - Exact vehicle 3D pose, line positions, wheel slip                               |
|  - Computes e_lat, e_heading, line clearance                                       |
|               |                                           |                        |
|               v                                           v                        |
|        +---------------+                           +---------------+               |
|        | Reward Engine |                           |    CRITIC     |               |
|        | & Reset Logic |                           | (State Value) |               |
|        +---------------+                           +---------------+               |
+-----------------------------------------------------------|------------------------+
                                                            | Critic guides PPO loss
                                                            v
+------------------------------------------------------------------------------------+
|                    DEPLOYABLE ACTOR POLICY (VISION-ONLY AGENT)                     |
|                                                                                    |
|   +------------------------------------+                                           |
|   | Onboard TiledCamera                |                                           |
|   | (RGB / Grayscale, 84x84 or 128x128)|                                           |
|   +------------------------------------+                                           |
|                     |                                                              |
|                     v                                                              |
|        +-------------------------+                                                 |
|        | Nature-CNN / ResNet Enc |                                                 |
|        +-------------------------+                                                 |
|                     |                                                              |
|                     +<--- [ Proprioception: speed v_x, steer angle delta ]         |
|                     v                                                              |
|        +-------------------------+                                                 |
|        |      MLP Policy         |                                                 |
|        +-------------------------+                                                 |
|                     |                                                              |
|                     v                                                              |
|        +-------------------------+                                                 |
|        | Actions:                |                                                 |
|        | - Steer command a_steer |                                                 |
|        | - Speed / Throttle a_v  |                                                 |
|        +-------------------------+                                                 |
+------------------------------------------------------------------------------------+
```

### 3.1 Actor Network (Deployed Policy)
- **Inputs:**
  1. Camera Image Tensor: `(C, H, W)` with `H = 84, W = 84` (or `128 x 128`), stacked across 3 consecutive frames to capture optical flow and vehicle velocity.
  2. Proprioceptive State: `[current_steering_angle, forward_speed_v_x]` (2 floats).
- **Encoder:** 3-layer Convolutional Neural Network (Nature-CNN or MobileNet block) projecting to latent embedding `z_vision in R^128`.
- **Policy Head:** MLP `[128, 64] -> Action Mean & Std`.
- **Outputs (Continuous 2D Actions in [-1.0, 1.0]):**
  - `a_steer`: Steering angle command mapped to `[-30 deg, +30 deg]`.
  - `a_throttle`: Motor throttle / target velocity mapped to `[0.0 m/s, 3.5 m/s]`.

### 3.2 Critic Network (Value Estimator, Training Only)
- **Inputs (Privileged State):**
  - Lateral error `e_lat` (signed perpendicular distance from vehicle center to path centerline).
  - Heading error `e_heading` (relative angle between vehicle yaw and path tangent).
  - Perpendicular distance to left boundary line and right boundary line.
  - Linear velocity `(v_x, v_y)` and angular yaw rate `omega_z`.
  - Current steering angle and previous action step.
- **Benefit:** Providing privileged scalar states to the Critic drastically stabilizes PPO value function fitting and cuts training iterations by up to 10x compared to symmetric pixel-only critics. The Critic is discarded after training.

---

## 4. MDP Formulation

### 4.1 Reward Function (Evaluated by Simulation Oracle)

| Reward Term | Mathematical Formulation | Weight | Rationale |
|---|---|---|---|
| Track Centering | `exp(- (e_lat / sigma_lat)^2)` with `sigma_lat = 0.25 m` | +2.0 | Maximum reward when vehicle stays midway between both lines |
| Forward Progress | `v_x * cos(e_heading)` | +2.5 | Rewards positive velocity along the corridor direction |
| Heading Alignment | `cos(e_heading)` | +0.5 | Penalizes driving sideways or diagonally across the track |
| Alive Bonus | `+1.0` per step | +0.5 | Encourages survival and remaining inside the lines |
| Action Smoothness | `- ||action_t - action_{t-1}||^2` | -0.1 | Suppresses high-frequency steering chatter and oscillations |
| Line Proximity Penalty | `- (1.0 / (d_boundary + 0.05))` when `d_boundary < 0.15 m` | -1.0 | Pushes car away from edge lines before a breach occurs |
| Boundary Breach (Reset) | `-25.0` (terminal penalty) | -25.0 | Severe penalty when wheels or chassis cross either line |

### 4.2 Termination & Reset Logic
Episodes terminate and reset when any of the following occur:
1. **Line Crossing:** Either left or right wheels cross beyond the outer boundary of either line.
2. **Rollover / Tipping:** Vehicle pitch or roll exceeds 35 degrees.
3. **Stalled:** Vehicle forward velocity `< 0.05 m/s` for more than 100 consecutive environment steps.
4. **Episode Timeout:** 500 to 1000 control steps (10 to 20 seconds at 50 Hz).

---

## 5. Robot Hardware & Camera Configuration

### 5.1 Leatherback Robot Configuration
- **Chassis:** 4-wheel mobile platform with Ackermann front steering.
- **Front Axle:** Steering revolute joints (`joint_steer_fl`, `joint_steer_fr`), limits `[-30 deg, +30 deg]`, stiffness `500.0`, damping `30.0`.
- **Rear Axle:** Drive revolute joints (`joint_drive_rl`, `joint_drive_rr`), velocity/torque drive motors.
- **Wheels:** Rubber tire contact material with static friction `mu = 1.0`, dynamic friction `mu = 0.8`.

### 5.2 Onboard Camera Rig (TiledCamera)
- **Sensor:** `isaaclab.sensors.TiledCameraCfg` (GPU-vectorized rendering optimized for multi-environment RL).
- **Mount Prim:** Rigid body chassis link (`/Leatherback/chassis`).
- **Mount Offset:**
  - Position: `(x = +0.22 m, y = 0.0 m, z = +0.16 m)` (forward on bumper/hood).
  - Rotation: Tilted downward pitch by 25 to 30 degrees (`rotation = (0.0, 0.25, 0.0)` in Euler/quaternion).
- **Field of View:** Horizontal FOV 75 to 85 degrees.
- **Resolution:** `84 x 84` or `128 x 128` pixels (RGB or Grayscale).
- **Visible Ground Range:** Line markers visible from 0.25 m to ~3.5 m in front of the car.

---

## 6. Project Directory Structure

```
f1_racer/
├── PLAN.md                                    # This project specification
├── pyproject.toml                             # uv build & IsaacLab-3.0 environment configuration
├── README.md                                  # Quickstart and execution commands
├── isaacsim/
│   └── scenes/
│       ├── generate_f1_car.py                 # USD generator script
│       └── view_f1_car.py                     # Asset GUI visualizer
├── scripts/
│   ├── train.py                               # RSL-RL training launcher
│   ├── play.py                                # Checkpoint evaluation and visualization
│   └── test_camera_env.py                     # Zero-agent camera feed test script
└── src/
    └── leatherback_racer/
        ├── __init__.py
        ├── assets/
        │   ├── __init__.py
        │   └── leatherback.py                 # ArticulationCfg + TiledCameraCfg
        └── tasks/
            └── line_track/
                ├── __init__.py
                ├── mdp/
                │   ├── __init__.py
                │   ├── observations.py        # Camera stacking & proprioception terms
                │   ├── rewards.py             # Oracle centering & progress rewards
                │   └── terminations.py        # Boundary breach detection
                └── config/
                    ├── __init__.py
                    ├── line_track_env_cfg.py  # ManagerBasedRLEnvCfg (Asymmetric Actor-Critic)
                    └── agents/
                        ├── __init__.py
                        └── rsl_rl_ppo_cfg.py  # CNN-PPO hyperparameters
```

---

## 7. Phased Implementation Roadmap

### Phase 1: Environment & Asset Setup
- [ ] Create `pyproject.toml` linking `IsaacLab-3.0` and dependencies (`uv sync`).
- [ ] Define `Leatherback` robot `ArticulationCfg` with Ackermann steering and `TiledCameraCfg`.
- [ ] Verify robot spawn and downward camera view in Isaac Sim with a test script.

### Phase 2: Ground Line Scene & Oracle Math
- [ ] Create ground plane with procedural twin boundary lines (high-contrast ribbons).
- [ ] Implement GPU tensor math for path arc-length, centerline error `e_lat`, and line clearance.
- [ ] Validate camera image feed capturing the two boundary lines clearly.

### Phase 3: MDP Implementation & Environment Registration
- [ ] Build camera observation manager (frame stacking, normalization).
- [ ] Build privileged critic observation manager (oracle error states).
- [ ] Implement reward functions (centering, progress, line proximity penalty).
- [ ] Implement termination managers (line crossing, rollover).
- [ ] Register task `Isaac-Leatherback-LineTrack-v0`.

### Phase 4: Training with RSL-RL (Asymmetric PPO)
- [ ] Configure CNN Actor and MLP Critic in `rsl_rl_ppo_cfg.py`.
- [ ] Launch training across 2,048 environments in headless mode.
- [ ] Implement curriculum: begin with wide corridor (e.g. 1.8 m) and straight paths, narrowing to 1.1 m and introducing curves.
- [ ] Monitor TensorBoard metrics (reward, episode length, tracking accuracy).

### Phase 5: Evaluation & Real-Time Visualization
- [ ] Run `play.py` in GUI mode to inspect the policy driving between the lines.
- [ ] Log performance metrics: boundary compliance percentage, average lap speed, lateral deviation.
