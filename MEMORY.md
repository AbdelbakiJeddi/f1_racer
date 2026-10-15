# Leatherback F1 Racer - Project Memory & System Architecture

## 1. Project Overview & Philosophy
- **Objective:** Autonomous lane-keeping and trajectory tracking for the **NVIDIA Leatherback** Ackermann-steering robot in **Isaac Lab 3.0**.
- **Task Corridor:** A procedural corridor bounded by two parallel high-contrast yellow lines ($1.1\text{ m}$ track width) on dark asphalt. The vehicle must navigate smoothly between the lines while maximizing speed and stability.
- **Vision-Only Deployable Policy:** The deployed actor policy never receives ground-truth coordinates, GPS, LiDAR, or simulation oracle states. Perception is purely vision-based from a single onboard nose camera.
- **Perception Architecture (GPU Classical CV vs CNN):**
  - All end-to-end CNN components (frame stacking, convolutional backbones, visual latent encoders) have been intentionally eliminated.
  - In their place, a GPU-vectorized classical computer vision module (`BatchedVisionTrackDetector`) extracts 3D geometric tracking features `[e_lat, e_heading, confidence]` directly from camera tensors in CUDA memory with zero latency.
  - The resulting 3D visual perception vector is concatenated with 4D vehicle proprioception `[v_x, v_y, \omega_z, \delta_{\text{steer}}]` to form a compact **7D state vector**.
  - A lightweight, sample-efficient Multi-Layer Perceptron (MLP) Actor controls the vehicle.

---

## 2. Asymmetric Actor-Critic (PPO) Specification
- **Actor (Policy Network - Deployed):**
  - **Input:** 7D vector `[e_lat, e_heading, confidence, v_x, v_y, omega_z, steer]`
  - **Architecture:** MLP `[256, 128, 64]` with ELU activations.
  - **Output:** 2D continuous action `[steer_cmd, throttle_cmd]` in range `[-1.0, 1.0]`.
- **Critic (Value Network - Simulation Training Only):**
  - **Input:** 8D privileged ground-truth state `[e_lat_gt, e_heading_gt, d_left, d_right, v_x, v_y, omega_z, steer]`.
  - **Architecture:** MLP `[128, 64, 32]` with ELU activations.
  - **Role:** Provides high-fidelity value estimation during PPO updates, drastically stabilizing policy gradient training without exposing privileged information to the deployed actor.

---

## 3. Robot Hardware & Camera Configuration
- **Platform:** NVIDIA Leatherback 4-wheel Ackermann-steering mobile robot.
  - Steering Joints: `joint_steer_fl`, `joint_steer_fr` ($\pm 30^\circ$ limit).
  - Drive Joints: `joint_drive_rl`, `joint_drive_rr`.
- **Onboard Sensor:** `TiledCameraCfg` mounted to `/Leatherback/chassis`.
  - **Position:** `(0.35, 0.0, 0.15)` m (nose bumper, $15\text{ cm}$ above ground).
  - **Orientation:** Pitch tilted downward $15^\circ$ (`rot = (0.0, 0.130526, 0.0, 0.991445)` in quaternion format `(w, x, y, z)`).
  - **Horizontal FOV:** $\approx 82^\circ$.
  - **Image Resolution:** $128 \times 128$ RGB.
- **Optical Geometry & Lookahead Horizon:**
  - Track width is $1.1\text{ m}$. At ground distances $< 0.63\text{ m}$, the corridor is wider than the camera's FOV; boundary lines enter the camera frame beyond $0.63\text{ m}$.
  - The boundary lines occupy image rows $40\%$ through $52\%$ ($0.7\text{ m}$ to $4.0\text{ m}$ ahead of the bumper).
  - **Near Horizon ROI:** Rows $46\% - 52\%$ (lookahead $0.7\text{ m} - 1.2\text{ m}$) $\rightarrow$ measures immediate lateral deviation $e_{\text{lat}}$.
  - **Far Horizon ROI:** Rows $40\% - 45\%$ (lookahead $1.5\text{ m} - 4.0\text{ m}$) $\rightarrow$ measures lookahead heading misalignment and upcoming turn curvature $e_{\text{heading}}$.

---

## 4. Perception Module: `BatchedVisionTrackDetector`
- **Location:** `src/leatherback_racer/vision/track_detector.py`
- **Processing Steps (Fully Vectorized on GPU):**
  1. **Color Segmentation:** Normalized chromaticity difference:
     $$\text{contrast} = \text{clamp}(R + G - 2B, 0, 1) > 0.25$$
     Effectively isolates yellow boundary lines under varying lighting and dark asphalt backgrounds.
  2. **Scanline Mass Integration:**
     - Computes horizontal intensity sums across Near and Far ROIs.
  3. **Dual-Boundary Centroid Tracking:**
     - Computes left boundary centroid $x_L$ ($u < W/2$) and right boundary centroid $x_R$ ($u \ge W/2$).
     - Midpoint: $x_c = (x_L + x_R) / 2$.
     - **Single-Line Fallback:** If one line exits the camera frame during aggressive turns, the detector switches dynamically to single-line offset tracking using calibrated expected half-spans ($0.65$ near, $0.45$ far).
  4. **Geometric State Normalization:**
     - $e_{\text{lat}} = (x_{c,\text{near}} - W/2) / (W/2) \in [-1.0, 1.0]$.
     - $e_{\text{heading}} = \text{atan2}(x_{c,\text{far}} - x_{c,\text{near}}, \Delta y_{\text{ROI}}) \in [-\pi, \pi]$.
     - $\text{confidence} \in [0.0, 1.0]$ based on detected mass threshold.

---

## 5. MDP Terms & Environment Setup
- **Gymnasium Task ID:** `F1RacerRl-F1-Racer-F1-Car`
- **Environment Class:** `isaaclab.envs.ManagerBasedRLEnv`
- **Config Entry Points:**
  - Env Config: `f1_racer_rl.tasks.f1_racer.config.f1_car.env_cfg:F1RacerEnvCfg`
  - Agent Config: `f1_racer_rl.tasks.f1_racer.config.f1_car.agents.rsl_rl_ppo_cfg:PPORunnerCfg`
- **MDP Observation Terms:**
  - `policy`: `mdp.vision_track_state` returning `(N, 7)` tensor.
  - `critic`: `mdp.privileged_track_state` returning `(N, 8)` tensor.
- **Reward Function Weights:**
  - Centering: `+2.0 * exp(-(e_lat / 0.25)^2)`
  - Forward Progress: `+2.5 * v_x * cos(e_heading)`
  - Heading Alignment: `+0.5 * cos(e_heading)`
  - Alive Bonus: `+0.5`
  - Steering Smoothness: `-0.1 * ||a_t - a_{t-1}||^2`
  - Line Proximity Penalty: `-1.0 / (d_boundary + 0.05)` when $d < 0.15\text{ m}$
  - Boundary Breach Penalty: `-25.0` (terminates episode)

---

## 6. Execution Environment & Dependencies
- **Isaac Lab Root:** `/home/abok-omen/IsaacLab-3.0`
- **Active Python Interpreter:** `/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python` (PyTorch 2.11.0+cu130, Isaac Sim 6.0 / Isaac Lab 3.0).
- **Package Installs:**
  - `leatherback-racer` installed in editable mode (`pip install -e .`).
  - `f1_racer_rl` registered with Isaac Lab and Gymnasium.

---

## 7. Verification & Diagnostic Tools
- `scripts/test_vision_pipeline.py`: Pure GPU PyTorch unit test verifying Near/Far lateral and heading calculations across 4 canonical road scenarios (Straight, Displaced, Left Curve, Right Curve).
- `scripts/test_vision_pipeline_live.py`: Full Isaac Lab simulation test initializing 3D physics, rendering nose camera frames, testing `BatchedVisionTrackDetector`, and generating the multi-panel diagnostic dashboard.
- `scripts/verify_pipeline.py`: Comprehensive integrity verification of USD assets, camera configurations, observation tensor shapes, and articulation properties.
- `outputs/vision_pipeline_dashboard.png`: Visual proof artifact verifying camera stream, yellow mask, scanline mass profiles, and state estimation HUD.
