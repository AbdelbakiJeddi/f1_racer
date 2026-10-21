# F1 Racer: Leatherback Autonomous Racing in Isaac Lab

Autonomous trajectory tracking and high-speed racing for the **NVIDIA Leatherback** robot in **Isaac Sim / Isaac Lab 3.0**.

---

## Key Features

- **Official Leatherback Dynamics:** 26-joint multi-link articulation featuring double-wishbone suspension linkages, tuned shock absorbers, front steering knuckles, and 4-wheel drive.
- **Isaac Lab Extension (`f1_racer_rl`):** Modular, manager-based reinforcement learning environment registered under Gymnasium (`F1RacerRl-F1-Racer-F1-Car`).
- **Vision-Only Deployable Policy:** The deployed actor policy never receives ground-truth coordinates, GPS, LiDAR, or simulation oracle states. Perception is purely vision-based from a single onboard nose camera.
- **GPU Classical Computer Vision:** `TrackDetector` runs fully vectorized on CUDA tensors ($R+G-2B$ yellow boundary segmentation, near/far horizon ROIs, centroid tracking) yielding lateral error, heading error, and confidence with zero inference lag.
- **Asymmetric Actor-Critic (PPO):** 7D vision-kinematic actor state $[e_\text{lat}, e_\text{heading}, \text{conf}, v_x, v_y, \omega_z, \delta_\text{steer}]$ deployed at 30 Hz; privileged 8D ground-truth state used exclusively by the simulation Critic.
- **Procedural 3D Track & Spatial Oracle:** Continuous spline-based track oracle computing exact Frenet frame projection ($e_\text{lat}$, $e_\text{heading}$) and lookahead guidance vectors.

---

## Command Reference & Quick Start

Always run commands from the project root using the dedicated Isaac Lab 3.0 Python interpreter:
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python <script_path> [options]
```

### 1. Training (RSL-RL PPO)

- **Fast Headless Training (Recommended for Maximum FPS):**
  ```bash
  /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 64 --max_iterations 500
  ```
- **Live Visualized Training (Single or Multi-Env with 3D GUI):**
  ```bash
  /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 1 --viz kit
  ```
- **Resume Training from Checkpoint:**
  ```bash
  /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 64 --checkpoint latest
  ```

### 2. Policy Evaluation & Playback

- **Single Lap Challenge on Unseen Procedural Track (with Live Vision HUD):**
  Generates a brand new procedural track and evaluates the policy completing a full lap with live 3D chase camera and performance scorecard:
  ```bash
  /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/test_single_lap.py --track_type organic --seed 123 --viz kit
  ```
  *Options:*
  - `--track_type`: `organic`, `wavy_circle`, or `circuit`.
  - `--seed`: Procedural seed for track geometry (e.g. `123`, `42`).
  - `--chase_cam`: Locks camera in 3rd-person chase mode behind the car (by default, free camera).
  - `--show_vision` / `--no_show_vision`: Toggle the real-time OpenCV HUD window.

- **Playback on Default Training Circuit:**
  ```bash
  /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/play.py --checkpoint latest --viz kit
  ```

### 3. Interactive Manual Driving (Smoke Test)

Drive the car interactively with the keyboard to verify suspension, grip, and steering while viewing real-time vision feedback:
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/teleop_car.py
```
*Controls (active inside 3D viewport):*
- `W` / `Up Arrow`: Accelerate forward smoothly
- `S` / `Down Arrow`: Brake / Reverse
- `A` / `Left Arrow`: Steer Left
- `D` / `Right Arrow`: Steer Right
- `SPACE`: Handbrake (instant stop)
- `R`: Reset car back to starting line

### 4. Computer Vision Diagnostics

Runs camera rendering, GPU line segmentation, dual-ROI error estimation, and compares vision against ground truth:
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/test_vision_pipeline_live.py --steps 30
```
Generates a 4-panel diagnostic dashboard saved to `outputs/vision_pipeline_dashboard.png`.

### 5. TensorBoard Monitoring
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/tensorboard --logdir logs/rsl_rl/f1_racer_leatherback --port 6006
```

---

## Project Structure

For detailed class taxonomies and system design, see **[ARCHITECTURE.md](ARCHITECTURE.md)**.

```text
f1_racer/
├── README.md                       # Main project documentation & CLI reference
├── ARCHITECTURE.md                 # System architecture, class taxonomy & design specs
├── pyproject.toml                  # Python package configuration (leatherback-racer)
│
├── src/leatherback_racer/          # Core Python library
│   ├── vision/                     # Classes: TrackDetector, TrackVisionHUD
│   ├── track/                      # Classes: ProceduralTrack, TrackConfig, TrackOracle
│   ├── teleop/                     # Class: KeyboardTeleop
│   └── evaluation/                 # Classes: CheckpointLoader, SingleLapEvaluator, VisionPipelineTester
│
├── f1_racer_rl/                    # Isaac Lab Extension (Gymnasium task registration)
│   ├── pyproject.toml              # Extension package setup
│   └── src/f1_racer_rl/
│       ├── assets/                 # Articulation definition & active track USD/NPZ
│       │   ├── leatherback.py      # Articulation & suspension actuator configuration
│       │   └── data/               # track.usd, track_data.npz, circuit backups
│       └── tasks/f1_racer/
│           ├── config/f1_car/      # Environment & PPO hyperparameter configs
│           └── mdp/                # Class-grouped MDP terms (Observations, Rewards, Terminations, Events)
│
├── scripts/                        # Modular CLI entry points
│   ├── train.py                    # Multi-environment PPO training launcher
│   ├── play.py                     # Policy playback & evaluation
│   ├── teleop_car.py               # Keyboard driving smoke test + live Vision HUD
│   ├── test_single_lap.py          # Procedural single lap evaluation + live Vision HUD
│   └── test_vision_pipeline_live.py# Live vision diagnostics & dashboard generator
│
├── outputs/                        # Test artifacts, dashboards & procedural tracks
└── logs/                           # Training runs, checkpoints, and TensorBoard logs
```
