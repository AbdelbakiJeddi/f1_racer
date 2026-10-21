# F1 Racer Architecture & Project Structure

This document outlines the reorganized, class-oriented architecture of the **Leatherback F1 Racer** project in NVIDIA Isaac Lab 3.0.

---

## 1. Directory Layout

```text
f1_racer/
├── README.md                      # Project overview, CLI commands & quickstart
├── ARCHITECTURE.md                # System design and class hierarchy (this file)
├── pyproject.toml                 # Package definition for 'leatherback-racer'
│
├── src/leatherback_racer/         # Core Python library (Vision, Track, Control, Evaluation)
│   ├── __init__.py                # Top-level class exports
│   │
│   ├── vision/                    # Perception & Visualization
│   │   ├── __init__.py            # Exports: TrackDetector, TrackVisionHUD
│   │   ├── track_detector.py      # Class: TrackDetector / BatchedVisionTrackDetector
│   │   └── hud_visualizer.py      # Class: TrackVisionHUD
│   │
│   ├── track/                     # Procedural Generation & Analytical Geometry
│   │   ├── __init__.py            # Exports: ProceduralTrack, TrackConfig, TrackOracle
│   │   ├── procedural.py          # Classes: ProceduralTrack, TrackConfig
│   │   └── oracle.py              # Class: TrackOracle
│   │
│   ├── teleop/                    # Manual Driving & Input Interfaces
│   │   ├── __init__.py            # Exports: KeyboardTeleop
│   │   └── keyboard.py            # Class: KeyboardTeleop
│   │
│   ├── evaluation/                # Checkpoint loading, evaluation, & metrics
│   │   ├── __init__.py            # Exports: CheckpointLoader, SingleLapEvaluator, LapMetrics, VisionPipelineTester
│   │   ├── checkpoint.py          # Class: CheckpointLoader
│   │   ├── lap_evaluator.py       # Classes: SingleLapEvaluator, LapMetrics
│   │   └── vision_tester.py       # Class: VisionPipelineTester
│   │
│   └── arena/                     # Backward-compatibility alias to track/
│       ├── __init__.py
│       └── procedural_track.py
│
├── f1_racer_rl/                   # Isaac Lab Task Extension
│   ├── pyproject.toml             # Isaac Lab task extension registration
│   └── src/f1_racer_rl/
│       ├── assets/                # Asset descriptors and USD models
│       │   ├── __init__.py        # F1_RACER_RL_ASSETS_DIR, LEATHERBACK_CFG
│       │   └── leatherback.py     # Robot ArticulationCfg specification
│       │
│       └── tasks/f1_racer/        # RL Environment and Task Logic
│           ├── __init__.py
│           ├── config/f1_car/     # Task Configuration
│           │   ├── __init__.py    # Gymnasium environment registration
│           │   ├── env_cfg.py     # Class: F1RacerEnvCfg (MDP term bindings)
│           │   └── agents/        # RL Agent hyperparameters
│           │       └── rsl_rl_ppo_cfg.py  # Class: PPORunnerCfg (PPO hyperparameters)
│           │
│           └── mdp/               # Class-grouped MDP terms
│               ├── __init__.py    # Domain exports (Classes & Callable aliases)
│               ├── actions.py     # Class: LeatherbackDriveAction, LeatherbackDriveActionCfg
│               ├── observations.py# Class: CarObservations (vision, privileged, proprioception)
│               ├── rewards.py     # Class: CarRewards (centering, progress, alignment, penalties)
│               ├── terminations.py# Class: CarTerminations (off-track, rollover)
│               ├── events.py      # Class: CarEvents (reset pose onto track)
│               └── track_oracle.py# Class: TrackOracle
│
└── scripts/                       # Clean, modular CLI entry points
    ├── train.py                   # RSL-RL PPO Training
    ├── play.py                    # Trained policy playback
    ├── teleop_car.py              # Interactive keyboard driving (uses KeyboardTeleop)
    ├── test_single_lap.py         # Full lap evaluation on new procedural tracks (uses SingleLapEvaluator)
    └── test_vision_pipeline_live.py # Diagnostic vision verification (uses VisionPipelineTester)
```

---

## 2. Class Responsibilities & Taxonomy

### A. Perception (`leatherback_racer.vision`)
- **`TrackDetector`** (alias `BatchedVisionTrackDetector`):
  - Batched PyTorch GPU module for classical computer vision track boundary detection.
  - Computes chromaticity difference ($R + G - 2B$) to isolate yellow boundaries against asphalt.
  - Samples near ($46\% - 52\%$) and far ($40\% - 45\%$) lookahead ROIs.
  - Extracts lateral error ($e_{\text{lat}} \in [-1, 1]$), heading error ($e_{\text{heading}} \in [-\pi, \pi]$), and detection confidence ($[0, 1]$).
- **`TrackVisionHUD`**:
  - High-performance OpenCV HUD overlay renderer.
  - Draws camera feed, detected boundary centroids, projected centerline, and vehicle telemetry dashboard.

### B. Track & Analytical Geometry (`leatherback_racer.track`)
- **`TrackConfig`**: Dataclass defining track parameters (radius, waves, amplitude, width, samples).
- **`ProceduralTrack`**: Generates mathematical splines and builds 3D USD ribbon meshes for racing corridors.
- **`TrackOracle`**: Pre-loaded GPU tensor cache of centerline, tangents, normals, and line boundaries. Computes analytical ground-truth projections and errors in $\mathcal{O}(1)$ time.

### C. Teleoperation (`leatherback_racer.teleop`)
- **`KeyboardTeleop`**: Captures Omniverse Kit keyboard events with smooth steering filters, handbrake, and reset signals.

### D. Evaluation & Diagnostics (`leatherback_racer.evaluation`)
- **`CheckpointLoader`**: Utility for locating latest model weights and restoring actor MLP networks.
- **`SingleLapEvaluator`**: Procedural track generator, evaluator, and runner for single-lap benchmarks.
- **`LapMetrics`**: Dataclass tracking lap time, average speed, max speed, lateral/heading errors, and lane-keeping percentage.
- **`VisionPipelineTester`**: Automated live simulation tester and 4-panel diagnostic dashboard generator.

### E. Markov Decision Process (`f1_racer_rl.tasks.f1_racer.mdp`)
- **`CarObservations`**:
  - `vision_track_state`: 7D deployable observation vector $[e_{\text{lat}}, e_{\text{heading}}, \text{conf}, v_x, v_y, \omega_z, \delta_{\text{steer}}]$.
  - `privileged_track_state`: 8D ground-truth state for the PPO Critic during training.
  - `proprioception`: Vehicle joint positions and body velocities.
- **`CarRewards`**:
  - `centering`: Gaussian reward along centerline.
  - `forward_progress`: Projection of velocity along tangent.
  - `heading_alignment`: Cosine alignment with tangent.
  - `boundary_penalty`: Smooth quadratic penalty approaching boundary ribbons.
  - `steering_smoothness` & `steering_rate`: Penalties preventing high-frequency steering chatter.
- **`CarTerminations`**:
  - `off_track`: Triggers when vehicle exceeds track boundary threshold.
  - `rollover`: Triggers when roll or pitch exceeds $45^\circ$.
- **`CarEvents`**:
  - `reset_pose_on_track`: Places the car cleanly aligned at the start line or random track point with zero velocity.
- **`LeatherbackDriveAction`**:
  - Maps continuous 2D policy outputs $[-1, 1]$ to knuckle steering angles and wheel angular velocities.
