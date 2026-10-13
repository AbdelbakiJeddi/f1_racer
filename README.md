# F1 Racer: Leatherback Autonomous Racing in Isaac Lab

Autonomous trajectory tracking and high-speed racing for the **NVIDIA Leatherback** robot in **Isaac Sim / Isaac Lab**.

---

## Key Features

- **Official Leatherback Dynamics:** 26-joint multi-link articulation featuring double-wishbone suspension linkages, tuned shock absorbers, front steering knuckles, and 4-wheel drive.
- **Isaac Lab Extension (`f1_racer_rl`):** Modular, manager-based reinforcement learning environment registered under Gymnasium (`F1RacerRl-F1-Racer-F1-Car`).
- **Procedural 3D Track & Spatial Oracle:** Continuous spline-based track oracle computing exact Frenet frame projection ($e_\text{lat}$, $e_\text{heading}$) and lookahead guidance vectors.
- **State-Based RL Formulation:** Fast, stable 8D observation vector (forward velocity, yaw rate, lateral error, heading alignment error, lookahead waypoints, previous action).
- **Free Viewport Camera:** Unconstrained world perspective camera configured for multi-environment monitoring and visualization.

---

## Quick Start

### 1. Verify Extension & List Registered Environments
Verify that the `f1_racer_rl` extension is recognized by Isaac Lab:

```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python f1_racer_rl/scripts/list_envs.py
```

### 2. Verify Simulation Pipeline & Articulation
Tests the USD track stage, official Leatherback suspension physics, state observations, and camera sensor rendering:

```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/verify_pipeline.py --headless
```

### 3. Step Random Agent in Parallel Environments
Runs interactive multi-environment stepping with the random agent baseline:

```bash
# Run headless with 2 parallel environments
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/play_random_agent.py --num_envs 2 --steps 60 --headless

# Or launch with 3D GUI window
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/play_random_agent.py --num_envs 2 --steps 60
```

### 4. Generate Procedural Track Arena (USD)
Generates standalone racing circuits and exports geometry data (`.npz`) and USD stages:

```bash
# Generate a natural motorsport circuit (Headless)
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/generate_wavy_track.py --headless --track_type circuit --arena_name arena_circuit_01

# Or launch with the interactive 3D Isaac Sim GUI
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/generate_wavy_track.py
```

### 5. View Robot in Simulation
Spawns the Leatherback robot between boundary lines and captures the onboard camera view:

```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/view_leatherback_camera.py --headless
```

---

## Project Structure

```text
f1_racer/
├── f1_racer_rl/                    # Isaac Lab Extension package
│   ├── config/extension.toml       # Omniverse extension definition
│   ├── pyproject.toml              # Extension installation metadata
│   ├── src/f1_racer_rl/
│   │   ├── assets/                 # Official Leatherback robot & track assets
│   │   │   ├── data/               # USD assets (leatherback.usd, track.usd, track_data.npz)
│   │   │   └── leatherback.py      # Articulation & suspension actuator configuration
│   │   └── tasks/f1_racer/         # Task MDP formulation
│   │       ├── config/f1_car/      # Environment & RSL-RL PPO agent configs
│   │       └── mdp/                # Actions, observations, rewards, terminations, track oracle
│   └── tests/                      # Gym registration and environment tests
├── scripts/
│   ├── verify_pipeline.py          # Isaac Lab pipeline & official articulation test
│   ├── play_random_agent.py        # Multi-environment random agent stepping
│   ├── generate_wavy_track.py      # Procedural track generator & scene builder
│   ├── view_leatherback_camera.py  # Robot USD staging & camera test
│   └── test_cnn_perception.py      # Perception & policy verification benchmark
├── src/leatherback_racer/
│   ├── arena/
│   │   └── procedural_track.py     # Ribbon mesh math & USD stage generation
│   ├── vision/
│   │   ├── transforms.py           # ROI crop, resize, contrast filter
│   │   └── frame_stack.py          # Temporal buffer for frame sequences
│   └── models/
│       ├── cnn_encoders.py         # Visual backbones & feature extractors
│       └── policy.py               # Actor-Critic network definitions
└── PLAN.md                         # Detailed RL formulation & training roadmap
```

---

## Documentation

- **[PLAN.md](PLAN.md):** Full reinforcement learning problem formulation, reward functions, and training roadmap.
