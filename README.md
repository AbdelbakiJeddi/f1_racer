# Leatherback Vision-Based Path Tracking

Autonomous path tracking and corridor following for the **Leatherback** robot in **Isaac Sim / Isaac Lab** using **onboard camera vision only** (no LiDAR, no GPS/odometry at deployment).

---

## Key Features

- **Procedural 3D Track Generator:** Creates closed motorsport circuits, organic loops, and harmonic wavy tracks with dual yellow boundary ribbons on dark asphalt.
- **Vision-Only Driving:** Onboard camera ($84\times 84$, $k=3$ frame-stacked) + speed proprioception $\to$ steering & throttle commands.
- **Asymmetric Actor-Critic (PPO):**
  - **Actor (Student):** Pure vision policy deployable on real robots.
  - **Critic (Teacher):** Privileged simulator ground-truth states ($e_\text{lat}, e_\text{heading}$) to accelerate training.
- **High Throughput:** Optimized CNN pipeline delivering **>100,000 FPS** on GPU ($<0.3\,\text{ms}$ latency per batch).

---

## Quick Start

### 1. Generate a Procedural Track Arena (USD)
Generates the track stage, exports geometry data (`.npz`), and captures overview and onboard PNGs in `isaacsim/scenes/<arena_name>/`:

```bash
# Generate a natural motorsport circuit (Headless)
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/generate_wavy_track.py --headless --track_type circuit --arena_name arena_circuit_01

# Or launch with the interactive 3D Isaac Sim GUI
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/generate_wavy_track.py
```

### 2. View Robot in Simulation
Spawns the Leatherback robot between boundary lines and captures the onboard camera view:

```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/view_leatherback_camera.py --headless
```

### 3. Verify the CNN Perception & Policy
Runs GPU tensor verification, frame stacking, and inference speed benchmarking:

```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/test_cnn_perception.py
```

---

## Project Structure

```text
f1_racer/
├── scripts/
│   ├── generate_wavy_track.py      # Procedural track generator & scene builder
│   ├── view_leatherback_camera.py  # Robot USD staging & camera test
│   └── test_cnn_perception.py      # Perception & policy verification benchmark
├── src/leatherback_racer/
│   ├── arena/
│   │   └── procedural_track.py     # Ribbon mesh math & USD stage generation
│   ├── vision/
│   │   ├── transforms.py           # ROI crop, resize (84x84), yellow contrast filter
│   │   └── frame_stack.py          # Temporal buffer (k=3) for optical flow / speed
│   └── models/
│       ├── cnn_encoders.py         # Nature-CNN & Spatial Softmax keypoint extractors
│       └── policy.py               # Asymmetric Actor-Critic policy networks
├── CNN_Archi.md                    # Visual presentation & CNN layer breakdowns
└── PLAN.md                         # Detailed RL formulation & training roadmap
```

---

## Documentation

- **[CNN_Archi.md](CNN_Archi.md):** Complete visual architecture guide, layer-by-layer convolution walkthroughs, and spatial keypoint explanations.
- **[PLAN.md](PLAN.md):** Full reinforcement learning problem formulation, reward functions, and asymmetric training plan.
