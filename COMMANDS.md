# F1 Racer — Command Reference & Execution Guide

This document lists all the standard commands to run, train, evaluate, and diagnose the NVIDIA Leatherback F1 Racer in Isaac Lab 3.0.

> **Environment Note:** Always run commands from the project root (`/home/abok-omen/Desktop/Abok/robotics-rl/f1_racer`) using the dedicated Isaac Lab 3.0 Python binary:
> ```bash
> /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python
> ```

---

## 1. Reinforcement Learning Training

### A. Fast Headless Training (Recommended for Maximum FPS)
Runs parallel GPU simulation without GUI rendering overhead:
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 64 --max_iterations 500
```
- `--num_envs 64`: Simulates 64 cars in parallel on the GPU.
- `--max_iterations 500`: Runs 500 PPO iterations.
- Logs and checkpoints are automatically saved to `logs/rsl_rl/f1_racer_leatherback/<timestamp>/`.

### B. Live Visualized Training (Single or Multi-Env with 3D GUI)
Opens the Omniverse Kit viewport to watch the agent learn in real-time:
```bash
# 1 environment with 3D window:
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 1 --viz kit

# 8 parallel environments with 3D window:
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 8 --viz kit
```

### C. Resuming Training from Checkpoint
```bash
# Resume from the most recent checkpoint:
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 64 --checkpoint latest

# Resume from a specific model file:
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 64 --checkpoint logs/rsl_rl/f1_racer_leatherback/<run_dir>/model_200.pt
```

---

## 2. Policy Evaluation & Playback

### A. Watch Trained Policy Drive on a New Unseen Track (Single Lap Test)
Generates a brand new procedural track (different geometry & curves) and evaluates the policy completing a full lap with live 3D chase camera and performance scorecard:
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/test_single_lap.py --track_type organic --seed 123 --viz kit
```
- `--track_type`: `organic`, `wavy_circle`, or `circuit`.
- `--seed`: Any integer to generate different procedural curves.

### B. Watch Trained Policy Drive on Default Track
Load the latest trained model and watch it navigate the track in the 3D window:
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/play.py --checkpoint latest --viz kit
```

### C. Evaluate a Specific Model Checkpoint
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/play.py --checkpoint logs/rsl_rl/f1_racer_leatherback/<run_dir>/model_749.pt --num_envs 1 --viz kit
```

---

## 3. Interactive Manual Driving (Smoke Test)

Drive the physical car model interactively using the keyboard to verify steering, throttle, suspension, and tire grip:
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/teleop_car.py
```
**Controls inside the 3D window:**
- `W` / `Up Arrow` : Accelerate forward smoothly
- `S` / `Down Arrow` : Brake / Reverse
- `A` / `Left Arrow` : Steer Left
- `D` / `Right Arrow`: Steer Right
- `SPACE` : Handbrake (instant stop)
- `R` : Reset car back to starting line
- Features an automatic 3rd-person dynamic chase camera following the vehicle.

---

## 4. Computer Vision Diagnostics

### A. Live Simulation Vision Test & Dashboard Generator
Tests the camera rendering, GPU line segmentation, dual-ROI error estimation, and compares vision against ground truth:
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/test_vision_pipeline_live.py --steps 30
```
- Generates a 4-panel diagnostic image at `outputs/vision_pipeline_dashboard.png`.

### B. Static Vision Pipeline Test
Tests the vision detector on pre-saved track images:
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/test_vision_pipeline.py
```

---

## 5. Monitoring & Metrics

### TensorBoard
Monitor training curves (reward, episode length, policy loss, value loss):
```bash
/home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/tensorboard --logdir logs/rsl_rl/f1_racer_leatherback --port 6006
```
Open `http://localhost:6006` in your browser.

