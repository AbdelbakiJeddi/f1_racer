"""Run a Random Agent in the F1 Racer environment.

Usage:
    # Run with GUI window to watch the car drive:
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/play_random_agent.py

    # Run headless in terminal:
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/play_random_agent.py --headless --steps 200
"""

import argparse
import math
import sys
import torch

parser = argparse.ArgumentParser(description="Run Random Agent in F1 Racer environment.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments (default: 1).")
parser.add_argument("--steps", type=int, default=500, help="Total simulation steps to run (default: 500).")
parser.add_argument("--headless", action="store_true", default=False, help="Run headless without 3D window.")
args = parser.parse_args()

# Launch Isaac Lab simulation app
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import f1_racer_rl.tasks  # noqa: F401
from f1_racer_rl.tasks.f1_racer.config.f1_car.env_cfg import F1RacerEnvCfg


def main():
    print("\n" + "=" * 75)
    print(" F1 RACER: RANDOM AGENT PLAYBACK")
    print("=" * 75)
    print(f" Environments: {args.num_envs} | Steps: {args.steps} | Headless: {args.headless}\n")

    env_cfg = F1RacerEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    env = gym.make("F1RacerRl-F1-Racer-F1-Car", cfg=env_cfg)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]

    obs, info = env.reset()
    print("[INFO] Environment reset complete. Starting interaction loop...\n")

    step_count = 0
    total_reward = 0.0

    while simulation_app.is_running() and step_count < args.steps:
        # Sample smooth random actions: steer in [-0.8, 0.8], forward throttle in [0.2, 0.9]
        steer = 1.6 * torch.rand((args.num_envs, 1), device=device) - 0.8
        throttle = 0.7 * torch.rand((args.num_envs, 1), device=device) + 0.2
        actions = torch.cat([steer, throttle], dim=-1)

        obs, rewards, terminated, truncated, info = env.step(actions)

        # Telemetry
        vel = robot.data.root_lin_vel_b.torch[0].cpu().numpy()
        pos = robot.data.root_pos_w.torch[0].cpu().numpy()
        speed = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
        total_reward += rewards.mean().item()

        if step_count % 25 == 0 or terminated.any() or truncated.any():
            print(
                f"[Step {step_count:04d}/{args.steps:04d}] "
                f"Speed: {speed:4.2f} m/s | "
                f"Pos: ({pos[0]:5.2f}, {pos[1]:5.2f}) | "
                f"Steer: {steer[0].item():+5.2f} | "
                f"Throttle: {throttle[0].item():4.2f} | "
                f"Step Rew: {rewards.mean().item():+6.3f} | "
                f"Done: {terminated.any().item() or truncated.any().item()}"
            )

        if terminated.any() or truncated.any():
            print(f"  --> Episode finished at step {step_count}. Resetting...")
            obs, info = env.reset()

        step_count += 1

    print("\n" + "=" * 75)
    print(" RANDOM AGENT RUN COMPLETED")
    print(f" Steps run: {step_count} | Average Step Reward: {total_reward / max(1, step_count):+.4f}")
    print("=" * 75 + "\n")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
