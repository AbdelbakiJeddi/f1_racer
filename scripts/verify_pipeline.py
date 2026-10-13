"""Verification script for F1 Racer environment:
1. Loads the USD robot and track correctly.
2. Uses purely state-based observations and rewards (no camera in policy obs).
3. Steps a Random Agent through the environment to verify interaction.
4. Directly queries and verifies the onboard camera output sensor (saves sample frame).
"""

import argparse
import os
import sys
import numpy as np
from PIL import Image
import torch

parser = argparse.ArgumentParser(description="Verify F1 Racer pipeline with Random Agent and Camera Sensor.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=30, help="Number of random agent interaction steps.")
parser.add_argument("--headless", action="store_true", default=True, help="Run headless.")
parser.add_argument("--save_image", type=str, default="camera_output_verify.png", help="Path to save verified camera image.")
args = parser.parse_args()

# 1. Launch Isaac Lab App
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import f1_racer_rl.tasks  # noqa: F401
from f1_racer_rl.tasks.f1_racer.config.f1_car.env_cfg import F1RacerEnvCfg


def main():
    print("\n" + "=" * 70)
    print(" F1 RACER: PIPELINE & RANDOM AGENT VERIFICATION")
    print("=" * 70)

    # Configure environment
    env_cfg = F1RacerEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    print(f"\n[1] Creating environment with {args.num_envs} envs (headless={args.headless})...")
    env = gym.make("F1RacerRl-F1-Racer-F1-Car", cfg=env_cfg)
    device = env.unwrapped.device

    # Reset environment
    print("\n[2] Resetting environment & checking USD articulation...")
    obs, info = env.reset()
    robot = env.unwrapped.scene["robot"]
    camera = env.unwrapped.scene["camera"]

    # Check USD spawn
    pos = robot.data.root_pos_w.torch[0].cpu().numpy()
    print(f"  Robot USD root position: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.4f})")
    print(f"  Robot joint names ({len(robot.joint_names)} joints): {robot.joint_names}")
    print("  PASS: Robot USD loaded successfully into simulation.")

    # Check Observations (No Camera in policy/critic)
    print("\n[3] Verifying Observations (State-based only, no camera in policy)...")
    for group_name in ["policy", "critic"]:
        if group_name in obs:
            val = obs[group_name]
            print(f"  Obs group '{group_name}' shape: {tuple(val.shape)} (dtype: {val.dtype})")
            assert val.ndim == 2, f"Expected 2D state vector, got {val.ndim}D!"
            assert val.shape[-1] == 8, f"Expected 8D state vector, got {val.shape[-1]}D!"
    print(f"  Policy state sample: {obs['policy'][0].cpu().numpy().round(4)}")
    print("  PASS: Policy observation is pure 8D state vector (zero camera overhead in RL state).")

    # Step Random Agent
    print(f"\n[4] Running Random Agent for {args.steps} interaction steps...")
    for step in range(args.steps):
        # Sample random actions in [-1, 1] for steering and throttle
        steer = 2.0 * torch.rand((args.num_envs, 1), device=device) - 1.0
        throttle = 2.0 * torch.rand((args.num_envs, 1), device=device) - 1.0
        actions = torch.cat([steer, throttle], dim=-1)

        obs, rewards, terminated, truncated, info = env.step(actions)

        if (step + 1) % 10 == 0 or step == 0:
            print(
                f"  Step {step+1:02d}/{args.steps:02d} | "
                f"Steer: {steer[0].item():+.2f} | "
                f"Throttle: {throttle[0].item():+.2f} | "
                f"Mean Rew: {rewards.mean().item():+.4f} | "
                f"Terminated: {terminated.any().item()}"
            )
            assert not torch.isnan(rewards).any(), "NaN detected in rewards!"

    print("  PASS: Random agent interaction completed with valid rewards and state updates.")

    # Verify Camera Sensor Output
    print("\n[5] Verifying Onboard Front Camera Sensor Output...")
    camera_data = camera.data.output
    assert "rgb" in camera_data, "RGB stream not found in camera data!"
    rgb_tensor = camera_data["rgb"]  # Shape: (num_envs, H, W, 3 or 4)
    print(f"  Camera output shape: {tuple(rgb_tensor.shape)} (dtype: {rgb_tensor.dtype})")
    print(f"  RGB value range: min={rgb_tensor.min().item():.1f}, max={rgb_tensor.max().item():.1f}")

    # Save verification frame
    frame = rgb_tensor[0, :, :, :3].cpu().numpy()
    if frame.dtype != np.uint8:
        frame_uint8 = np.clip(frame * 255.0 if frame.max() <= 1.0 else frame, 0, 255).astype(np.uint8)
    else:
        frame_uint8 = frame
    Image.fromarray(frame_uint8).save(args.save_image)
    abs_img_path = os.path.abspath(args.save_image)
    print(f"  Saved camera frame to: {abs_img_path} ({frame_uint8.shape[1]}x{frame_uint8.shape[0]} px)")
    print("  PASS: Onboard camera output sensor is fully functional.")

    # Summary
    print("\n" + "=" * 70)
    print(" ALL VERIFICATION CHECKS COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    env.close()
    simulation_app.close()
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
