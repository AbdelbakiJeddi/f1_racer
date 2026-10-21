"""Comprehensive Live Isaac Lab Simulation Test for Computer Vision Pipeline.

Tests:
1. End-to-end integration of TiledCamera, BatchedVisionTrackDetector, and vision_track_state in Isaac Lab.
2. Observation tensor shapes: policy=(N, 7), critic=(N, 8).
3. Evaluates line detection on real rendered simulation frames.
4. Compares Vision-derived errors against Ground-Truth mathematical track oracle.
5. Generates high-resolution diagnostic visual output saved to outputs/vision_pipeline_dashboard.png.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import numpy as np
import torch

# 1. Parse arguments BEFORE AppLauncher
parser = argparse.ArgumentParser(description="Live Isaac Lab Vision Pipeline Test.")
parser.add_argument("--headless", action="store_true", default=True, help="Run headless.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel environments.")
parser.add_argument("--steps", type=int, default=30, help="Warmup / interaction simulation steps.")
parser.add_argument("--output", type=str, default="outputs/vision_pipeline_dashboard.png", help="Path to save output.")
args = parser.parse_args()

# 2. Launch Isaac Lab App
from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=args.headless, enable_cameras=True)
simulation_app = app_launcher.app

import gymnasium as gym
import f1_racer_rl.tasks  # noqa: F401
from f1_racer_rl.tasks.f1_racer.config.f1_car.env_cfg import F1RacerEnvCfg
from f1_racer_rl.tasks.f1_racer.mdp.track_oracle import TrackOracle
from isaaclab.utils.math import euler_xyz_from_quat
from leatherback_racer.evaluation import VisionPipelineTester
from leatherback_racer.vision import BatchedVisionTrackDetector


def main() -> bool:
    print("\n" + "=" * 80)
    print(" 🏁 LEATHERBACK F1 RACER: LIVE SIMULATION COMPUTER VISION TEST")
    print("=" * 80)

    # A. Setup Environment
    env_cfg = F1RacerEnvCfg()
    env_cfg.scene.num_envs = args.num_envs

    print(f"\n[1] Initializing Gymnasium environment (num_envs={args.num_envs}, headless={args.headless})...")
    env = gym.make("F1RacerRl-F1-Racer-F1-Car", cfg=env_cfg)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    camera = env.unwrapped.scene["camera"]
    oracle = TrackOracle.get_instance(device=str(device))

    # B. Reset Environment
    print("\n[2] Resetting environment & placing car on track...")
    obs, info = env.reset()

    # Verify Observation Dimensions
    print("\n[3] Verifying Observation Tensor Spaces:")
    assert "policy" in obs, "Obs group 'policy' missing!"
    assert "critic" in obs, "Obs group 'critic' missing!"

    policy_obs = obs["policy"]
    critic_obs = obs["critic"]
    print(f"  • Policy observation shape: {tuple(policy_obs.shape)} (dtype: {policy_obs.dtype})")
    print(f"  • Critic observation shape: {tuple(critic_obs.shape)} (dtype: {critic_obs.dtype})")

    assert policy_obs.shape[-1] == 7, f"Expected 7D policy obs [e_lat, e_head, conf, vx, vy, wz, steer], got {policy_obs.shape[-1]}!"
    assert critic_obs.shape[-1] == 8, f"Expected 8D privileged critic obs, got {critic_obs.shape[-1]}!"
    print("  ✅ PASS: Policy obs is 7D vision-kinematic state; Critic is 8D privileged state.")

    # C. Warmup & Step Simulation
    print(f"\n[4] Stepping simulation for {args.steps} steps to render camera frames and drive forward...")
    detector = BatchedVisionTrackDetector(img_h=128, img_w=128).to(device)
    recorded_frames = []

    for step in range(args.steps):
        if step < 10:
            steer_val, throttle_val = 0.0, 0.4
        elif step < 20:
            steer_val, throttle_val = 0.2, 0.5
        else:
            steer_val, throttle_val = -0.2, 0.5

        action = torch.tensor([[steer_val, throttle_val]], device=device).repeat(args.num_envs, 1)
        obs, rewards, terminated, truncated, info = env.step(action)

        # Telemetry
        pos_w = robot.data.root_pos_w.torch[0]
        quat_w = robot.data.root_quat_w.torch[0:1]
        _, _, yaw = euler_xyz_from_quat(quat_w)
        pos_2d = (pos_w[:2] - env.unwrapped.scene.env_origins[0, :2]).unsqueeze(0)
        gt_query = oracle.query(pos_2d, yaw)

        gt_lat = gt_query["lateral_error"][0, 0].item()
        gt_head = gt_query["heading_error"][0, 0].item()
        half_width = oracle.track_width / 2.0
        gt_norm_lat = gt_lat / half_width

        vis_lat = obs["policy"][0, 0].item()
        vis_head = obs["policy"][0, 1].item()
        vis_conf = obs["policy"][0, 2].item()
        vx = obs["policy"][0, 3].item()

        if step % 5 == 0 or step == args.steps - 1:
            print(
                f"  Step {step:02d} | Speed: {vx:.2f} m/s | "
                f"Vision e_lat: {vis_lat:+.3f} (GT: {gt_norm_lat:+.3f}) | "
                f"Vision e_head: {vis_head:+.3f} rad ({vis_head*180/math.pi:+.1f}°) | "
                f"Conf: {vis_conf:.2f}"
            )

        if step in [5, 15, 25, args.steps - 1]:
            raw_rgb = camera.data.output["rgb"]
            features, mask = detector(raw_rgb)
            recorded_frames.append({
                "step": step,
                "rgb": raw_rgb[0].detach().cpu(),
                "mask": mask[0].detach().cpu(),
                "vis_lat": vis_lat,
                "vis_head": vis_head,
                "vis_conf": vis_conf,
                "gt_lat": gt_lat,
                "gt_norm_lat": gt_norm_lat,
                "gt_head": gt_head,
                "vx": vx,
                "steer": steer_val,
            })

    # D. Build High-Resolution Diagnostic Dashboard using VisionPipelineTester class
    print("\n[5] Building comprehensive multi-panel diagnostic visualization...")
    latest = recorded_frames[-1]
    saved_path = VisionPipelineTester.render_diagnostic_dashboard(latest, detector, args.output)
    print(f"  ✅ Successfully generated visual diagnostic dashboard: {os.path.abspath(saved_path)}")

    print("\n" + "=" * 80)
    print(" ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")

    env.close()
    simulation_app.close()
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
