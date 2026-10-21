"""Interactive manual keyboard teleoperation for the F1 Leatherback car in Isaac Lab.

Usage:
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/teleop_car.py

Controls (active inside the 3D simulation window):
    W / UP          : Throttle Forward (accelerate smoothly)
    S / DOWN        : Reverse / Brake
    A / LEFT        : Steer Left
    D / RIGHT       : Steer Right
    SPACE           : Handbrake (instant stop)
    R               : Reset car to track starting line
"""

from __future__ import annotations

import argparse
import math
import sys
import torch

from isaaclab.app import AppLauncher

# 1. Setup CLI argument parser with Isaac Lab launcher args
parser = argparse.ArgumentParser(description="Manual drive smoke test for Leatherback F1 Racer.")
parser.add_argument("--chase_cam", action="store_true", default=False, help="Lock camera in 3rd-person chase mode behind the car (default: False, camera is free).")
parser.add_argument("--show_vision", action="store_true", default=True, help="Display live real-time HUD window with onboard camera and detected centerline (default: True).")
parser.add_argument("--no_show_vision", dest="show_vision", action="store_false", help="Disable vision HUD window.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.visualizer is None:
    args_cli.visualizer = "kit"

# 2. Launch Isaac Lab simulation
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# 3. Post-launch imports
import gymnasium as gym

import f1_racer_rl.tasks  # noqa: F401
from f1_racer_rl.tasks.f1_racer.config.f1_car.env_cfg import F1RacerEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat
from leatherback_racer.teleop import KeyboardTeleop
from leatherback_racer.vision import BatchedVisionTrackDetector, TrackVisionHUD


def main():
    print("\n" + "=" * 78)
    print(" 🏎️  F1 RACER: INTERACTIVE MANUAL DRIVE TELEOP")
    print("=" * 78)
    print(" Controls (Click inside the 3D window to focus):")
    print("   [W / UP]     : Accelerate forward")
    print("   [S / DOWN]   : Brake / Decelerate")
    print("   [A / LEFT]   : Steer Left")
    print("   [D / RIGHT]  : Steer Right")
    print("   [SPACE]      : Handbrake (instant stop)")
    print("   [R]          : Reset car back to track start")
    print("=" * 78 + "\n")

    env_cfg = F1RacerEnvCfg()
    env_cfg.scene.num_envs = 1

    # Position viewport camera behind and above car
    env_cfg.sim.default_visualizer_cfg.eye = (19.5, 4.0, 1.8)
    env_cfg.sim.default_visualizer_cfg.lookat = (17.0, 3.9, 0.05)
    env_cfg.viewer.eye = (19.5, 4.0, 1.8)
    env_cfg.viewer.lookat = (17.0, 3.9, 0.05)

    env = gym.make("F1RacerRl-F1-Racer-F1-Car", cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    camera = env.unwrapped.scene["camera"]
    device = env.unwrapped.device

    # Initialize Computer Vision Detector & Live HUD
    detector = BatchedVisionTrackDetector(img_h=128, img_w=128).to(device)
    hud = TrackVisionHUD(
        window_name="F1 Racer Teleop: Onboard Camera & Centerline HUD",
        panel_size=320,
        enable_display=args_cli.show_vision,
    )

    teleop = KeyboardTeleop()
    obs, info = env.reset()

    body_names = robot.data.body_names
    wheel_names = ["Wheel_Front_Left", "Wheel_Front_Right", "Wheel_Rear_Left", "Wheel_Rear_Right"]
    wheel_indices = [body_names.index(w) for w in wheel_names if w in body_names]
    steer_fl_idx = robot.data.joint_names.index("Knuckle__Upright__Front_Left")

    step = 0
    try:
        while simulation_app.is_running():
            steer, throttle, reset_req = teleop.update()

            if reset_req:
                print("\n[RESET] Teleop requested track reset...")
                obs, info = env.reset()
                continue

            action = torch.tensor([[steer, throttle]], device=device)
            obs, rewards, terminated, truncated, info = env.step(action)

            # Telemetry
            vel = robot.data.root_lin_vel_b[0].cpu().numpy()
            speed_ms = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            speed_kmh = speed_ms * 3.6
            root_pos = robot.data.root_pos_w[0].cpu().numpy()
            root_quat = robot.data.root_quat_w[0:1]
            roll, pitch, yaw = euler_xyz_from_quat(root_quat)
            pitch_deg = torch.rad2deg(pitch)[0].item()
            yaw_rad = yaw[0].item()

            # Dynamic 3rd-person chase camera smoothly tracking the car (if enabled)
            if args_cli.chase_cam:
                cam_dist = 2.2
                cam_height = 0.85
                cam_x = root_pos[0] - cam_dist * math.cos(yaw_rad)
                cam_y = root_pos[1] - cam_dist * math.sin(yaw_rad)
                cam_z = root_pos[2] + cam_height
                env.unwrapped.sim.set_camera_view(
                    eye=[float(cam_x), float(cam_y), float(cam_z)],
                    target=[float(root_pos[0]), float(root_pos[1]), float(root_pos[2] + 0.1)],
                )

            body_pos = robot.data.body_pos_w[0].cpu().numpy()
            wheel_zs = [body_pos[idx, 2] for idx in wheel_indices]
            knuckle_deg = torch.rad2deg(robot.data.joint_pos[0, steer_fl_idx]).item()

            # Live Computer Vision HUD update
            raw_rgb = camera.data.output["rgb"]
            detection_details = detector.detect_detailed(raw_rgb)
            telemetry_data = {
                "speed_kmh": speed_kmh,
                "speed_ms": speed_ms,
                "steer_deg": knuckle_deg,
                "elapsed_s": step * env.unwrapped.step_dt,
            }
            hud.update(raw_rgb, detection_details, telemetry=telemetry_data, wait_ms=1)

            if step % 15 == 0:
                print(
                    f"[Step {step:05d}] Speed: {speed_kmh:4.1f} km/h ({speed_ms:4.2f} m/s) | "
                    f"Steer: {knuckle_deg:+5.1f}° (cmd: {steer:+4.2f}) | "
                    f"Pitch: {pitch_deg:+5.1f}° | "
                    f"Wheels Z: FL={wheel_zs[0]:.3f} FR={wheel_zs[1]:.3f} RL={wheel_zs[2]:.3f} RR={wheel_zs[3]:.3f}"
                )

            step += 1
    finally:
        hud.close()
        teleop.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
