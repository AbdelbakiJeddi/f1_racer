"""Test trained F1 Racer policy on a new unseen procedural track for a single full lap.

Usage:
    python scripts/test_single_lap.py --track_type organic --seed 123 --viz kit

Options:
    --track_type    : Track style: organic, wavy_circle, circuit (default: organic)
    --seed          : Random seed for procedural track geometry (default: 123)
    --checkpoint    : Checkpoint path or 'latest' (default: latest)
    --radius        : Base track scale in meters (default: 13.0)
    --viz           : Visualizer: kit (default: kit)
"""

from __future__ import annotations

import argparse
import math
import os
import shutil

import cv2
import numpy as np
import torch
from isaaclab.app import AppLauncher

# 1. Parse CLI arguments BEFORE launching simulation
parser = argparse.ArgumentParser(description="Single Lap Evaluation on New Track.")
parser.add_argument(
    "--track_type", type=str, default="organic", choices=["organic", "wavy_circle", "circuit"], help="Track type."
)
parser.add_argument("--seed", type=int, default=123, help="Procedural seed for track geometry.")
parser.add_argument("--radius", type=float, default=13.0, help="Base radius of track in meters.")
parser.add_argument("--amplitude", type=float, default=2.2, help="Amplitude of curves in meters.")
parser.add_argument("--checkpoint", type=str, default="latest", help="Path to checkpoint .pt or 'latest'.")
parser.add_argument(
    "--chase_cam",
    action="store_true",
    default=False,
    help="Lock camera in 3rd-person chase mode behind the car (default: False, camera is free).",
)
parser.add_argument(
    "--show_vision",
    action="store_true",
    default=True,
    help="Display live real-time HUD window with onboard camera and detected centerline (default: True).",
)
parser.add_argument(
    "--no_show_vision", dest="show_vision", action="store_false", help="Disable the live vision HUD window."
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.visualizer is None:
    args_cli.visualizer = "kit"
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

# 2. Imports after Kit startup
import f1_racer_rl.tasks  # noqa: F401
import gymnasium as gym
from f1_racer_rl.assets import F1_RACER_RL_ASSETS_DIR
from f1_racer_rl.tasks.f1_racer.config.f1_car.env_cfg import F1RacerEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat

from leatherback_racer.evaluation import CheckpointLoader, SingleLapEvaluator
from leatherback_racer.track import TrackOracle
from leatherback_racer.vision import BatchedVisionTrackDetector, TrackVisionHUD


def main():
    print("\n" + "=" * 80)
    print(" 🏁 F1 RACER: NEW TRACK SINGLE LAP EVALUATION")
    print("=" * 80)

    # 1. Resolve Checkpoint
    ckpt_path = CheckpointLoader.find_latest() if args_cli.checkpoint == "latest" else args_cli.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 2. Generate New Procedural Track
    usd_path, npz_path, track = SingleLapEvaluator.generate_eval_track(
        track_type=args_cli.track_type,
        seed=args_cli.seed,
        radius=args_cli.radius,
        amplitude=args_cli.amplitude,
    )

    # 3. Initialize TrackOracle with the new track data
    oracle = TrackOracle(npz_path=npz_path)
    TrackOracle._instance = oracle

    # Compute start spawn pose
    start_pos_2d = track.centerline[0]
    start_tx, start_ty = track.tangents[0]
    start_yaw = math.atan2(start_ty, start_tx)

    # 4. Configure Isaac Lab Environment
    env_cfg = F1RacerEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 120.0

    cam_start_x = start_pos_2d[0] - 4.0 * math.cos(start_yaw)
    cam_start_y = start_pos_2d[1] - 4.0 * math.sin(start_yaw)
    env_cfg.sim.default_visualizer_cfg.eye = (float(cam_start_x), float(cam_start_y), 2.5)
    env_cfg.sim.default_visualizer_cfg.lookat = (float(start_pos_2d[0]), float(start_pos_2d[1]), 0.1)
    env_cfg.viewer.eye = (float(cam_start_x), float(cam_start_y), 2.5)
    env_cfg.viewer.lookat = (float(start_pos_2d[0]), float(start_pos_2d[1]), 0.1)

    print(
        f"\n[Environment] Spawning Leatherback at start line: "
        f"({start_pos_2d[0]:.2f}, {start_pos_2d[1]:.2f}) "
        f"heading: {math.degrees(start_yaw):+.1f}°"
    )

    env = gym.make("F1RacerRl-F1-Racer-F1-Car", cfg=env_cfg)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]
    camera = env.unwrapped.scene["camera"]
    steer_fl_idx = (
        robot.data.joint_names.index("Knuckle__Upright__Front_Left")
        if "Knuckle__Upright__Front_Left" in robot.data.joint_names
        else 0
    )

    # Initialize Computer Vision Detector & Live HUD
    detector = BatchedVisionTrackDetector(img_h=128, img_w=128).to(device)
    hud = TrackVisionHUD(
        window_name="F1 Racer: Onboard Camera & Centerline HUD",
        panel_size=320,
        enable_display=args_cli.show_vision,
    )
    last_hud_frame = None

    # Load Trained Policy
    policy = CheckpointLoader.load_actor_policy(ckpt_path, device=str(device))
    obs, _ = env.reset()

    # Lap Tracking State
    total_waypoints = oracle.num_points
    max_idx_reached = 0
    lap_completed = False
    halfway_passed = False
    step_count = 0

    speeds = []
    lat_errors = []
    heading_errors = []
    off_track_count = 0

    print("\n" + "=" * 80)
    print(" 🏎️  LAP COMMENCED! WATCH THE 3D VIEWPORT WINDOW")
    print("=" * 80 + "\n")

    try:
        while simulation_app.is_running() and not lap_completed:
            policy_obs = obs["policy"]
            with torch.no_grad():
                action = policy(policy_obs)

            obs, rewards, terminated, truncated, info = env.step(action)

            # Robot state
            root_pos = robot.data.root_pos_w[0].cpu().numpy()
            root_quat = robot.data.root_quat_w[0:1]
            roll, pitch, yaw = euler_xyz_from_quat(root_quat)
            yaw_rad = yaw[0].item()

            vel = robot.data.root_lin_vel_b[0].cpu().numpy()
            speed_ms = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            speed_kmh = speed_ms * 3.6
            speeds.append(speed_ms)

            # Query ground truth oracle
            pos_2d_t = torch.tensor([[root_pos[0], root_pos[1]]], device=device)
            yaw_t = torch.tensor([yaw_rad], device=device)
            oracle_info = oracle.query(pos_2d_t, yaw_t)

            closest_idx = oracle_info["closest_idx"][0].item()
            lat_err = abs(oracle_info["lateral_error"][0, 0].item())
            head_err = abs(oracle_info["heading_error"][0, 0].item())

            lat_errors.append(lat_err)
            heading_errors.append(head_err)

            if lat_err > (oracle.track_width / 2.0):
                off_track_count += 1

            # Lap progress tracking
            progress_pct = (closest_idx / total_waypoints) * 100.0
            if closest_idx > max_idx_reached:
                max_idx_reached = closest_idx

            if closest_idx > (total_waypoints * 0.5):
                halfway_passed = True

            if halfway_passed and closest_idx < 15 and max_idx_reached > (total_waypoints * 0.85):
                lap_completed = True

            # Chase cam
            if args_cli.chase_cam:
                cam_dist = 2.4
                cam_height = 0.95
                cam_x = root_pos[0] - cam_dist * math.cos(yaw_rad)
                cam_y = root_pos[1] - cam_dist * math.sin(yaw_rad)
                cam_z = root_pos[2] + cam_height
                env.unwrapped.sim.set_camera_view(
                    eye=[float(cam_x), float(cam_y), float(cam_z)],
                    target=[float(root_pos[0]), float(root_pos[1]), float(root_pos[2] + 0.1)],
                )

            # Live HUD
            raw_rgb = camera.data.output["rgb"]
            detection_details = detector.detect_detailed(raw_rgb)
            steer_angle_deg = math.degrees(robot.data.joint_pos[0, steer_fl_idx].item())
            telemetry_data = {
                "speed_kmh": speed_kmh,
                "speed_ms": speed_ms,
                "progress_pct": progress_pct,
                "steer_deg": steer_angle_deg,
                "elapsed_s": step_count * env.unwrapped.step_dt,
            }
            last_hud_frame = hud.update(raw_rgb, detection_details, telemetry=telemetry_data, wait_ms=1)

            if step_count % 15 == 0:
                elapsed_s = step_count * env.unwrapped.step_dt
                print(
                    f"[{elapsed_s:5.1f}s | Step {step_count:04d}] "
                    f"Progress: {progress_pct:5.1f}% (idx {closest_idx:03d}/{total_waypoints}) | "
                    f"Speed: {speed_kmh:4.1f} km/h ({speed_ms:4.2f} m/s) | "
                    f"Lat Err: {lat_err * 100:4.1f} cm | "
                    f"Heading Err: {math.degrees(head_err):4.1f}°"
                )

            step_count += 1

            if terminated or truncated:
                print("\n⚠️ Environment triggered termination or truncation before lap finished!")
                break

    except Exception:
        import traceback

        print("\n❌ EXCEPTION CAUGHT IN MAIN LOOP:")
        traceback.print_exc()

    finally:
        total_time_s = step_count * env.unwrapped.step_dt
        avg_speed_ms = float(np.mean(speeds)) if speeds else 0.0
        max_speed_kmh = float(np.max(speeds) * 3.6) if speeds else 0.0
        mean_lat_cm = float(np.mean(lat_errors) * 100.0) if lat_errors else 0.0
        max_lat_cm = float(np.max(lat_errors) * 100.0) if lat_errors else 0.0
        lane_keeping_pct = (1.0 - (off_track_count / max(1, step_count))) * 100.0

        print("\n" + "=" * 80)
        print(" 🏆 SINGLE LAP EVALUATION SCORECARD")
        print("=" * 80)
        if lap_completed:
            print("  STATUS          : ✅ SINGLE LAP SUCCESSFULLY COMPLETED!")
        else:
            print(
                f"  STATUS          : ⚠️ INCOMPLETE "
                f"(Max Progress Reached: {max_idx_reached / total_waypoints * 100:.1f}%)"
            )

        print(f"  Track Style     : {args_cli.track_type.upper()} (Seed: {args_cli.seed})")
        print(f"  Lap Time        : {total_time_s:.2f} seconds ({step_count} control steps)")
        print(f"  Average Speed   : {avg_speed_ms * 3.6:.1f} km/h ({avg_speed_ms:.2f} m/s)")
        print(f"  Max Speed       : {max_speed_kmh:.1f} km/h")
        print(f"  Mean Lat Error  : {mean_lat_cm:.1f} cm (Lane half-width: {oracle.track_width / 2.0 * 100:.0f} cm)")
        print(f"  Max Lat Error   : {max_lat_cm:.1f} cm")
        print(f"  Lane Keeping    : {lane_keeping_pct:.1f}% on-track accuracy")
        print("=" * 80 + "\n")

        # Save snapshot of vision HUD
        if last_hud_frame is not None:
            os.makedirs("outputs/eval_track", exist_ok=True)
            snapshot_path = "outputs/eval_track/single_lap_vision_hud.png"
            cv2.imwrite(snapshot_path, last_hud_frame)
            print(f"  📸 Saved high-resolution Vision HUD snapshot to: {snapshot_path}")

        hud.close()
        env.close()
        simulation_app.close()

        # Restore default training circuit track
        backup_usd = str(F1_RACER_RL_ASSETS_DIR / "track_default_circuit.usd")
        backup_npz = str(F1_RACER_RL_ASSETS_DIR / "track_default_circuit.npz")
        if os.path.exists(backup_usd) and os.path.exists(backup_npz):
            shutil.copyfile(backup_usd, str(F1_RACER_RL_ASSETS_DIR / "track.usd"))
            shutil.copyfile(backup_npz, str(F1_RACER_RL_ASSETS_DIR / "track_data.npz"))
            print("[Track Cleanup] Restored default training circuit track successfully.")


if __name__ == "__main__":
    main()
