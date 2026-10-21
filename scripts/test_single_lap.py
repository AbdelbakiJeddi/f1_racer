"""Test trained F1 Racer policy on a new unseen track for a single full lap.

Usage:
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/test_single_lap.py --track_type organic --seed 123 --viz kit

Options:
    --track_type    : Track style: organic, wavy_circle, circuit (default: organic)
    --seed          : Random seed for procedural track geometry (default: 123)
    --checkpoint    : Checkpoint path or 'latest' (default: latest)
    --radius        : Base track scale in meters (default: 13.0)
    --viz           : Visualizer: kit (default: kit)
"""

from __future__ import annotations

import argparse
import glob
import math
import os
import sys
import time
import numpy as np
import torch

from isaaclab.app import AppLauncher

# 1. Parse CLI arguments BEFORE launching simulation
parser = argparse.ArgumentParser(description="Single Lap Evaluation on New Track.")
parser.add_argument("--track_type", type=str, default="organic", choices=["organic", "wavy_circle", "circuit"], help="Track type.")
parser.add_argument("--seed", type=int, default=123, help="Procedural seed for track geometry.")
parser.add_argument("--radius", type=float, default=13.0, help="Base radius of track in meters.")
parser.add_argument("--amplitude", type=float, default=2.2, help="Amplitude of curves in meters.")
parser.add_argument("--checkpoint", type=str, default="latest", help="Path to checkpoint .pt or 'latest'.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.visualizer is None:
    args_cli.visualizer = "kit"
args_cli.enable_cameras = True

launcher = AppLauncher(args_cli)
simulation_app = launcher.app

# 2. Imports after Kit startup
from pxr import Usd, UsdGeom
import gymnasium as gym

import f1_racer_rl.tasks  # noqa: F401
import leatherback_racer  # noqa: F401
from f1_racer_rl.tasks.f1_racer.config.f1_car.env_cfg import F1RacerEnvCfg
from f1_racer_rl.tasks.f1_racer.mdp.track_oracle import TrackOracle
from isaaclab.utils.math import euler_xyz_from_quat, quat_from_euler_xyz
from leatherback_racer.arena.procedural_track import ProceduralTrack, TrackConfig


def find_latest_checkpoint() -> str:
    """Finds the most recent model_*.pt checkpoint."""
    ckpts = glob.glob("logs/rsl_rl/f1_racer_leatherback/*/*.pt")
    if not ckpts:
        raise FileNotFoundError("No trained checkpoints found in logs/rsl_rl/f1_racer_leatherback/!")
    ckpts.sort(key=os.path.getmtime, reverse=True)
    return ckpts[0]


def load_actor_policy(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    """Reconstructs and loads the trained actor MLP from an RSL-RL checkpoint."""
    print(f"[Policy] Loading weights from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["actor_state_dict"]

    mlp = torch.nn.Sequential(
        torch.nn.Linear(7, 256),
        torch.nn.ELU(),
        torch.nn.Linear(256, 128),
        torch.nn.ELU(),
        torch.nn.Linear(128, 64),
        torch.nn.ELU(),
        torch.nn.Linear(64, 2),
    ).to(device)

    mlp_dict = {k[4:]: v for k, v in state_dict.items() if k.startswith("mlp.")}
    mlp.load_state_dict(mlp_dict)
    mlp.eval()
    print("  ✅ Actor MLP (7 -> 256 -> 128 -> 64 -> 2) loaded successfully.")
    return mlp


def generate_eval_track(track_type: str, seed: int, radius: float, amplitude: float) -> tuple[str, str, ProceduralTrack]:
    """Generates a new procedural track USD and NPZ metadata in outputs/eval_track/."""
    out_dir = os.path.abspath("outputs/eval_track")
    os.makedirs(out_dir, exist_ok=True)
    usd_path = os.path.join(out_dir, f"track_{track_type}_{seed}.usd")
    npz_path = os.path.join(out_dir, f"track_{track_type}_{seed}_data.npz")

    cfg = TrackConfig(
        track_type=track_type,
        seed=seed,
        radius=radius,
        amplitude=amplitude,
        track_width=1.2,
        line_thickness=0.08,
        num_samples=500,
    )
    track = ProceduralTrack(cfg)

    # Save NPZ metadata
    np.savez(
        npz_path,
        centerline=track.centerline,
        tangents=track.tangents,
        normals=track.normals,
        left_line=track.left_line,
        right_line=track.right_line,
        track_width=track.cfg.track_width,
        is_closed=track.is_closed,
    )

    # Save USD mesh ribbons
    if os.path.exists(usd_path):
        os.remove(usd_path)
    stage = Usd.Stage.CreateNew(usd_path)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    track.build_usd_meshes(stage, root_path="/Track")
    stage.GetRootLayer().Save()

    total_len = np.sum(np.linalg.norm(np.diff(track.centerline, axis=0), axis=1)) + np.linalg.norm(track.centerline[0] - track.centerline[-1])
    print(f"\n[Track Generation] Created new '{track_type}' track (seed={seed}):")
    print(f"  • Total Lap Length: {total_len:.1f} meters")
    print(f"  • Centerline Waypoints: {track.centerline.shape[0]} points")
    print(f"  • USD Path: {usd_path}")
    print(f"  • NPZ Path: {npz_path}")

    return usd_path, npz_path, track


def main():
    print("\n" + "=" * 80)
    print(" 🏁 F1 RACER: NEW TRACK SINGLE LAP EVALUATION")
    print("=" * 80)

    # 1. Resolve Checkpoint
    ckpt_path = find_latest_checkpoint() if args_cli.checkpoint == "latest" else args_cli.checkpoint
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    # 2. Generate New Unseen Track
    usd_path, npz_path, track = generate_eval_track(
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
    start_quat = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([start_yaw]))[0]

    # 4. Configure Isaac Lab Environment
    env_cfg = F1RacerEnvCfg()
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = 90.0  # Allow plenty of time for full lap

    # Replace track mesh with new track
    env_cfg.scene.track.spawn.usd_path = usd_path

    # Set robot start pose
    env_cfg.scene.robot.init_state.pos = (float(start_pos_2d[0]), float(start_pos_2d[1]), 0.035)
    env_cfg.scene.robot.init_state.rot = (float(start_quat[0]), float(start_quat[1]), float(start_quat[2]), float(start_quat[3]))

    print(f"\n[Environment] Spawning Leatherback at start line: ({start_pos_2d[0]:.2f}, {start_pos_2d[1]:.2f}) heading: {math.degrees(start_yaw):+.1f}°")

    env = gym.make("F1RacerRl-F1-Racer-F1-Car", cfg=env_cfg)
    device = env.unwrapped.device
    robot = env.unwrapped.scene["robot"]

    # Load Actor Policy
    policy = load_actor_policy(ckpt_path, device=str(device))

    obs, _ = env.reset()

    # Lap Tracking State
    total_waypoints = oracle.num_points
    max_idx_reached = 0
    lap_completed = False
    halfway_passed = False
    step_count = 0
    start_time = time.time()

    speeds = []
    lat_errors = []
    heading_errors = []
    off_track_count = 0

    print("\n" + "=" * 80)
    print(" 🏎️  LAP COMMENCED! WATCH THE 3D VIEWPORT WINDOW")
    print("=" * 80 + "\n")

    try:
        while simulation_app.is_running() and not lap_completed:
            # Policy inference
            policy_obs = obs["policy"]  # (1, 7)
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

            # Lap finish detection: car passed halfway and now returns near start index (index < 15)
            if halfway_passed and closest_idx < 15 and max_idx_reached > (total_waypoints * 0.85):
                lap_completed = True

            # Update dynamic 3rd-person chase camera
            cam_dist = 2.4
            cam_height = 0.95
            cam_x = root_pos[0] - cam_dist * math.cos(yaw_rad)
            cam_y = root_pos[1] - cam_dist * math.sin(yaw_rad)
            cam_z = root_pos[2] + cam_height
            env.unwrapped.sim.set_camera_view(
                eye=[float(cam_x), float(cam_y), float(cam_z)],
                target=[float(root_pos[0]), float(root_pos[1]), float(root_pos[2] + 0.1)]
            )

            # Telemetry print
            if step_count % 15 == 0:
                elapsed_s = step_count * env.unwrapped.step_dt
                print(
                    f"[{elapsed_s:5.1f}s | Step {step_count:04d}] "
                    f"Progress: {progress_pct:5.1f}% (idx {closest_idx:03d}/{total_waypoints}) | "
                    f"Speed: {speed_kmh:4.1f} km/h ({speed_ms:4.2f} m/s) | "
                    f"Lat Err: {lat_err*100:4.1f} cm | "
                    f"Heading Err: {math.degrees(head_err):4.1f}°"
                )

            step_count += 1

            if terminated or truncated:
                print("\n⚠️ Environment triggered termination or truncation before lap finished!")
                break

    finally:
        total_time_s = step_count * env.unwrapped.step_dt
        avg_speed_ms = np.mean(speeds) if speeds else 0.0
        max_speed_kmh = (np.max(speeds) * 3.6) if speeds else 0.0
        mean_lat_cm = np.mean(lat_errors) * 100.0 if lat_errors else 0.0
        max_lat_cm = np.max(lat_errors) * 100.0 if lat_errors else 0.0
        lane_keeping_pct = (1.0 - (off_track_count / max(1, step_count))) * 100.0

        print("\n" + "=" * 80)
        print(" 🏆 SINGLE LAP EVALUATION SCORECARD")
        print("=" * 80)
        if lap_completed:
            print("  STATUS          : ✅ SINGLE LAP SUCCESSFULLY COMPLETED!")
        else:
            print(f"  STATUS          : ⚠️ INCOMPLETE (Max Progress Reached: {max_idx_reached/total_waypoints*100:.1f}%)")

        print(f"  Track Style     : {args_cli.track_type.upper()} (Seed: {args_cli.seed})")
        print(f"  Lap Time        : {total_time_s:.2f} seconds ({step_count} control steps)")
        print(f"  Average Speed   : {avg_speed_ms * 3.6:.1f} km/h ({avg_speed_ms:.2f} m/s)")
        print(f"  Max Speed       : {max_speed_kmh:.1f} km/h")
        print(f"  Mean Lat Error  : {mean_lat_cm:.1f} cm (Lane half-width: {oracle.track_width/2.0*100:.0f} cm)")
        print(f"  Max Lat Error   : {max_lat_cm:.1f} cm")
        print(f"  Lane Keeping    : {lane_keeping_pct:.1f}% on-track accuracy")
        print("=" * 80 + "\n")

        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()
