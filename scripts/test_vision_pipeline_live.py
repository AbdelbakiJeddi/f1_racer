"""Comprehensive Live Isaac Lab Simulation Test for Computer Vision Pipeline.

Tests:
1. End-to-end integration of TiledCamera, BatchedVisionTrackDetector, and vision_track_state in Isaac Lab.
2. Observation tensor shapes: policy=(N, 7), critic=(N, 8).
3. Evaluates line detection on real rendered simulation frames.
4. Compares Vision-derived errors against Ground-Truth mathematical track oracle.
5. Generates high-resolution diagnostic visual output saved to outputs/vision_pipeline_live_dashboard.png.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

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
from leatherback_racer.vision.track_detector import BatchedVisionTrackDetector


def main():
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
        # Apply forward throttle with a gentle steering curve
        if step < 10:
            steer_val = 0.0
            throttle_val = 0.4
        elif step < 20:
            steer_val = 0.2  # Steer slightly right
            throttle_val = 0.5
        else:
            steer_val = -0.2  # Steer slightly left
            throttle_val = 0.5

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

        # Vision State from policy obs
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
            # Capture data for visualization
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

    # D. Build High-Resolution Diagnostic Dashboard
    print("\n[5] Building comprehensive multi-panel diagnostic visualization...")
    latest = recorded_frames[-1]
    rgb_raw = latest["rgb"].numpy()
    if rgb_raw.shape[-1] in (3, 4):
        rgb_disp = rgb_raw[:, :, :3]
    else:
        rgb_disp = np.transpose(rgb_raw[:3], (1, 2, 0))

    if rgb_disp.dtype != np.uint8:
        rgb_disp = np.clip(rgb_disp * 255.0 if rgb_disp.max() <= 1.0 else rgb_disp, 0, 255).astype(np.uint8)

    mask_raw = latest["mask"].numpy()
    mask_disp = np.clip(mask_raw * 255.0, 0, 255).astype(np.uint8)

    H, W = rgb_disp.shape[:2]
    upscale = 256

    # 1. Panel A: Raw camera image with ROI overlays
    pil_raw = Image.fromarray(rgb_disp).resize((upscale, upscale), Image.BILINEAR)
    draw_raw = ImageDraw.Draw(pil_raw)

    # Near & Far ROI Scanline bands (tuned to onboard camera perspective)
    near_y1 = int(upscale * 0.46)
    near_y2 = int(upscale * 0.52)
    far_y1 = int(upscale * 0.40)
    far_y2 = int(upscale * 0.45)

    draw_raw.rectangle([(0, near_y1), (upscale, near_y2)], outline=(0, 255, 255), width=2)
    draw_raw.text((5, near_y1 - 14), "Near ROI (e_lat)", fill=(0, 255, 255))

    draw_raw.rectangle([(0, far_y1), (upscale, far_y2)], outline=(255, 128, 0), width=2)
    draw_raw.text((5, far_y1 - 14), "Far ROI (e_head)", fill=(255, 128, 0))

    # Center crosshair (camera optical center)
    cam_mid = upscale // 2
    draw_raw.line([(cam_mid, 0), (cam_mid, upscale)], fill=(255, 50, 50), width=1)

    # Detected track center at near and far horizons
    vis_lat_norm = latest["vis_lat"]
    vis_head_rad = latest["vis_head"]

    # Reconstruct normalized image coordinates [-1, 1]
    x_near_img = vis_lat_norm * (detector.half_track_width / detector.near_half_width)
    detected_cx_near = int(np.clip(cam_mid + x_near_img * cam_mid, 0, upscale - 1))
    near_mid_y = (near_y1 + near_y2) // 2

    # Reconstruct far horizon centroid from heading
    y_near_phys = x_near_img * detector.near_half_width
    delta_y_phys = math.tan(vis_head_rad) * detector.longitudinal_delta_d
    y_far_phys = y_near_phys + delta_y_phys
    x_far_img = y_far_phys / detector.far_half_width
    detected_cx_far = int(np.clip(cam_mid + x_far_img * cam_mid, 0, upscale - 1))
    far_mid_y = (far_y1 + far_y2) // 2

    # Draw trajectory vector line connecting near and far horizon centers
    draw_raw.line([(detected_cx_near, near_mid_y), (detected_cx_far, far_mid_y)], fill=(255, 255, 0), width=2)

    # Near Horizon marker (Lime green tick)
    draw_raw.line([(detected_cx_near, near_mid_y - 10), (detected_cx_near, near_mid_y + 10)], fill=(50, 255, 50), width=3)
    draw_raw.line([(cam_mid, near_mid_y), (detected_cx_near, near_mid_y)], fill=(50, 255, 50), width=2)

    # Far Horizon marker (Orange tick)
    draw_raw.line([(detected_cx_far, far_mid_y - 8), (detected_cx_far, far_mid_y + 8)], fill=(255, 128, 0), width=3)

    # 2. Panel B: Line Probability Mask
    pil_mask = Image.fromarray(mask_disp, mode="L").convert("RGB").resize((upscale, upscale), Image.NEAREST)
    draw_mask = ImageDraw.Draw(pil_mask)
    draw_mask.text((5, 5), "Yellow Line Segment Mask", fill=(255, 255, 0))
    draw_mask.line([(detected_cx_near, near_mid_y), (detected_cx_far, far_mid_y)], fill=(255, 255, 0), width=2)
    draw_mask.line([(detected_cx_near, near_mid_y - 10), (detected_cx_near, near_mid_y + 10)], fill=(50, 255, 50), width=3)
    draw_mask.line([(detected_cx_far, far_mid_y - 8), (detected_cx_far, far_mid_y + 8)], fill=(255, 128, 0), width=3)

    # 3. Panel C: Scanline Profiles (1D Intensity plot)
    profile_img = Image.new("RGB", (upscale, upscale), (15, 18, 24))
    draw_prof = ImageDraw.Draw(profile_img)

    near_prof = mask_raw[int(H * 0.46):int(H * 0.52), :].mean(axis=0)
    far_prof = mask_raw[int(H * 0.40):int(H * 0.45), :].mean(axis=0)

    # Draw grid
    draw_prof.line([(upscale // 2, 20), (upscale // 2, upscale - 20)], fill=(60, 60, 70), width=1)
    draw_prof.text((5, 5), "Scanline Intensity Profile", fill=(220, 220, 220))
    draw_prof.text((upscale // 2 - 25, upscale - 18), "Center (0)", fill=(140, 140, 140))

    # Plot Near Profile (Cyan)
    pts_near = []
    for col_idx, val in enumerate(near_prof):
        x_pix = int(col_idx / W * upscale)
        y_pix = int((upscale - 30) - val * (upscale - 80))
        pts_near.append((x_pix, y_pix))
    if len(pts_near) > 1:
        draw_prof.line(pts_near, fill=(0, 255, 255), width=2)

    # Plot Far Profile (Orange)
    pts_far = []
    for col_idx, val in enumerate(far_prof):
        x_pix = int(col_idx / W * upscale)
        y_pix = int((upscale - 30) - val * (upscale - 80))
        pts_far.append((x_pix, y_pix))
    if len(pts_far) > 1:
        draw_prof.line(pts_far, fill=(255, 140, 0), width=2)

    draw_prof.text((10, 25), "[Near Horizon] Cyan", fill=(0, 255, 255))
    draw_prof.text((10, 42), "[Far Horizon] Orange", fill=(255, 140, 0))

    # 4. Panel D: Telemetry & Evaluation HUD
    hud_img = Image.new("RGB", (upscale + 60, upscale), (20, 24, 32))
    draw_hud = ImageDraw.Draw(hud_img)

    draw_hud.text((15, 12), "TELEMETRY & COMPARISON", fill=(255, 215, 0))
    draw_hud.line([(15, 32), (upscale + 45, 32)], fill=(70, 80, 95), width=1)

    y_off = 45
    line_h = 24
    telemetry_items = [
        ("Simulation Step", f"{latest['step']:03d}"),
        ("Vehicle Speed (vx)", f"{latest['vx']:.2f} m/s"),
        ("Steering Angle", f"{latest['steer']:+.2f}"),
        ("Detection Confidence", f"{latest['vis_conf']:.1%}"),
        ("Vision e_lat (norm)", f"{latest['vis_lat']:+.3f}"),
        ("Oracle e_lat (norm)", f"{latest['gt_norm_lat']:+.3f}"),
        ("Vision e_heading", f"{latest['vis_head']*180/math.pi:+.1f} deg ({latest['vis_head']:+.3f} rad)"),
        ("Oracle e_heading", f"{latest['gt_head']*180/math.pi:+.1f} deg ({latest['gt_head']:+.3f} rad)"),
    ]

    for label, val in telemetry_items:
        draw_hud.text((15, y_off), label, fill=(160, 175, 190))
        draw_hud.text((195, y_off), val, fill=(240, 245, 250))
        y_off += line_h

    # Overall Banner Canvas
    banner_w = upscale * 3 + (upscale + 60) + 40
    banner_h = upscale + 70
    canvas = Image.new("RGB", (banner_w, banner_h), (12, 14, 18))

    canvas.paste(pil_raw, (10, 50))
    canvas.paste(pil_mask, (upscale + 20, 50))
    canvas.paste(profile_img, (upscale * 2 + 30, 50))
    canvas.paste(hud_img, (upscale * 3 + 40, 50))

    draw_canvas = ImageDraw.Draw(canvas)
    draw_canvas.text((15, 15), "LEATHERBACK F1 RACER :: ONBOARD COMPUTER VISION DIAGNOSTIC PIPELINE", fill=(255, 255, 255))
    draw_canvas.text((10, 32), "1. Camera View + Dual ROIs", fill=(0, 255, 255))
    draw_canvas.text((upscale + 20, 32), "2. PyTorch Yellow Segment", fill=(255, 255, 0))
    draw_canvas.text((upscale * 2 + 30, 32), "3. Scanline Mass Profile", fill=(200, 200, 200))
    draw_canvas.text((upscale * 3 + 40, 32), "4. State Estimation vs Ground Truth", fill=(255, 215, 0))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    canvas.save(args.output)
    abs_out = os.path.abspath(args.output)
    print(f"\n  ✅ Successfully generated visual diagnostic dashboard: {abs_out}")

    print("\n" + "=" * 80)
    print(" ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80 + "\n")

    env.close()
    simulation_app.close()
    return True


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)

