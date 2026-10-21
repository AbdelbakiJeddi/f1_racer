"""Diagnostic tester and visual dashboard generator for live simulation computer vision pipeline."""

from __future__ import annotations

import math
import os
from typing import Dict, Any, List
import numpy as np
import torch
from PIL import Image, ImageDraw

from ..track import TrackOracle
from ..vision import BatchedVisionTrackDetector


class VisionPipelineTester:
    """Automated tester and visual dashboard generator for the onboard vision system."""

    @staticmethod
    def render_diagnostic_dashboard(
        latest: Dict[str, Any],
        detector: BatchedVisionTrackDetector,
        output_path: str = "outputs/vision_pipeline_dashboard.png",
    ) -> str:
        """Constructs and saves a high-resolution 4-panel diagnostic dashboard image.

        Panels:
        1. Onboard RGB Camera View with ROI overlays & trajectory vector.
        2. Yellow Line Segmentation Soft Mask.
        3. Near & Far Horizon Scanline Mass Profiles.
        4. Vision vs. Ground-Truth Telemetry Comparison HUD.
        """
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

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

        near_y1 = int(upscale * 0.46)
        near_y2 = int(upscale * 0.52)
        far_y1 = int(upscale * 0.40)
        far_y2 = int(upscale * 0.45)

        draw_raw.rectangle([(0, near_y1), (upscale, near_y2)], outline=(0, 255, 255), width=2)
        draw_raw.text((5, near_y1 - 14), "Near ROI (e_lat)", fill=(0, 255, 255))
        draw_raw.rectangle([(0, far_y1), (upscale, far_y2)], outline=(255, 128, 0), width=2)
        draw_raw.text((5, far_y1 - 14), "Far ROI (e_head)", fill=(255, 128, 0))

        cam_mid = upscale // 2
        draw_raw.line([(cam_mid, 0), (cam_mid, upscale)], fill=(255, 50, 50), width=1)

        vis_lat_norm = latest["vis_lat"]
        vis_head_rad = latest["vis_head"]

        x_near_img = vis_lat_norm * (detector.half_track_width / detector.near_half_width)
        detected_cx_near = int(np.clip(cam_mid + x_near_img * cam_mid, 0, upscale - 1))
        near_mid_y = (near_y1 + near_y2) // 2

        y_near_phys = x_near_img * detector.near_half_width
        delta_y_phys = math.tan(vis_head_rad) * detector.longitudinal_delta_d
        y_far_phys = y_near_phys + delta_y_phys
        x_far_img = y_far_phys / detector.far_half_width
        detected_cx_far = int(np.clip(cam_mid + x_far_img * cam_mid, 0, upscale - 1))
        far_mid_y = (far_y1 + far_y2) // 2

        draw_raw.line([(detected_cx_near, near_mid_y), (detected_cx_far, far_mid_y)], fill=(255, 255, 0), width=2)
        draw_raw.line([(detected_cx_near, near_mid_y - 10), (detected_cx_near, near_mid_y + 10)], fill=(50, 255, 50), width=3)
        draw_raw.line([(cam_mid, near_mid_y), (detected_cx_near, near_mid_y)], fill=(50, 255, 50), width=2)
        draw_raw.line([(detected_cx_far, far_mid_y - 8), (detected_cx_far, far_mid_y + 8)], fill=(255, 128, 0), width=3)

        # 2. Panel B: Line Probability Mask
        pil_mask = Image.fromarray(mask_disp, mode="L").convert("RGB").resize((upscale, upscale), Image.NEAREST)
        draw_mask = ImageDraw.Draw(pil_mask)
        draw_mask.text((5, 5), "Yellow Line Segment Mask", fill=(255, 255, 0))
        draw_mask.line([(detected_cx_near, near_mid_y), (detected_cx_far, far_mid_y)], fill=(255, 255, 0), width=2)
        draw_mask.line([(detected_cx_near, near_mid_y - 10), (detected_cx_near, near_mid_y + 10)], fill=(50, 255, 50), width=3)
        draw_mask.line([(detected_cx_far, far_mid_y - 8), (detected_cx_far, far_mid_y + 8)], fill=(255, 128, 0), width=3)

        # 3. Panel C: Scanline Profiles
        profile_img = Image.new("RGB", (upscale, upscale), (15, 18, 24))
        draw_prof = ImageDraw.Draw(profile_img)

        near_prof = mask_raw[int(H * 0.46):int(H * 0.52), :].mean(axis=0)
        far_prof = mask_raw[int(H * 0.40):int(H * 0.45), :].mean(axis=0)

        draw_prof.line([(upscale // 2, 20), (upscale // 2, upscale - 20)], fill=(60, 60, 70), width=1)
        draw_prof.text((5, 5), "Scanline Intensity Profile", fill=(220, 220, 220))
        draw_prof.text((upscale // 2 - 25, upscale - 18), "Center (0)", fill=(140, 140, 140))

        pts_near = []
        for col_idx, val in enumerate(near_prof):
            x_pix = int(col_idx / W * upscale)
            y_pix = int((upscale - 30) - val * (upscale - 80))
            pts_near.append((x_pix, y_pix))
        if len(pts_near) > 1:
            draw_prof.line(pts_near, fill=(0, 255, 255), width=2)

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

        canvas.save(output_path)
        return output_path
