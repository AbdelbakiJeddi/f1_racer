"""Real-time HUD Visualization for F1 Racer Onboard Vision and Centerline Detection."""

from __future__ import annotations

import contextlib
import math
from typing import Any

import cv2
import numpy as np
import torch


class TrackVisionHUD:
    """Renders a rich real-time visual HUD showing onboard camera feedback,
    segmented track boundaries, detected centerline, horizons, and telemetry.
    """

    def __init__(
        self,
        window_name: str = "F1 Racer - Onboard Vision & Centerline Detection",
        panel_size: int = 320,
        enable_display: bool = True,
    ):
        self.window_name = window_name
        self.panel_size = panel_size
        self.enable_display = enable_display
        self._tk_root = None
        self._tk_label = None
        self._tk_img_ref = None
        self._display_failed = False

    def draw_hud(
        self,
        rgb_tensor: torch.Tensor,
        detection_details: dict[str, Any],
        telemetry: dict[str, Any] | None = None,
    ) -> np.ndarray:
        """Constructs an annotated composite dashboard image.

        Args:
            rgb_tensor: (H, W, 3) or (1, H, W, 3) or (3, H, W) in [0, 255] or [0, 1].
            detection_details: Output dict from BatchedVisionTrackDetector.detect_detailed().
            telemetry: Optional dict with keys like 'speed_kmh', 'steer_deg', 'progress_pct', etc.

        Returns:
            composite_bgr: (H_comp, W_comp, 3) uint8 numpy array in BGR format for OpenCV.
        """
        # 1. Convert raw RGB tensor (torch.Tensor, ProxyArray, etc.) to uint8 numpy array
        curr = rgb_tensor
        if hasattr(curr, "torch"):
            curr = curr.torch
        if hasattr(curr, "detach"):
            curr = curr.detach()
        if hasattr(curr, "cpu"):
            curr = curr.cpu()
        img_np = curr.numpy() if hasattr(curr, "numpy") else np.asarray(curr)

        if img_np.ndim == 4:
            img_np = img_np[0]
        if img_np.shape[0] in (3, 4) and img_np.shape[-1] not in (3, 4):
            img_np = np.transpose(img_np, (1, 2, 0))

        if img_np.shape[-1] == 4:
            img_np = img_np[:, :, :3]

        if img_np.dtype != np.uint8:
            if img_np.max() <= 1.5:
                img_np = np.clip(img_np * 255.0, 0, 255).astype(np.uint8)
            else:
                img_np = np.clip(img_np, 0, 255).astype(np.uint8)

        # Convert RGB to BGR for OpenCV
        rgb_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        orig_h, orig_w = rgb_bgr.shape[:2]

        # 2. Extract Detection Details
        mask_raw = detection_details["mask"][0]
        if hasattr(mask_raw, "detach"):
            mask_raw = mask_raw.detach()
        if hasattr(mask_raw, "cpu"):
            mask_raw = mask_raw.cpu()
        mask_t = mask_raw.numpy() if hasattr(mask_raw, "numpy") else np.asarray(mask_raw)

        mask_u8 = np.clip(mask_t * 255.0, 0, 255).astype(np.uint8)
        mask_bgr = cv2.applyColorMap(mask_u8, cv2.COLORMAP_VIRIDIS)

        x_near_norm = float(detection_details["x_near"][0].item())
        x_far_norm = float(detection_details["x_far"][0].item())
        near_roi_range = detection_details["near_roi_range"]
        far_roi_range = detection_details["far_roi_range"]

        feat_raw = detection_details["features"][0]
        if hasattr(feat_raw, "detach"):
            feat_raw = feat_raw.detach()
        if hasattr(feat_raw, "cpu"):
            feat_raw = feat_raw.cpu()
        features = feat_raw.numpy() if hasattr(feat_raw, "numpy") else np.asarray(feat_raw)

        e_lat = float(features[0])
        e_head_rad = float(features[1])
        confidence = float(features[2])

        # 3. Resize panels to target HUD display size
        ps = self.panel_size
        cam_panel = cv2.resize(rgb_bgr, (ps, ps), interpolation=cv2.INTER_LINEAR)
        mask_panel = cv2.resize(mask_bgr, (ps, ps), interpolation=cv2.INTER_NEAREST)

        # Coordinate mappings for display size
        cam_cx = ps // 2
        near_y1 = int(ps * near_roi_range[0])
        near_y2 = int(ps * near_roi_range[1])
        far_y1 = int(ps * far_roi_range[0])
        far_y2 = int(ps * far_roi_range[1])
        near_y_mid = (near_y1 + near_y2) // 2
        far_y_mid = (far_y1 + far_y2) // 2

        # Reconstruct pixel positions in panel coords
        near_px = int(np.clip(cam_cx + x_near_norm * cam_cx, 0, ps - 1))
        far_px = int(np.clip(cam_cx + x_far_norm * cam_cx, 0, ps - 1))

        # --- A. Annotate Camera View ---
        # Draw vehicle optical center (reference centerline - red dashed line)
        for y in range(0, ps, 12):
            cv2.line(cam_panel, (cam_cx, y), (cam_cx, min(ps, y + 6)), (0, 0, 255), 1)

        # Draw Near ROI scanline band (Cyan)
        cv2.rectangle(cam_panel, (0, near_y1), (ps, near_y2), (255, 255, 0), 1)
        cv2.putText(cam_panel, "Near ROI", (6, near_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

        # Draw Far ROI scanline band (Orange)
        cv2.rectangle(cam_panel, (0, far_y1), (ps, far_y2), (0, 165, 255), 1)
        cv2.putText(cam_panel, "Far ROI", (6, far_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)

        # Draw Detected Centerline (Thick Bright Yellow)
        # Extend line to the bottom and top of view for clear visualization
        if abs(far_px - near_px) > 0 or abs(far_y_mid - near_y_mid) > 0:
            dy = far_y_mid - near_y_mid
            dx = far_px - near_px
            if abs(dy) > 1e-3:
                slope = dx / dy
                # Bottom intercept at y = ps
                bottom_x = int(near_px + (ps - near_y_mid) * slope)
                # Top projection
                top_x = int(far_px + (far_y1 - 15 - far_y_mid) * slope)
                cv2.line(cam_panel, (bottom_x, ps), (top_x, max(0, far_y1 - 15)), (0, 255, 255), 2, cv2.LINE_AA)

        # Highlight Near & Far Center points
        cv2.circle(cam_panel, (near_px, near_y_mid), 5, (255, 255, 0), -1)  # Cyan circle
        cv2.circle(cam_panel, (near_px, near_y_mid), 7, (0, 0, 0), 1)
        cv2.circle(cam_panel, (far_px, far_y_mid), 5, (0, 165, 255), -1)  # Orange circle
        cv2.circle(cam_panel, (far_px, far_y_mid), 7, (0, 0, 0), 1)

        # Lateral Error Vector Arrow (from camera optical center to near center)
        arrow_color = (0, 255, 0) if abs(e_lat) < 0.25 else ((0, 165, 255) if abs(e_lat) < 0.6 else (0, 0, 255))
        cv2.arrowedLine(cam_panel, (cam_cx, near_y_mid), (near_px, near_y_mid), arrow_color, 2, tipLength=0.2)

        # Title on Camera Panel
        cv2.rectangle(cam_panel, (0, 0), (ps, 22), (30, 30, 30), -1)
        cv2.putText(
            cam_panel, "ONBOARD CAMERA + CENTERLINE", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
        )

        # --- B. Annotate Segmentation Mask Panel ---
        # Add center points and detected path on mask panel
        cv2.line(mask_panel, (near_px, near_y_mid), (far_px, far_y_mid), (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(mask_panel, (near_px, near_y_mid), 4, (255, 255, 0), -1)
        cv2.circle(mask_panel, (far_px, far_y_mid), 4, (0, 165, 255), -1)
        cv2.line(mask_panel, (cam_cx, 0), (cam_cx, ps), (100, 100, 100), 1)

        # Title on Mask Panel
        cv2.rectangle(mask_panel, (0, 0), (ps, 22), (30, 30, 30), -1)
        cv2.putText(
            mask_panel, "TRACK BOUNDARY SEGMENTATION", (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1
        )

        # 4. Concatenate side-by-side
        video_strip = np.hstack([cam_panel, mask_panel])
        comp_w = ps * 2

        # 5. Build Telemetry Bottom Banner
        banner_h = 105
        banner = np.zeros((banner_h, comp_w, 3), dtype=np.uint8)
        banner[:] = (20, 20, 24)

        # Extract telemetry details
        speed_kmh = telemetry.get("speed_kmh", 0.0) if telemetry else 0.0
        speed_ms = telemetry.get("speed_ms", speed_kmh / 3.6) if telemetry else 0.0
        progress_pct = telemetry.get("progress_pct", 0.0) if telemetry else 0.0
        steer_deg = telemetry.get("steer_deg", 0.0) if telemetry else 0.0
        lap_time = telemetry.get("elapsed_s", 0.0) if telemetry else 0.0

        # Lateral error in physical cm (track half width = 0.6m = 60cm)
        lat_cm = e_lat * 60.0
        head_deg = math.degrees(e_head_rad)

        # Row 1: Key Metrics
        conf_color = (0, 255, 0) if confidence > 0.75 else ((0, 215, 255) if confidence > 0.45 else (0, 0, 255))
        cv2.putText(
            banner,
            f"SPEED: {speed_kmh:4.1f} km/h ({speed_ms:3.1f} m/s)",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            banner,
            f"CONFIDENCE: {confidence * 100:3.0f}%",
            (ps + 12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            conf_color,
            1,
        )

        # Row 2: Error metrics
        lat_color = (0, 255, 0) if abs(lat_cm) < 15.0 else ((0, 165, 255) if abs(lat_cm) < 35.0 else (0, 0, 255))
        cv2.putText(
            banner,
            f"LAT ERROR: {lat_cm:+5.1f} cm ({e_lat:+4.2f})",
            (12, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            lat_color,
            1,
        )
        cv2.putText(
            banner,
            f"HEADING ERR: {head_deg:+5.1f} deg",
            (ps + 12, 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        # Row 3: Progress & Steering
        cv2.putText(
            banner,
            f"PROGRESS: {progress_pct:4.1f}% | LAP: {lap_time:4.1f}s",
            (12, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            banner,
            f"STEER ANGLE: {steer_deg:+4.1f} deg",
            (ps + 12, 64),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 200),
            1,
        )

        # Row 4: Graphical Lateral Alignment Gauge
        gauge_y = 86
        gauge_x_start = 60
        gauge_w = comp_w - 120
        gauge_mid = gauge_x_start + gauge_w // 2

        # Background track bar
        cv2.rectangle(banner, (gauge_x_start, gauge_y), (gauge_x_start + gauge_w, gauge_y + 10), (45, 45, 50), -1)
        # Left and Right track boundaries
        cv2.line(banner, (gauge_x_start, gauge_y - 2), (gauge_x_start, gauge_y + 12), (255, 255, 255), 2)
        cv2.line(
            banner, (gauge_x_start + gauge_w, gauge_y - 2), (gauge_x_start + gauge_w, gauge_y + 12), (255, 255, 255), 2
        )
        # Centerline marker
        cv2.line(banner, (gauge_mid, gauge_y - 2), (gauge_mid, gauge_y + 12), (0, 255, 255), 2)

        # Vehicle position marker on gauge
        veh_gauge_x = int(np.clip(gauge_mid + e_lat * (gauge_w // 2), gauge_x_start, gauge_x_start + gauge_w))
        cv2.circle(banner, (veh_gauge_x, gauge_y + 5), 6, lat_color, -1)
        cv2.circle(banner, (veh_gauge_x, gauge_y + 5), 7, (255, 255, 255), 1)

        cv2.putText(
            banner, "LEFT", (gauge_x_start - 38, gauge_y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1
        )
        cv2.putText(
            banner,
            "RIGHT",
            (gauge_x_start + gauge_w + 6, gauge_y + 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (160, 160, 160),
            1,
        )

        # Combine video strip and banner
        composite = np.vstack([video_strip, banner])
        return composite

    def update(
        self,
        rgb_tensor: torch.Tensor,
        detection_details: dict[str, Any],
        telemetry: dict[str, Any] | None = None,
        wait_ms: int = 1,
    ) -> np.ndarray:
        """Draws HUD and updates the interactive GUI window if display is enabled."""
        frame = self.draw_hud(rgb_tensor, detection_details, telemetry)

        if self.enable_display and not self._display_failed:
            # 1. Initialize Tkinter window on first frame
            if self._tk_root is None:
                try:
                    import tkinter as tk

                    self._tk_root = tk.Tk()
                    self._tk_root.title(self.window_name)
                    self._tk_root.geometry("+30+60")
                    self._tk_label = tk.Label(self._tk_root, bg="#141418")
                    self._tk_label.pack(fill="both", expand=True)
                    self._tk_root.update()
                except Exception as e:
                    print(f"[HUD] Note: Graphical window not available: {e}")
                    self._display_failed = True

            # 2. Update window with new composite HUD frame
            if self._tk_root is not None and not self._display_failed:
                try:
                    from PIL import Image, ImageTk

                    # Convert BGR to RGB for PIL / Tkinter
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_frame)
                    self._tk_img_ref = ImageTk.PhotoImage(pil_img)
                    self._tk_label.configure(image=self._tk_img_ref)
                    self._tk_root.update_idletasks()
                    self._tk_root.update()
                except Exception:
                    pass

        return frame

    def close(self):
        """Closes live display window."""
        if self._tk_root is not None:
            with contextlib.suppress(Exception):
                self._tk_root.destroy()
            self._tk_root = None
