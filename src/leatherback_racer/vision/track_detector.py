"""GPU-accelerated batched classical computer vision detector for race track boundaries.

Extracts lateral error (e_lat), heading error (e_heading), and detection confidence
from onboard RGB camera images without any heavy neural networks.
"""

from __future__ import annotations

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchedVisionTrackDetector(nn.Module):
    """Batched, GPU-accelerated classical vision detector for racing track boundaries.
    
    The track features Racing Yellow boundary lines (R~1.0, G~0.82, B~0.0)
    on dark grey asphalt tarmac (RGB~0.12).
    
    Processing Steps:
    1. Color Isolation: Enhances yellow line contrast using chromaticity difference (R + G - 2*B).
    2. Dual-horizon Scanline Sampling: Samples near lookahead (bottom 20-35%) and far lookahead (middle 45-60%).
    3. Left/Right Boundary Detection: Identifies boundary line centroids across the vehicle view.
    4. Error Extraction: Computes lateral offset from image center and heading angular deflection.
    """

    def __init__(
        self,
        img_h: int = 128,
        img_w: int = 128,
        near_roi_range: Tuple[float, float] = (0.46, 0.52),
        far_roi_range: Tuple[float, float] = (0.40, 0.45),
        yellow_threshold: float = 0.25,
        expected_track_width_px: float = 64.0,
        near_half_width: float = 0.525,
        far_half_width: float = 0.990,
        longitudinal_delta_d: float = 0.532,
        half_track_width: float = 0.550,
    ):
        """
        Args:
            img_h: Camera image height (pixels).
            img_w: Camera image width (pixels).
            near_roi_range: Fraction from top for near lookahead (0.0=top, 1.0=bottom).
            far_roi_range: Fraction from top for far lookahead.
            yellow_threshold: Contrast threshold for line mask.
            expected_track_width_px: Approximate pixel distance between boundaries at near horizon.
            near_half_width: Calibrated physical visible half-width (m) at Near ROI ground plane.
            far_half_width: Calibrated physical visible half-width (m) at Far ROI ground plane.
            longitudinal_delta_d: Physical longitudinal ground distance (m) between Near and Far ROIs.
            half_track_width: Track corridor half-width (m) for normalizing lateral error.
        """
        super().__init__()
        self.img_h = img_h
        self.img_w = img_w
        self.near_roi_range = near_roi_range
        self.far_roi_range = far_roi_range
        self.yellow_threshold = yellow_threshold
        self.expected_track_width_px = expected_track_width_px
        self.near_half_width = near_half_width
        self.far_half_width = far_half_width
        self.longitudinal_delta_d = longitudinal_delta_d
        self.half_track_width = half_track_width

        # Normalized horizontal coordinate grid [-1.0, 1.0] across image width
        grid_x = torch.linspace(-1.0, 1.0, img_w)
        self.register_buffer("grid_x", grid_x)

    def extract_line_mask(self, rgb: torch.Tensor) -> torch.Tensor:
        """Isolates racing yellow lines against dark tarmac.
        
        Args:
            rgb: (N, 3, H, W) normalized in [0.0, 1.0].
        Returns:
            mask: (N, H, W) soft mask in [0.0, 1.0].
        """
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]

        # Yellow line formula: High R & G, very low B
        yellow_contrast = torch.clamp(r + g - 2.0 * b, min=0.0, max=1.0)

        # Soft threshold to remain differentiable if ever needed, and smooth noise
        mask = torch.sigmoid((yellow_contrast - self.yellow_threshold) * 12.0)
        return mask

    def _find_scanline_center(
        self,
        mask_slice: torch.Tensor,
        expected_half_span: float = 0.65,
        prev_estimate: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Finds track centerline from a horizontal image slice using dual boundary tracking.
        
        Args:
            mask_slice: (N, h_slice, W) line probability mask.
            expected_half_span: Expected distance from boundary line to corridor center in [-1, 1].
            prev_estimate: Optional (N, 1) prior center position to disambiguate.
        Returns:
            center_x: (N, 1) normalized horizontal position [-1.0, 1.0] where 0.0 is center.
            confidence: (N, 1) confidence score [0.0, 1.0].
        """
        N, _, W = mask_slice.shape
        u_mid = W // 2
        profile_1d = mask_slice.mean(dim=1)  # (N, W)

        left_p = profile_1d[:, :u_mid]
        right_p = profile_1d[:, u_mid:]

        mass_l = left_p.sum(dim=-1, keepdim=True) + 1e-6
        mass_r = right_p.sum(dim=-1, keepdim=True) + 1e-6

        grid_l = torch.linspace(-1.0, 0.0, u_mid, device=mask_slice.device, dtype=mask_slice.dtype).view(1, -1)
        grid_r = torch.linspace(0.0, 1.0, W - u_mid, device=mask_slice.device, dtype=mask_slice.dtype).view(1, -1)

        x_l = (left_p * grid_l).sum(dim=-1, keepdim=True) / mass_l
        x_r = (right_p * grid_r).sum(dim=-1, keepdim=True) / mass_r

        conf_l = torch.clamp(mass_l / (u_mid * 0.03), 0.0, 1.0)
        conf_r = torch.clamp(mass_r / ((W - u_mid) * 0.03), 0.0, 1.0)

        both_detected = (conf_l > 0.20) & (conf_r > 0.20)
        only_left = (conf_l > 0.20) & (~both_detected)
        only_right = (conf_r > 0.20) & (~both_detected)

        center_both = (x_l + x_r) * 0.5
        center_left = x_l + expected_half_span
        center_right = x_r - expected_half_span
        fallback = prev_estimate if prev_estimate is not None else torch.zeros_like(x_l)

        center_x = torch.where(
            both_detected,
            center_both,
            torch.where(only_left, center_left, torch.where(only_right, center_right, fallback)),
        )
        center_x = torch.clamp(center_x, -1.0, 1.0)

        confidence = torch.where(both_detected, (conf_l + conf_r) * 0.5, torch.max(conf_l, conf_r) * 0.75)
        return center_x, confidence

    def forward(
        self, rgb_images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Processes a batch of camera frames.
        
        Args:
            rgb_images: Tensor of shape (N, H, W, 3) or (N, 3, H, W)
                        in range [0, 255] (uint8) or [0.0, 1.0] (float32).
        Returns:
            errors: (N, 3) tensor of [e_lat, e_heading, confidence]
                    - e_lat: [-1.0, 1.0] normalized lateral deviation from centerline.
                    - e_heading: [-pi, pi] heading angular error relative to path.
                    - confidence: [0.0, 1.0] line detection quality indicator.
            mask: (N, H, W) binary/soft line mask for visualization/debugging.
        """
        # Ensure (N, 3, H, W)
        if rgb_images.dim() == 3:
            rgb_images = rgb_images.unsqueeze(0)
        if rgb_images.shape[-1] in (3, 4) and rgb_images.shape[1] not in (3, 4):
            rgb_images = rgb_images.permute(0, 3, 1, 2)

        # Normalize to float [0.0, 1.0]
        if rgb_images.dtype == torch.uint8:
            rgb = rgb_images.float() / 255.0
        elif rgb_images.max() > 1.5:
            rgb = rgb_images / 255.0
        else:
            rgb = rgb_images.clone()

        # Discard alpha channel if RGBA
        if rgb.shape[1] == 4:
            rgb = rgb[:, :3]

        H, W = rgb.shape[-2], rgb.shape[-1]

        # 1. Color Mask
        mask = self.extract_line_mask(rgb)  # (N, H, W)

        # 2. Slice Near Horizon (immediate vehicle lookahead)
        near_start = int(H * self.near_roi_range[0])
        near_end = int(H * self.near_roi_range[1])
        near_slice = mask[:, near_start:near_end, :]

        # 3. Slice Far Horizon (ahead along curvature)
        far_start = int(H * self.far_roi_range[0])
        far_end = int(H * self.far_roi_range[1])
        far_slice = mask[:, far_start:far_end, :]

        # 4. Find Centerlines at both horizons
        x_near, conf_near = self._find_scanline_center(near_slice, expected_half_span=0.75)
        x_far, conf_far = self._find_scanline_center(far_slice, expected_half_span=0.55, prev_estimate=x_near)

        # 5. Compute Geometric Errors with perspective ground projection
        # Physical lateral position in meters at near and far lookaheads
        y_near_phys = x_near * self.near_half_width
        y_far_phys = x_far * self.far_half_width

        # Lateral error: normalized by half track width so [-1.0, 1.0] corresponds to corridor edges
        e_lat = torch.clamp(y_near_phys / self.half_track_width, -1.0, 1.0)

        # Heading error: true physical angular deviation between near and far centers
        delta_y_phys = y_far_phys - y_near_phys
        e_heading = torch.atan2(delta_y_phys, torch.full_like(delta_y_phys, self.longitudinal_delta_d))

        # Overall confidence
        confidence = (conf_near + conf_far) * 0.5

        features = torch.cat([e_lat, e_heading, confidence], dim=-1)  # (N, 3)
        return features, mask
