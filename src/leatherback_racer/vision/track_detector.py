"""GPU-accelerated batched classical computer vision detector for race track boundaries.

Extracts lateral error (e_lat), heading error (e_heading), and detection confidence
from onboard RGB camera images without any heavy neural networks.
"""

from __future__ import annotations

import torch
import torch.nn as nn


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
        near_roi_range: tuple[float, float] = (0.46, 0.52),
        far_roi_range: tuple[float, float] = (0.40, 0.45),
        yellow_threshold: float = 0.25,
        expected_track_width_px: float = 64.0,
        near_half_width: float = 0.525,
        far_half_width: float = 0.990,
        longitudinal_delta_d: float = 0.532,
        half_track_width: float = 0.550,
        near_half_span: float = 0.80,
        far_half_span: float = 0.55,
        line_level: float = 0.1,
        memory_len: int = 10,
        memory_decay: float = 0.7,
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
            near_half_span: Boundary line to corridor centre at the near ROI, normalized [-1, 1] units.
            far_half_span: Same at the far ROI.
            line_level: Profile value below which a column counts as empty (no line there).
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
        self.near_half_span = near_half_span
        self.far_half_span = far_half_span
        self.line_level = line_level

        # Normalized horizontal coordinate grid [-1.0, 1.0] across image width
        grid_x = torch.linspace(-1.0, 1.0, img_w)
        self.register_buffer("grid_x", grid_x)

        # Per-env centre estimates from the previous frame, (N, 1) normalized. Used to choose
        # which boundary a lone line is when the image alone cannot tell.
        self._prior_near: torch.Tensor | None = None
        self._prior_far: torch.Tensor | None = None
        self._prior_valid: torch.Tensor | None = None  # (N, 1) bool: the priors hold a real estimate
        # Ring buffer of the last `memory_len` frames per env: (N, memory_len, 3) = [x_near, x_far, confidence],
        # oldest first. The priors are the confidence-weighted, recency-weighted mean of its entries.
        self.memory_len = memory_len
        self.memory_decay = memory_decay
        self._memory: torch.Tensor | None = None

    def reset(self, env_mask: torch.Tensor | None = None) -> None:
        """Forgets the stored frames, for all envs or for those where `env_mask` (N,) is True."""
        if env_mask is None or self._memory is None:
            self._prior_near = self._prior_far = self._prior_valid = self._memory = None
            return
        self._memory[env_mask] = 0.0
        self._prior_near[env_mask] = 0.0
        self._prior_far[env_mask] = 0.0
        self._prior_valid[env_mask] = False

    def _update_memory(
        self, x_near: torch.Tensor, conf_near: torch.Tensor, x_far: torch.Tensor, conf_far: torch.Tensor
    ) -> None:
        """Pushes this frame into the ring buffer and recomputes the priors for the next frame.

        Each prior is sum(w * x) / sum(w) with w = confidence * decay**age, so frames that saw no line
        (confidence 0) leave the last good centre in place instead of resetting it to 0. The priors are
        valid once any stored frame has confidence. A wrong lone-line decision is stored too, so it repeats
        until a pair (confidence 1) outweighs it.
        """
        frame = torch.stack([x_near, x_far, (conf_near + conf_far) * 0.5], dim=-1).squeeze(1)  # (N, 3)
        self._memory = torch.roll(self._memory, shifts=-1, dims=1)
        self._memory[:, -1] = frame
        age = torch.arange(self.memory_len - 1, -1, -1, device=frame.device, dtype=frame.dtype)
        recency = self.memory_decay**age  # newest = 1
        weight = self._memory[:, :, 2] * recency  # (N, n)
        total = weight.sum(dim=1, keepdim=True)
        valid = total > 1e-6
        safe = total.clamp_min(1e-6)
        self._prior_near = (self._memory[:, :, 0] * weight).sum(dim=1, keepdim=True) / safe
        self._prior_far = (self._memory[:, :, 1] * weight).sum(dim=1, keepdim=True) / safe
        self._prior_valid = valid

    def extract_line_mask(self, rgb: torch.Tensor) -> torch.Tensor:
        """Isolates racing yellow lines against dark tarmac with zero noise floor.

        Args:
            rgb: (N, 3, H, W) normalized in [0.0, 1.0].
        Returns:
            mask: (N, H, W) mask in [0.0, 1.0] with zero background noise floor.
        """
        r = rgb[:, 0]
        g = rgb[:, 1]
        b = rgb[:, 2]

        # Yellow line formula: High R & G, very low B
        yellow_contrast = torch.clamp(r + g - 2.0 * b, min=0.0, max=1.0)

        # Zero-noise floor thresholding: pixels below threshold become strictly 0.0
        mask = torch.relu(yellow_contrast - self.yellow_threshold) / (1.0 - self.yellow_threshold + 1e-6)
        return torch.clamp(mask, 0.0, 1.0)

    @staticmethod
    def _halves(profile_1d: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Centre of mass and confidence of the line mass in each image half. All outputs (N, 1)."""
        W = profile_1d.shape[-1]
        u_mid = W // 2

        left_p = profile_1d[:, :u_mid]
        right_p = profile_1d[:, u_mid:]

        mass_l = left_p.sum(dim=-1, keepdim=True)
        mass_r = right_p.sum(dim=-1, keepdim=True)

        grid_l = torch.linspace(-1.0, 0.0, u_mid, device=profile_1d.device, dtype=profile_1d.dtype).view(1, -1)
        grid_r = torch.linspace(0.0, 1.0, W - u_mid, device=profile_1d.device, dtype=profile_1d.dtype).view(1, -1)

        x_l = (left_p * grid_l).sum(dim=-1, keepdim=True) / (mass_l + 1e-6)
        x_r = (right_p * grid_r).sum(dim=-1, keepdim=True) / (mass_r + 1e-6)

        # A visible line segment has integrated profile mass >= 2.0
        min_line_mass = 2.0
        conf_l = torch.clamp(mass_l / min_line_mass, 0.0, 1.0)
        conf_r = torch.clamp(mass_r / min_line_mass, 0.0, 1.0)

        both_detected = (conf_l > 0.60) & (conf_r > 0.60)
        only_left = (conf_l > 0.60) & (~both_detected)
        only_right = (conf_r > 0.60) & (~both_detected)
        return x_l, x_r, conf_l, conf_r, both_detected, only_left, only_right

    def _lone_line_side(self, mask_slice: torch.Tensor, expected_half_span: float) -> torch.Tensor:
        """Which boundary a lone line must be, from where the other boundary would have to appear.

        If the line were the left boundary, the right one would sit 2 * half_span to its right (and vice
        versa). When that spot is inside the image but the profile is empty there, that reading is refuted.

        Returns:
            side: (N, 1) +1 = the line is the left boundary, -1 = the right boundary, 0 = undecided
                  (both or neither reading refuted, or the band does not hold exactly one line).
        """
        profile_1d = mask_slice.mean(dim=1)  # (N, W)
        W = profile_1d.shape[-1]
        x_l, x_r, _, _, _, only_left, only_right = self._halves(profile_1d)
        x_line = torch.where(only_left, x_l, x_r)

        margin = 6.0 / (W - 1)  # 3 px, in normalized units
        offsets = torch.arange(-4, 5, device=profile_1d.device).view(1, -1)

        def refuted(other: torch.Tensor) -> torch.Tensor:
            in_view = other.abs() <= 1.0 - margin
            col = torch.round((other.clamp(-1.0, 1.0) + 1.0) * 0.5 * (W - 1)).long()
            window = profile_1d.gather(1, (col + offsets).clamp(0, W - 1))
            return in_view & (window.max(dim=-1, keepdim=True).values < self.line_level)

        not_left = refuted(x_line + 2.0 * expected_half_span)
        not_right = refuted(x_line - 2.0 * expected_half_span)
        side = not_right.float() - not_left.float()
        return torch.where(only_left | only_right, side, torch.zeros_like(side))

    def _find_scanline_center(
        self,
        mask_slice: torch.Tensor,
        expected_half_span: float = 0.80,
        prev_estimate: torch.Tensor | None = None,
        line_side: torch.Tensor | None = None,
        prior: torch.Tensor | None = None,
        trust_prior: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Finds track centerline from a horizontal image slice using dual boundary tracking.

        Args:
            mask_slice: (N, h_slice, W) line probability mask.
            expected_half_span: Expected distance from boundary line to corridor center in [-1, 1].
            prev_estimate: Optional (N, 1) centre to report when no line is seen.
            line_side: Optional (N, 1) from `_lone_line_side`: +1 left boundary, -1 right, 0 undecided.
            prior: Optional (N, 1) expected centre; a lone line takes the reading closest to it when the prior
                is trusted or `line_side` is undecided.
            trust_prior: Optional (N, 1) bool; where True the prior beats `line_side` (worn or shadowed paint
                also looks like a missing line, so the test only decides when nothing better is known).
        Returns:
            center_x: (N, 1) normalized horizontal position [-1.0, 1.0] where 0.0 is center.
            confidence: (N, 1) confidence score [0.0, 1.0].
        """
        profile_1d = mask_slice.mean(dim=1)  # (N, W)
        x_l, x_r, conf_l, conf_r, both_detected, only_left, only_right = self._halves(profile_1d)

        # Lone line: two readings, centre to its right (it is the left boundary) or to its left.
        x_line = torch.where(only_left, x_l, x_r)
        if_left_boundary = x_line + expected_half_span
        if_right_boundary = x_line - expected_half_span
        prior = prior if prior is not None else torch.zeros_like(x_line)
        closest_is_left = (if_left_boundary - prior).abs() < (if_right_boundary - prior).abs()
        if line_side is None:
            line_side = torch.zeros_like(x_line)
        if trust_prior is None:
            trust_prior = torch.zeros_like(x_line, dtype=torch.bool)
        is_left_boundary = torch.where(~trust_prior & (line_side != 0), line_side > 0, closest_is_left)
        center_single = torch.where(is_left_boundary, if_left_boundary, if_right_boundary)

        center_both = (x_l + x_r) * 0.5
        fallback = prev_estimate if prev_estimate is not None else torch.zeros_like(x_l)

        center_x = torch.where(
            both_detected,
            center_both,
            torch.where(only_left | only_right, center_single, fallback),
        )
        center_x = torch.clamp(center_x, -1.0, 1.0)

        confidence = torch.where(both_detected, (conf_l + conf_r) * 0.5, torch.max(conf_l, conf_r) * 0.85)
        return center_x, confidence

    def detect_detailed(self, rgb_images: torch.Tensor) -> dict:
        """Processes a batch of camera frames and returns detailed detection geometry.

        Args:
            rgb_images: Tensor of shape (N, H, W, 3) or (N, 3, H, W).
        Returns:
            dict containing:
                features: (N, 3) [e_lat, e_heading, confidence]
                mask: (N, H, W) binary line probability mask
                x_near: (N, 1) normalized near center in [-1, 1]
                x_far: (N, 1) normalized far center in [-1, 1]
                conf_near: (N, 1) confidence at near horizon
                conf_far: (N, 1) confidence at far horizon
                near_y: int near horizon scanline row
                far_y: int far horizon scanline row
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

        H, _W = rgb.shape[-2], rgb.shape[-1]

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
        N = mask.shape[0]
        if self._prior_near is None or self._prior_near.shape[0] != N or self._prior_near.device != mask.device:
            self._prior_near = torch.zeros(N, 1, device=mask.device, dtype=mask.dtype)
            self._prior_far = torch.zeros(N, 1, device=mask.device, dtype=mask.dtype)
            self._prior_valid = torch.zeros(N, 1, device=mask.device, dtype=torch.bool)
            self._memory = torch.zeros(N, self.memory_len, 3, device=mask.device, dtype=mask.dtype)

        # A lone line in both bands is the same boundary, so an undecided band borrows the other's answer.
        side_near = self._lone_line_side(near_slice, self.near_half_span)
        side_far = self._lone_line_side(far_slice, self.far_half_span)
        side_near, side_far = (
            torch.where(side_near == 0, side_far, side_near),
            torch.where(side_far == 0, side_near, side_far),
        )

        # A lone line is placed by, in order: a pair on the other band (converted through metres), last
        # frame's centre, the missing-line test. Worn or shadowed paint also looks missing, so the test is last.
        x_l_n, x_r_n, _, _, pair_near, _, _ = self._halves(near_slice.mean(dim=1))
        x_l_f, x_r_f, _, _, pair_far, _, _ = self._halves(far_slice.mean(dim=1))
        scale = self.far_half_width / self.near_half_width
        prior_near = torch.where(pair_far, (x_l_f + x_r_f) * 0.5 * scale, self._prior_near)
        prior_far = torch.where(pair_near, (x_l_n + x_r_n) * 0.5 / scale, self._prior_far)

        x_near, conf_near = self._find_scanline_center(
            near_slice, self.near_half_span, prev_estimate=prior_near, line_side=side_near, prior=prior_near,
            trust_prior=pair_far | self._prior_valid,
        )
        x_far, conf_far = self._find_scanline_center(
            far_slice, self.far_half_span, prev_estimate=prior_far, line_side=side_far, prior=prior_far,
            trust_prior=pair_near | self._prior_valid,
        )
        self._update_memory(x_near, conf_near, x_far, conf_far)

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

        near_y = (near_start + near_end) // 2
        far_y = (far_start + far_end) // 2

        return {
            "features": features,
            "mask": mask,
            "x_near": x_near,
            "x_far": x_far,
            "conf_near": conf_near,
            "conf_far": conf_far,
            "near_y": near_y,
            "far_y": far_y,
            "near_roi_range": self.near_roi_range,
            "far_roi_range": self.far_roi_range,
        }

    def forward(self, rgb_images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
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
        details = self.detect_detailed(rgb_images)
        return details["features"], details["mask"]


# Clean class alias
TrackDetector = BatchedVisionTrackDetector
