import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VisionPreprocessor(nn.Module):
    """
    GPU-accelerated image preprocessing and visual feature conditioning.
    
    Performs:
    1. Channel normalization ([0, 255] uint8 -> [0.0, 1.0] float32).
    2. Region of Interest (ROI) hood/ground cropping (removes non-informative sky).
    3. Resolution resizing (e.g. 640x480 -> 84x84 or 128x128).
    4. Optional yellow-line contrast enhancement (isolates yellow boundary stripes).
    """

    def __init__(
        self,
        output_size: Tuple[int, int] = (84, 84),
        crop_top_ratio: float = 0.25,   # Crop top 25% (sky / horizon)
        crop_bottom_ratio: float = 0.05, # Crop bottom 5% (extreme vehicle hood edge)
        color_mode: str = "rgb",         # "rgb", "grayscale", "yellow_contrast"
    ):
        super().__init__()
        self.output_size = output_size
        self.crop_top_ratio = crop_top_ratio
        self.crop_bottom_ratio = crop_bottom_ratio
        self.color_mode = color_mode

    @property
    def num_channels(self) -> int:
        if self.color_mode in ["grayscale", "yellow_contrast"]:
            return 1
        elif self.color_mode == "rgb_with_yellow":
            return 4
        return 3

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img: Tensor of shape (B, C, H, W) or (B, H, W, C), uint8 or float.
        Returns:
            Preprocessed Tensor of shape (B, C_out, H_out, W_out) in [0.0, 1.0].
        """
        # Ensure (B, C, H, W)
        if img.dim() == 3:
            img = img.unsqueeze(0)
        if img.shape[-1] in [1, 3, 4] and img.shape[1] not in [1, 3, 4]:
            img = img.permute(0, 3, 1, 2)

        # Normalize to [0.0, 1.0]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        elif img.max() > 1.5:
            img = img / 255.0

        # Discard alpha channel if RGBA
        if img.shape[1] == 4 and self.color_mode != "rgb_with_yellow":
            img = img[:, :3, :, :]

        # ROI Crop: focus on the driving corridor ahead
        _, _, h, w = img.shape
        top_idx = int(h * self.crop_top_ratio)
        bottom_idx = int(h * (1.0 - self.crop_bottom_ratio))
        img = img[:, :, top_idx:bottom_idx, :]

        # Resize to target resolution
        if (img.shape[2], img.shape[3]) != self.output_size:
            img = F.interpolate(
                img,
                size=self.output_size,
                mode="bilinear",
                align_corners=False
            )

        # Color mode transformation
        if self.color_mode == "grayscale":
            # ITU-R BT.601 luminance weights
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            img = 0.299 * r + 0.587 * g + 0.114 * b
        elif self.color_mode == "yellow_contrast":
            # Yellow has high R and G, very low B: S = clamp(R + G - 2*B)
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            yellow = torch.clamp(r + g - 2.0 * b, 0.0, 1.0)
            img = yellow
        elif self.color_mode == "rgb_with_yellow":
            r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
            yellow = torch.clamp(r + g - 2.0 * b, 0.0, 1.0)
            img = torch.cat([r, g, b, yellow], dim=1)

        return img


class VisualDomainRandomizer(nn.Module):
    """
    Online visual domain randomization applied during training.
    Prevents CNN overfitting to specific track brightness, contrast, or lighting angles.
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
        noise_std: float = 0.02,
    ):
        super().__init__()
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        b, c, _, _ = x.shape
        # Random brightness scale
        brightness = torch.empty(b, 1, 1, 1, device=x.device).uniform_(*self.brightness_range)
        x = x * brightness

        # Random contrast scale around mean luminance
        contrast = torch.empty(b, 1, 1, 1, device=x.device).uniform_(*self.contrast_range)
        mean = x.mean(dim=(-2, -1), keepdim=True)
        x = (x - mean) * contrast + mean

        # Additive Gaussian sensor noise
        if self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise

        return torch.clamp(x, 0.0, 1.0)
