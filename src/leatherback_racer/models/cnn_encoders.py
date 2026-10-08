import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class NatureCNNEncoder(nn.Module):
    """
    Nature-CNN Architecture (Mnih et al., 2015).
    Standard baseline for visual reinforcement learning in robotics.
    
    Fast 3-stage downsampling conv layers with ReLU activations:
    - Conv 1: 8x8 kernel, stride 4, 32 channels
    - Conv 2: 4x4 kernel, stride 2, 64 channels
    - Conv 3: 3x3 kernel, stride 1, 64 channels
    - Dense projection to features_dim
    """

    def __init__(self, in_channels: int = 3, features_dim: int = 128, input_size: Tuple[int, int] = (84, 84)):
        super().__init__()
        self.in_channels = in_channels
        self.features_dim = features_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flatten dimension dynamically based on input size
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size[0], input_size[1])
            flatten_dim = self.conv(dummy).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(flatten_dim, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            z: (B, features_dim)
        """
        return self.linear(self.conv(x))


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax layer (Levine et al., 2016).
    Converts 2D feature maps into expected spatial coordinates (x, y) in [-1, 1].
    Ideal for boundary line tracking and visual servoing.
    """

    def __init__(self, height: int, width: int, temperature: float = 1.0):
        super().__init__()
        self.height = height
        self.width = width
        self.temperature = temperature

        # Create coordinate grids in [-1.0, 1.0]
        pos_x, pos_y = torch.meshgrid(
            torch.linspace(-1.0, 1.0, width),
            torch.linspace(-1.0, 1.0, height),
            indexing="xy",
        )
        self.register_buffer("pos_x", pos_x.reshape(1, 1, height * width))
        self.register_buffer("pos_y", pos_y.reshape(1, 1, height * width))

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            feature_map: (B, K, H, W)
        Returns:
            keypoints: (B, 2 * K) coordinates [x_1, y_1, x_2, y_2, ..., x_K, y_K]
        """
        b, k, h, w = feature_map.shape
        flat_features = feature_map.reshape(b, k, h * w) / self.temperature
        softmax_attention = F.softmax(flat_features, dim=-1)

        expected_x = torch.sum(softmax_attention * self.pos_x, dim=-1, keepdim=True)
        expected_y = torch.sum(softmax_attention * self.pos_y, dim=-1, keepdim=True)

        # Interleave (x, y) coordinates: shape (B, K, 2) -> (B, 2 * K)
        keypoints = torch.cat([expected_x, expected_y], dim=-1).reshape(b, 2 * k)
        return keypoints


class SpatialSoftmaxEncoder(nn.Module):
    """
    Spatial Keypoint Feature Extractor for Lane & Boundary Line Tracking.
    
    Extracts K visual keypoints tracking line landmarks in the image,
    providing high sample efficiency and natural interpretability.
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_keypoints: int = 16,
        features_dim: int = 128,
        input_size: Tuple[int, int] = (84, 84),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.features_dim = features_dim

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, num_keypoints, kernel_size=3, stride=1, padding=1),
        )

        # Compute output feature map dimensions for Spatial Softmax
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, input_size[0], input_size[1])
            f_map = self.conv(dummy)
            _, _, h_out, w_out = f_map.shape

        self.spatial_softmax = SpatialSoftmax(h_out, w_out)

        # Project 2*K keypoint coordinates to latent representation
        self.projection = nn.Sequential(
            nn.Linear(2 * num_keypoints, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            z: (B, features_dim)
            keypoints: (B, 2 * K) raw (x, y) coordinates for visualization
        """
        feature_maps = self.conv(x)
        keypoints = self.spatial_softmax(feature_maps)
        z = self.projection(keypoints)
        return z, keypoints
