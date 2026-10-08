import math
import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Dict, Tuple, Optional

from .cnn_encoders import NatureCNNEncoder, SpatialSoftmaxEncoder


class VisionActor(nn.Module):
    """
    Vision-Only Actor Policy Network for the Leatherback Racer.
    
    Inputs:
    1. Visual Observation: Tensor of shape (B, C * k, H, W)
    2. Proprioception: (B, proprio_dim) [current steering angle, forward speed v_x]
    
    Outputs:
    Action Distribution Normal(mean, std) for continuous control:
    - a[0]: Steering command mapped to [-30 deg, +30 deg]
    - a[1]: Throttle command mapped to [0.0 m/s, 3.5 m/s]
    """

    def __init__(
        self,
        in_channels: int = 3,
        proprio_dim: int = 2,
        action_dim: int = 2,
        encoder_type: str = "nature_cnn",  # "nature_cnn" or "spatial_softmax"
        features_dim: int = 128,
        input_size: Tuple[int, int] = (84, 84),
        hidden_dims: Tuple[int, ...] = (128, 64),
    ):
        super().__init__()
        self.encoder_type = encoder_type

        # 1. Visual Feature Extractor
        if encoder_type == "spatial_softmax":
            self.encoder = SpatialSoftmaxEncoder(
                in_channels=in_channels,
                num_keypoints=16,
                features_dim=features_dim,
                input_size=input_size,
            )
        else:
            self.encoder = NatureCNNEncoder(
                in_channels=in_channels,
                features_dim=features_dim,
                input_size=input_size,
            )

        # 2. Multimodal Fusion MLP
        fused_dim = features_dim + proprio_dim
        layers = []
        curr_dim = fused_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ReLU())
            curr_dim = h

        self.mlp = nn.Sequential(*layers)

        # 3. Action Mean Head & Learnable Log Std
        self.action_mean = nn.Linear(curr_dim, action_dim)
        self.action_log_std = nn.Parameter(torch.full((action_dim,), -0.5))

    def forward(self, visual_obs: torch.Tensor, proprio: torch.Tensor) -> Tuple[Normal, Optional[torch.Tensor]]:
        """
        Args:
            visual_obs: (B, C * k, H, W)
            proprio: (B, proprio_dim)
        Returns:
            dist: Normal distribution over continuous 2D action space
            keypoints: Visual keypoints (if spatial_softmax), else None
        """
        keypoints = None
        if self.encoder_type == "spatial_softmax":
            z_vis, keypoints = self.encoder(visual_obs)
        else:
            z_vis = self.encoder(visual_obs)

        # Concatenate vision features + proprioception
        z_fused = torch.cat([z_vis, proprio], dim=-1)
        h = self.mlp(z_fused)

        mean = torch.tanh(self.action_mean(h))
        std = torch.exp(torch.clamp(self.action_log_std, min=-2.0, max=1.0))
        dist = Normal(mean, std)

        return dist, keypoints


class PrivilegedCritic(nn.Module):
    """
    Privileged State-Based Critic (Value Function V(s)).
    
    Used strictly during training (Asymmetric Actor-Critic / PPO).
    Takes ground-truth geometric and kinematic state from simulator oracle:
    - Lateral tracking error e_lat
    - Heading alignment error e_heading
    - Distance to left and right boundary lines
    - Forward speed v_x and lateral speed v_y
    - Yaw rate omega_z
    - Steering joint position
    - Previous step action commands
    
    Benefit: Drastically accelerates PPO convergence by eliminating value function noise.
    """

    def __init__(
        self,
        privileged_dim: int = 10,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
    ):
        super().__init__()
        layers = []
        curr_dim = privileged_dim
        for h in hidden_dims:
            layers.append(nn.Linear(curr_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.ELU())
            curr_dim = h

        layers.append(nn.Linear(curr_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, privileged_state: torch.Tensor) -> torch.Tensor:
        """
        Args:
            privileged_state: (B, privileged_dim)
        Returns:
            value: (B, 1) estimated state value V(s)
        """
        return self.net(privileged_state)


class AsymmetricActorCriticPolicy(nn.Module):
    """
    Complete Asymmetric Actor-Critic Agent for the Leatherback Racer.
    
    Combines:
    - Vision-Only Actor Policy (for deployment)
    - Privileged State Critic (for accelerated PPO training)
    """

    def __init__(
        self,
        in_channels: int = 3,
        proprio_dim: int = 2,
        action_dim: int = 2,
        privileged_dim: int = 10,
        encoder_type: str = "nature_cnn",
        features_dim: int = 128,
        input_size: Tuple[int, int] = (84, 84),
    ):
        super().__init__()
        self.actor = VisionActor(
            in_channels=in_channels,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            encoder_type=encoder_type,
            features_dim=features_dim,
            input_size=input_size,
        )
        self.critic = PrivilegedCritic(
            privileged_dim=privileged_dim,
        )

    def get_action(
        self,
        visual_obs: torch.Tensor,
        proprio: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Samples an action from the policy for environment stepping.
        
        Returns:
            action: (B, action_dim) in [-1.0, 1.0]
            log_prob: (B,) log probability of sampled action
            keypoints: Optional (B, 2*K) keypoint coordinates
        """
        dist, keypoints = self.actor(visual_obs, proprio)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        action = torch.clamp(action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, keypoints

    def evaluate_actions(
        self,
        visual_obs: torch.Tensor,
        proprio: torch.Tensor,
        privileged_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        PPO evaluation step computing log-probabilities, entropy, and baseline values.
        """
        dist, _ = self.actor(visual_obs, proprio)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(privileged_state).squeeze(-1)

        return values, log_prob, entropy
