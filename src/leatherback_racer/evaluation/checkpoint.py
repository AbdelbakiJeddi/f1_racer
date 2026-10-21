"""Model checkpoint utilities for F1 Racer policies."""

from __future__ import annotations

import glob
import os
from typing import Optional
import torch
import torch.nn as nn


class CheckpointLoader:
    """Manages loading and restoring trained policy weights from checkpoints."""

    @staticmethod
    def find_latest(log_dir: str = "logs/rsl_rl/f1_racer_leatherback") -> str:
        """Finds the most recent model_*.pt checkpoint in the given directory tree."""
        pattern = os.path.join(log_dir, "*", "*.pt")
        ckpts = glob.glob(pattern)
        if not ckpts:
            # Also check direct child pt files
            ckpts = glob.glob(os.path.join(log_dir, "*.pt"))
        if not ckpts:
            raise FileNotFoundError(f"No trained checkpoints found matching: {pattern}")
        ckpts.sort(key=os.path.getmtime, reverse=True)
        return ckpts[0]

    @staticmethod
    def load_actor_policy(
        checkpoint_path: str,
        input_dim: int = 7,
        hidden_dims: tuple[int, ...] = (256, 128, 64),
        action_dim: int = 2,
        device: str = "cuda",
    ) -> nn.Module:
        """Reconstructs and loads the trained actor MLP from an RSL-RL checkpoint.

        Args:
            checkpoint_path: Path to the .pt file or 'latest'.
            input_dim: Observation dimension (default 7 for deployable policy).
            hidden_dims: MLP layer widths (default [256, 128, 64]).
            action_dim: Action dimension (default 2 for [steer, throttle]).
            device: Target torch device.

        Returns:
            nn.Module: Loaded policy in eval mode.
        """
        if checkpoint_path == "latest":
            checkpoint_path = CheckpointLoader.find_latest()

        print(f"[Policy] Loading weights from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt["actor_state_dict"]

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ELU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, action_dim))

        mlp = nn.Sequential(*layers).to(device)

        # Extract only mlp.* keys from actor_state_dict
        mlp_dict = {k[4:]: v for k, v in state_dict.items() if k.startswith("mlp.")}
        mlp.load_state_dict(mlp_dict)
        mlp.eval()
        print(f"  ✅ Actor MLP ({input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {action_dim}) loaded successfully.")
        return mlp
