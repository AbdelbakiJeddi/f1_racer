import torch
import torch.nn as nn


class BatchedFrameStack(nn.Module):
    """
    Maintains a rolling ring buffer of past camera frames for batched RL environments.
    
    Stacking k consecutive frames allows the CNN to infer:
    - Vehicle forward velocity v_x
    - Lateral slip rate
    - Curve approach velocity (optical flow)
    """

    def __init__(self, num_envs: int, stack_size: int, num_channels: int, height: int, width: int, device: str = "cpu"):
        super().__init__()
        self.num_envs = num_envs
        self.stack_size = stack_size
        self.num_channels = num_channels
        self.height = height
        self.width = width
        self.device = device

        # Buffer shape: (num_envs, stack_size * num_channels, height, width)
        self.buffer = torch.zeros(
            (num_envs, stack_size * num_channels, height, width),
            dtype=torch.float32,
            device=device,
        )

    def reset(self, env_ids: torch.Tensor, initial_frame: torch.Tensor):
        """
        Resets selected environments by populating all historical slots with initial_frame.
        
        Args:
            env_ids: 1D tensor of environment indices to reset.
            initial_frame: Tensor of shape (len(env_ids), num_channels, height, width).
        """
        for i in range(self.stack_size):
            start_c = i * self.num_channels
            end_c = start_c + self.num_channels
            self.buffer[env_ids, start_c:end_c] = initial_frame

    def step(self, new_frame: torch.Tensor) -> torch.Tensor:
        """
        Appends new_frame to the buffer (shifting older frames back) and returns full stacked tensor.
        
        Args:
            new_frame: Tensor of shape (num_envs, num_channels, height, width).
        Returns:
            stacked_obs: Tensor of shape (num_envs, stack_size * num_channels, height, width).
        """
        # Shift older frames: [t-2, t-1, t] -> drop oldest [t-2]
        shift = self.num_channels
        self.buffer[:, :-shift] = self.buffer[:, shift:].clone()
        # Insert latest frame at the end
        self.buffer[:, -shift:] = new_frame
        return self.buffer
