"""Manual smoke test: drive the F1 Leatherback car interactively with keyboard.

Usage:
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/teleop_car.py

Controls (active inside the 3D simulation window):
    W / UP          : Throttle Forward (accelerate smoothly)
    S / DOWN        : Reverse / Brake
    A / LEFT        : Steer Left
    D / RIGHT       : Steer Right
    SPACE           : Handbrake (instant stop)
    R               : Reset car to track starting line
"""

import argparse
import math
import sys
import torch

from isaaclab.app import AppLauncher

# Setup CLI argument parser with Isaac Lab launcher args
parser = argparse.ArgumentParser(description="Manual drive smoke test for Leatherback F1 Racer.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Ensure 3D Kit GUI window opens by default
if args_cli.visualizer is None:
    args_cli.visualizer = "kit"

# Launch Isaac Lab simulation with GUI and cameras enabled
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import gymnasium as gym
import omni.appwindow

import f1_racer_rl.tasks  # noqa: F401
import leatherback_racer  # noqa: F401
from f1_racer_rl.tasks.f1_racer.config.f1_car.env_cfg import F1RacerEnvCfg
from isaaclab.utils.math import euler_xyz_from_quat


class KeyboardTeleop:
    """Listens to Omniverse window keyboard events for manual driving."""

    def __init__(self):
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        self.steer_cmd = 0.0      # [-1.0, 1.0]
        self.throttle_cmd = -1.0  # [-1.0, 1.0] (-1.0 is stopped / 0 speed)
        self.reset_requested = False

        self._keys_pressed = set()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_key_event)

    def _on_key_event(self, event, *args, **kwargs):
        key = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._keys_pressed.add(key)
            if key == "R":
                self.reset_requested = True
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys_pressed.discard(key)
        return True

    def update(self) -> tuple[float, float, bool]:
        """Returns (steer [-1, 1], throttle [-1, 1], reset_requested)."""
        # Steering input
        steer_target = 0.0
        if "A" in self._keys_pressed or "LEFT" in self._keys_pressed:
            steer_target += 1.0  # Turn left (+steer)
        if "D" in self._keys_pressed or "RIGHT" in self._keys_pressed:
            steer_target -= 1.0  # Turn right (-steer)

        # Smooth steering filter
        self.steer_cmd = 0.7 * self.steer_cmd + 0.3 * steer_target

        # Throttle input: -1.0 = stopped, +1.0 = full forward (2.5 m/s)
        if "SPACE" in self._keys_pressed:
            self.throttle_cmd = -1.0  # Handbrake
        elif "W" in self._keys_pressed or "UP" in self._keys_pressed:
            self.throttle_cmd = min(1.0, self.throttle_cmd + 0.08)
        elif "S" in self._keys_pressed or "DOWN" in self._keys_pressed:
            self.throttle_cmd = max(-1.0, self.throttle_cmd - 0.12)
        else:
            # Coast down to stop when no throttle pressed
            self.throttle_cmd = max(-1.0, self.throttle_cmd - 0.05)

        do_reset = self.reset_requested
        self.reset_requested = False
        return self.steer_cmd, self.throttle_cmd, do_reset

    def close(self):
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None


def main():
    print("\n" + "=" * 78)
    print(" F1 RACER: INTERACTIVE MANUAL DRIVE SMOKE TEST")
    print("=" * 78)
    print(" Controls (Click inside the 3D window to focus):")
    print("   [W / UP]     : Accelerate forward")
    print("   [S / DOWN]   : Brake / Decelerate")
    print("   [A / LEFT]   : Steer Left")
    print("   [D / RIGHT]  : Steer Right")
    print("   [SPACE]      : Handbrake (instant stop)")
    print("   [R]          : Reset car back to track start")
    print("=" * 78 + "\n")

    env_cfg = F1RacerEnvCfg()
    env_cfg.scene.num_envs = 1

    # Position the viewport camera behind and slightly above the car for great visibility
    env_cfg.sim.default_visualizer_cfg.eye = (19.5, 4.0, 1.8)
    env_cfg.sim.default_visualizer_cfg.lookat = (17.0, 3.9, 0.05)
    env_cfg.viewer.eye = (19.5, 4.0, 1.8)
    env_cfg.viewer.lookat = (17.0, 3.9, 0.05)

    env = gym.make("F1RacerRl-F1-Racer-F1-Car", cfg=env_cfg)
    robot = env.unwrapped.scene["robot"]
    device = env.unwrapped.device

    teleop = KeyboardTeleop()
    obs, info = env.reset()

    body_names = robot.data.body_names
    wheel_names = ["Wheel_Front_Left", "Wheel_Front_Right", "Wheel_Rear_Left", "Wheel_Rear_Right"]
    wheel_indices = [body_names.index(w) for w in wheel_names if w in body_names]
    steer_fl_idx = robot.data.joint_names.index("Knuckle__Upright__Front_Left")

    step = 0
    try:
        while simulation_app.is_running():
            steer, throttle, reset_req = teleop.update()

            if reset_req:
                print("\n[RESET] Teleop requested track reset...")
                obs, info = env.reset()
                continue

            action = torch.tensor([[steer, throttle]], device=device)
            obs, rewards, terminated, truncated, info = env.step(action)

            # Telemetry
            vel = robot.data.root_lin_vel_b[0].cpu().numpy()
            speed_ms = math.sqrt(vel[0] ** 2 + vel[1] ** 2)
            speed_kmh = speed_ms * 3.6
            root_pos = robot.data.root_pos_w[0].cpu().numpy()
            root_quat = robot.data.root_quat_w[0:1]
            roll, pitch, yaw = euler_xyz_from_quat(root_quat)
            pitch_deg = torch.rad2deg(pitch)[0].item()

            body_pos = robot.data.body_pos_w[0].cpu().numpy()
            wheel_zs = [body_pos[idx, 2] for idx in wheel_indices]
            knuckle_deg = torch.rad2deg(robot.data.joint_pos[0, steer_fl_idx]).item()

            if step % 15 == 0:
                print(
                    f"[Step {step:05d}] Speed: {speed_kmh:4.1f} km/h ({speed_ms:4.2f} m/s) | "
                    f"Steer: {knuckle_deg:+5.1f}° (cmd: {steer:+4.2f}) | "
                    f"Pitch: {pitch_deg:+5.1f}° | "
                    f"Wheels Z: FL={wheel_zs[0]:.3f} FR={wheel_zs[1]:.3f} RL={wheel_zs[2]:.3f} RR={wheel_zs[3]:.3f}"
                )

            step += 1
    finally:
        teleop.close()
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

