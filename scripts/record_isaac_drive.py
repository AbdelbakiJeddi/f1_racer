"""Record onboard camera frames + ground truth from Isaac Sim, for offline detector checks.

The car weaves toward the boundaries so some frames show only one line. Frames are saved raw
(no video compression, which would shift the yellow colour the mask relies on).

Run: f1_racer_rl/.venv/bin/python scripts/record_isaac_drive.py --num_envs 4 --steps 400
Output: outputs/isaac_drive.npz with
    frames      (T, N, 128, 128, 3) uint8
    lat_norm    (T, N) oracle lateral error / half track width (car position, oracle sign convention)
    heading     (T, N) oracle heading error, rad
    new_episode (T, N) bool, True on the first frame of an episode
"""

from __future__ import annotations

import argparse
import math

parser = argparse.ArgumentParser(description="Record Isaac Sim camera frames with ground truth.")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=400)
parser.add_argument("--output", type=str, default="outputs/isaac_drive.npz")
args = parser.parse_args()

from isaaclab.app import AppLauncher  # noqa: E402

simulation_app = AppLauncher(headless=True, enable_cameras=True).app

import f1_racer_rl.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from f1_racer_rl.tasks.f1_racer.config.f1_car.env_cfg import F1RacerEnvCfg  # noqa: E402
from f1_racer_rl.tasks.f1_racer.mdp.track_oracle import TrackOracle  # noqa: E402
from isaaclab.utils.math import euler_xyz_from_quat  # noqa: E402

from leatherback_racer.paths import resolve_project_path  # noqa: E402


def main() -> None:
    cfg = F1RacerEnvCfg()
    cfg.scene.num_envs = args.num_envs
    env = gym.make("F1RacerRl-F1-Racer-F1-Car", cfg=cfg)
    base = env.unwrapped
    robot, camera = base.scene["robot"], base.scene["camera"]
    oracle = TrackOracle.get_instance(device=str(base.device))
    half_width = oracle.track_width / 2.0
    env.reset()

    # each env weaves with its own amplitude and phase, so the set covers centred and edge driving
    n = args.num_envs
    amp = torch.linspace(0.25, 0.5, n, device=base.device)
    phase = torch.linspace(0.0, math.pi, n, device=base.device)

    frames, lat, head, new_ep = [], [], [], []
    for step in range(args.steps):
        steer = amp * torch.sin(2 * math.pi * step / 90 + phase)
        action = torch.stack([steer, torch.full_like(steer, 0.45)], dim=-1)
        env.step(action)

        pos_2d = robot.data.root_pos_w.torch[:, :2] - base.scene.env_origins[:, :2]
        _, _, yaw = euler_xyz_from_quat(robot.data.root_quat_w.torch)
        q = oracle.query(pos_2d, yaw)

        frames.append(camera.data.output["rgb"][..., :3].cpu().numpy().astype(np.uint8))
        lat.append((q["lateral_error"][:, 0] / half_width).cpu().numpy())
        head.append(q["heading_error"][:, 0].cpu().numpy())
        new_ep.append((base.episode_length_buf == 0).cpu().numpy())
        if step % 50 == 0:
            print(f"step {step}/{args.steps}")

    out = resolve_project_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=np.stack(frames), lat_norm=np.stack(lat), heading=np.stack(head),
                        new_episode=np.stack(new_ep))
    print(f"saved {out}  frames {np.stack(frames).shape}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
