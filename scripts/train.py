"""Training script for F1 Racer using RSL-RL PPO in Isaac Lab 3.0.

Usage:
    # Quick headless training (fastest):
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 64 --max_iterations 500

    # Training with live 3D window visualization:
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --num_envs 16 --viz kit

    # Resume from checkpoint:
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/train.py --checkpoint latest
"""

import sys
import warp as wp

# Warp captures enable_backward when module is created; disable adjoint codegen for speed
wp.config.enable_backward = False

# Import task definitions to ensure Gymnasium registration
import f1_racer_rl.tasks  # noqa: F401
import leatherback_racer  # noqa: F401

from isaaclab_rl.entrypoints import run_train_cli


def main(argv: list[str] | None = None) -> int:
    """Run RSL-RL training for F1 Racer."""
    if argv is None:
        argv = sys.argv[1:]

    # Inject defaults if not specified
    args = list(argv)
    if not any(arg.startswith("--rl_library") for arg in args):
        args = ["--rl_library", "rsl_rl"] + args
    if not any(arg.startswith("--task") for arg in args):
        args = ["--task", "F1RacerRl-F1-Racer-F1-Car"] + args

    return run_train_cli(args)


if __name__ == "__main__":
    from torch.distributed.elastic.multiprocessing.errors import record

    raise SystemExit(record(main)())
