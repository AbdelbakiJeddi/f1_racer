"""Playback script for evaluating trained F1 Racer policies in Isaac Lab 3.0.

Usage:
    # Play latest trained checkpoint with 3D window visualization:
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/play.py --checkpoint latest --viz kit

    # Play specific checkpoint with 4 vehicles:
    /home/abok-omen/IsaacLab-3.0/env_isaaclab/bin/python scripts/play.py --checkpoint logs/rsl_rl/f1_racer_leatherback/model_500.pt --num_envs 4 --viz kit
"""

import sys
import warp as wp

# Warp captures enable_backward when module is created; disable adjoint codegen for speed
wp.config.enable_backward = False

# Import task definitions to ensure Gymnasium registration
import f1_racer_rl.tasks  # noqa: F401
import leatherback_racer  # noqa: F401

from isaaclab_rl.entrypoints import run_play_cli


def main(argv: list[str] | None = None) -> int:
    """Run RSL-RL playback for F1 Racer."""
    if argv is None:
        argv = sys.argv[1:]

    # Inject defaults if not specified
    args = list(argv)
    if not any(arg.startswith("--rl_library") for arg in args):
        args = ["--rl_library", "rsl_rl"] + args
    if not any(arg.startswith("--task") for arg in args):
        args = ["--task", "F1RacerRl-F1-Racer-F1-Car"] + args

    return run_play_cli(args)


if __name__ == "__main__":
    raise SystemExit(main())
