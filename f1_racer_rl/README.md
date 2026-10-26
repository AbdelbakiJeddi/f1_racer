# f1_racer_rl

The Isaac Lab task package for the F1 Racer project. It registers the Leatherback racing environment:

```text
F1RacerRl-F1-Racer-F1-Car
```

See the [root README](../README.md) for the project overview, prerequisites, and the full command reference.

## Install

This package resolves Isaac Lab from the local checkout declared in `[tool.uv.sources]` in `pyproject.toml`. Update
those paths first if your Isaac Lab directory is somewhere else.

```bash
uv sync
uv sync --extra isaacsim   # add Isaac Sim / Kit rendering
```

`uv sync` also installs the root `leatherback-racer` package from `..`.

## Train and play

```bash
# Headless training
uv run --extra isaacsim isaaclab train \
  --rl_library rsl_rl \
  --task F1RacerRl-F1-Racer-F1-Car \
  --num_envs 64

# Visualized training
uv run --extra isaacsim isaaclab train \
  --rl_library rsl_rl \
  --task F1RacerRl-F1-Racer-F1-Car \
  --num_envs 1 \
  --viz kit

# Play the latest checkpoint
uv run --extra isaacsim isaaclab play \
  --rl_library rsl_rl \
  --task F1RacerRl-F1-Racer-F1-Car \
  --checkpoint latest \
  --viz kit
```

Drop `--extra isaacsim` if your environment already contains Isaac Sim.

## Assets and task layout

```text
src/f1_racer_rl/
├── assets/
│   ├── leatherback.py            # Robot articulation and actuator configuration
│   └── data/                     # Leatherback USD, track USD/NPZ data
└── tasks/f1_racer/
    ├── config/f1_car/            # Gymnasium registration, environment config, PPO config
    └── mdp/                      # Observations, rewards, events, terminations, track oracle
```

Project-owned USD and NPZ files live under `src/f1_racer_rl/assets/data`. Use `F1_RACER_RL_ASSETS_DIR` from
`f1_racer_rl.assets` to resolve that directory from an editable checkout or an installed package.

## Test the registration

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest
```

The environment variable avoids unrelated system pytest plugins in some ROS/Isaac setups.
