"""Lone-line check: a left line seen on the right of the image must not put the centre outside the road.

Run: .venv/bin/python scripts/check_lone_line.py
"""

import importlib.util
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "labs"))
from helpers.vision_preview import SyntheticTrack  # noqa: E402
from lanedet.camera import Camera  # noqa: E402
from lanedet.detector import LaneDetector  # noqa: E402

# load the GPU detector file directly: the package __init__ pulls in Isaac-only modules
spec = importlib.util.spec_from_file_location("track_detector", ROOT / "src/leatherback_racer/vision/track_detector.py")
track_detector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(track_detector)

base = SyntheticTrack()
# car on the track, 5 cm inside the left line; the road bends right, so the left line swings right of centre
bend = replace(base, offset=0.25, curvature=1.2)
frame = bend.render()

# lab detector, no memory
cam = Camera(base.width, base.height, base.focal, base.cam_height, base.v_horizon)
lab = LaneDetector(cam)
assert lab(frame).e_lat > 0.9, "lab: lone left line read as the right one"

# GPU detector, constants recomputed for the synthetic camera (same rows as the Isaac config)
half_width = lambda v: base.cam_height * base.width / 2 / (v - base.v_horizon)  # noqa: E731  m per normalized unit
v_near, v_far = (58 + 66) / 2, (51 + 57) / 2
gpu = track_detector.TrackDetector(
    near_half_width=half_width(v_near),
    far_half_width=half_width(v_far),
    longitudinal_delta_d=base.distance(v_far) - base.distance(v_near),
    half_track_width=0.30,
    near_half_span=0.30 / half_width(v_near),
    far_half_span=0.30 / half_width(v_far),
)
batch = torch.from_numpy(np.stack([frame, base.render()]))  # bend case + centred car
e_lat = gpu(batch)[0][:, 0]
assert e_lat[0] > 0.9, "gpu: lone left line read as the right one"
assert abs(e_lat[1]) < 0.1, "gpu: centred car not centred"

# reset only touches the masked envs
gpu._prior_valid[:] = True
gpu.reset(torch.tensor([True, False]))
assert gpu._prior_valid.flatten().tolist() == [False, True], "gpu: reset mask"

print("lone-line check OK")
