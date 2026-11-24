"""Run the GPU track detector over a recording from record_isaac_drive.py and score it against ground truth.

Run: .venv/bin/python scripts/eval_isaac_drive.py [--detector path/to/track_detector.py] [--video out.mp4]
Compare with the committed version:
    git show HEAD:src/leatherback_racer/vision/track_detector.py > /tmp/old_td.py
    .venv/bin/python scripts/eval_isaac_drive.py --detector /tmp/old_td.py
"""

import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument("--data", default=str(ROOT / "outputs/isaac_drive.npz"))
parser.add_argument("--detector", default=str(ROOT / "src/leatherback_racer/vision/track_detector.py"))
parser.add_argument("--video", default=None, help="write an overlay mp4 of all envs side by side")
parser.add_argument("--off_centre", type=float, default=0.4, help="|truth| above this counts for the wrong-side score")
args = parser.parse_args()

# load the detector file directly: the package __init__ pulls in Isaac-only modules
spec = importlib.util.spec_from_file_location("track_detector", args.detector)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
det = module.TrackDetector()

data = np.load(args.data)
frames, truth, new_ep = data["frames"], data["lat_norm"], data["new_episode"]
T, N = truth.shape

e_lat, x_near, x_far = np.zeros((T, N)), np.zeros((T, N)), np.zeros((T, N))
for t in range(T):
    if hasattr(det, "reset"):
        det.reset(torch.from_numpy(new_ep[t]))
    out = det.detect_detailed(torch.from_numpy(frames[t]))
    e_lat[t] = out["features"][:, 0].numpy()
    x_near[t], x_far[t] = out["x_near"][:, 0].numpy(), out["x_far"][:, 0].numpy()

# the oracle and the detector may use opposite sign conventions: take the sign that agrees best
sign = 1.0 if np.corrcoef(e_lat.ravel(), truth.ravel())[0, 1] >= 0 else -1.0
pred = sign * e_lat
off = np.abs(truth) > args.off_centre
wrong = off & (np.sign(pred) != np.sign(truth))
print(f"{Path(args.detector).name}: {T * N} frames, sign convention {sign:+.0f}")
print(f"  car off centre (|truth| > {args.off_centre}): {off.sum()} frames, centre on the WRONG side in {wrong.sum()}")
print(f"  e_lat vs truth: corr {np.corrcoef(pred.ravel(), truth.ravel())[0, 1]:.3f}")

if args.video:
    import cv2
    import imageio.v2 as iio

    S, H, W = 3, frames.shape[2], frames.shape[3]
    near_v, far_v = int(H * 0.49), int(H * 0.425)  # ROI centres of the default config
    to_u = lambda x: int(round((x + 1) / 2 * (W - 1) * S))  # noqa: E731
    with iio.get_writer(args.video, fps=15) as writer:
        for t in range(T):
            tiles = []
            for n in range(N):
                img = cv2.resize(frames[t, n], (W * S, H * S), interpolation=cv2.INTER_NEAREST)
                for v, x, col in ((near_v, x_near[t, n], (28, 126, 214)), (far_v, x_far[t, n], (121, 80, 242))):
                    cv2.line(img, (0, v * S), (W * S, v * S), col, 1)
                    cv2.circle(img, (to_u(x), v * S), 6, col, -1)
                bad = wrong[t, n]
                cv2.putText(img, f"e_lat {pred[t, n]:+.2f} truth {truth[t, n]:+.2f}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (232, 89, 12) if bad else (255, 255, 255), 1)
                tiles.append(img)
            writer.append_data(np.concatenate(tiles, axis=1))
    print(f"  video: {args.video}")
