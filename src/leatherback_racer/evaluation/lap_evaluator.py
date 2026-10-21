"""Single lap evaluation engine on procedural tracks for trained policies."""

from __future__ import annotations

import dataclasses
import math
import os
import shutil
import time
from typing import Optional, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

from ..track import ProceduralTrack, TrackConfig, TrackOracle
from ..vision import BatchedVisionTrackDetector, TrackVisionHUD
from .checkpoint import CheckpointLoader


@dataclasses.dataclass
class LapMetrics:
    """Performance telemetry gathered during a full lap evaluation."""
    lap_time_s: float = 0.0
    lap_length_m: float = 0.0
    lap_completed: bool = False
    average_speed_ms: float = 0.0
    top_speed_ms: float = 0.0
    mean_lateral_error_m: float = 0.0
    max_lateral_error_m: float = 0.0
    mean_heading_error_deg: float = 0.0
    off_track_steps: int = 0
    total_steps: int = 0

    def print_scorecard(self) -> None:
        """Prints a rich terminal performance scorecard."""
        print("\n" + "=" * 80)
        print(" 🏆 F1 RACER: SINGLE LAP PERFORMANCE SCORECARD")
        print("=" * 80)
        status = "✅ FULL LAP COMPLETED" if self.lap_completed else "⚠️ LAP INCOMPLETE / TIMEOUT"
        print(f" Status:               {status}")
        print(f" Lap Time:             {self.lap_time_s:.2f} seconds ({self.total_steps} steps)")
        print(f" Track Distance:       {self.lap_length_m:.1f} meters")
        print(f" Average Speed:        {self.average_speed_ms:.2f} m/s ({self.average_speed_ms * 3.6:.1f} km/h)")
        print(f" Top Speed:            {self.top_speed_ms:.2f} m/s ({self.top_speed_ms * 3.6:.1f} km/h)")
        print(f" Mean Lateral Error:   {self.mean_lateral_error_m * 100.0:.1f} cm")
        print(f" Max Lateral Error:    {self.max_lateral_error_m * 100.0:.1f} cm")
        print(f" Mean Heading Error:   {self.mean_heading_error_deg:.1f}°")
        print(f" Boundary Transgressions: {self.off_track_steps} steps")
        print("=" * 80 + "\n")


class SingleLapEvaluator:
    """Manages procedural track generation and single-lap policy evaluation."""

    @staticmethod
    def generate_eval_track(
        track_type: str = "organic",
        seed: int = 123,
        radius: float = 13.0,
        amplitude: float = 2.2,
        out_dir: str = "outputs/eval_track",
    ) -> Tuple[str, str, ProceduralTrack]:
        """Generates a procedural track USD mesh and NPZ metadata.
        
        Also synchronizes the generated track into active project assets.
        """
        from pxr import Usd, UsdGeom

        out_path = os.path.abspath(out_dir)
        os.makedirs(out_path, exist_ok=True)
        usd_path = os.path.join(out_path, f"track_{track_type}_{seed}.usd")
        npz_path = os.path.join(out_path, f"track_{track_type}_{seed}_data.npz")

        cfg = TrackConfig(
            track_type=track_type,
            seed=seed,
            radius=radius,
            amplitude=amplitude,
            track_width=1.2,
            line_thickness=0.08,
            num_samples=500,
        )
        track = ProceduralTrack(cfg)

        # Save NPZ metadata
        np.savez(
            npz_path,
            centerline=track.centerline,
            tangents=track.tangents,
            normals=track.normals,
            left_line=track.left_line,
            right_line=track.right_line,
            track_width=track.cfg.track_width,
            is_closed=track.is_closed,
        )

        # Save USD mesh
        if os.path.exists(usd_path):
            os.remove(usd_path)
        stage = Usd.Stage.CreateNew(usd_path)
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        track.build_usd_meshes(stage, root_path="/Track")
        stage.SetDefaultPrim(stage.GetPrimAtPath("/Track"))
        stage.GetRootLayer().Save()

        # Synchronize into assets/data
        try:
            from f1_racer_rl.assets import F1_RACER_RL_ASSETS_DIR
            target_usd = str(F1_RACER_RL_ASSETS_DIR / "track.usd")
            target_npz = str(F1_RACER_RL_ASSETS_DIR / "track_data.npz")
            shutil.copyfile(usd_path, target_usd)
            shutil.copyfile(npz_path, target_npz)
        except Exception as e:
            print(f"[Warning] Could not sync to F1_RACER_RL_ASSETS_DIR: {e}")

        total_len = float(np.sum(np.linalg.norm(np.diff(track.centerline, axis=0), axis=1)) + np.linalg.norm(track.centerline[0] - track.centerline[-1]))
        print(f"\n[Track Generation] Created new '{track_type}' track (seed={seed}):")
        print(f"  • Total Lap Length: {total_len:.1f} meters")
        print(f"  • Centerline Waypoints: {track.centerline.shape[0]} points")
        print(f"  • USD Path: {usd_path}")
        print(f"  • NPZ Path: {npz_path}")

        return usd_path, npz_path, track
