"""Leatherback Autonomous Racer: Core perception, track geometry, control, and evaluation."""

from .vision import BatchedVisionTrackDetector, TrackDetector, TrackVisionHUD
from .track import ProceduralTrack, TrackConfig, TrackOracle
from .teleop import KeyboardTeleop
from .evaluation import CheckpointLoader, SingleLapEvaluator, LapMetrics, VisionPipelineTester

__all__ = [
    # Perception
    "BatchedVisionTrackDetector",
    "TrackDetector",
    "TrackVisionHUD",
    # Track & Geometry
    "ProceduralTrack",
    "TrackConfig",
    "TrackOracle",
    # Control
    "KeyboardTeleop",
    # Evaluation
    "CheckpointLoader",
    "SingleLapEvaluator",
    "LapMetrics",
    "VisionPipelineTester",
]
