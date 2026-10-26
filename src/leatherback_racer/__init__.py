"""Leatherback Autonomous Racer: Core perception, track geometry, control, and evaluation."""

from .evaluation import CheckpointLoader, LapMetrics, SingleLapEvaluator, VisionPipelineTester
from .teleop import KeyboardTeleop
from .track import ProceduralTrack, TrackConfig, TrackOracle
from .vision import BatchedVisionTrackDetector, TrackDetector, TrackVisionHUD

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
