# Leatherback Autonomous Racer: Procedural Arenas and Classical Vision Processing
from . import vision

def __getattr__(name):
    if name == "arena":
        from . import arena
        return arena
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["arena", "vision"]
