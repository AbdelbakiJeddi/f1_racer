"""Path helpers that keep runtime artifacts independent of the working directory."""

from __future__ import annotations

from pathlib import Path


def project_root() -> Path:
    """Return the repository root for an editable checkout, otherwise the CWD."""
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "scripts").is_dir():
            return parent
    return Path.cwd()


def resolve_project_path(path: str | Path) -> Path:
    """Resolve a relative path against the repository root instead of the CWD."""
    path = Path(path)
    return path if path.is_absolute() else project_root() / path
