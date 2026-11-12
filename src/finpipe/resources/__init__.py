"""Configuration and task templates shipped with the distribution."""

from importlib.resources import files
from pathlib import Path


def template_directory(path: str | Path | None = None) -> Path:
    """Use an explicit template directory or the installed defaults."""
    return (
        Path(path) if path is not None else Path(str(files(__package__) / "templates"))
    )
