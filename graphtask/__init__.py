"""
Expose public functionality.
"""
from importlib import metadata as importlib_metadata

from graphtask._task import Task

__all__ = ["config", "version", "Task"]


def get_version() -> str:
    """Returns the package version."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


config = None
version: str = get_version()
