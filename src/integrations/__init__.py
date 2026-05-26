"""External structure and docking integration contracts."""

from .structure_pipeline import (
    AlphaFold3JobSpec,
    CommandLineDockingRunner,
    DockingJobSpec,
    ProteinTarget,
)

__all__ = [
    "AlphaFold3JobSpec",
    "CommandLineDockingRunner",
    "DockingJobSpec",
    "ProteinTarget",
]
