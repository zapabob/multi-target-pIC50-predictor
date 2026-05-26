"""Contracts for AlphaFold-style target structures and docking jobs."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProteinTarget:
    """Protein target metadata for structure and docking workflows."""

    name: str
    sequence: str | None = None
    chain_id: str = "A"
    structure_path: str | None = None
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AlphaFold3JobSpec:
    """AlphaFold3 protein-ligand complex specification."""

    name: str
    protein: ProteinTarget
    ligand_smiles: str
    model_seeds: list[int]

    def to_alphafold3_json(self) -> dict[str, Any]:
        """Export a local AlphaFold3-style JSON input payload."""
        if not self.protein.sequence:
            raise ValueError("AlphaFold3 JSON export requires a protein sequence.")
        return {
            "name": self.name,
            "sequences": [
                {
                    "protein": {
                        "id": self.protein.chain_id,
                        "sequence": self.protein.sequence,
                    }
                },
                {
                    "ligand": {
                        "id": "L",
                        "smiles": self.ligand_smiles,
                    }
                },
            ],
            "modelSeeds": self.model_seeds,
            "dialect": "alphafold3",
            "version": 1,
        }

    def to_project_json(self) -> dict[str, Any]:
        """Backward-compatible alias for project callers."""
        return self.to_alphafold3_json()

    def write_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_alphafold3_json(), indent=2), encoding="utf-8")
        return output_path


@dataclass
class DockingJobSpec:
    """Docking job definition independent of a specific docking engine."""

    name: str
    receptor_path: str
    ligand_path: str
    center: Sequence[float]
    box_size: Sequence[float]
    engine: str = "vina"
    exhaustiveness: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CommandLineDockingRunner:
    """Run a configured docking command template.

    Example template:
    ``["vina", "--receptor", "{receptor}", "--ligand", "{ligand}", "--center_x", "{center_x}"]``
    """

    def __init__(self, command_template: Sequence[str], dry_run: bool = True):
        self.command_template = list(command_template)
        self.dry_run = dry_run

    def build_command(self, job: DockingJobSpec, output_dir: str | Path) -> list[str]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        values = {
            "name": job.name,
            "receptor": job.receptor_path,
            "ligand": job.ligand_path,
            "center_x": str(job.center[0]),
            "center_y": str(job.center[1]),
            "center_z": str(job.center[2]),
            "size_x": str(job.box_size[0]),
            "size_y": str(job.box_size[1]),
            "size_z": str(job.box_size[2]),
            "exhaustiveness": str(job.exhaustiveness),
            "output_dir": str(output_dir),
        }
        return [part.format(**values) for part in self.command_template]

    def run(self, job: DockingJobSpec, output_dir: str | Path) -> dict[str, Any]:
        command = self.build_command(job, output_dir)
        if self.dry_run:
            return {"status": "dry_run", "command": command, "job": job.to_dict()}

        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        return {
            "status": "completed" if completed.returncode == 0 else "failed",
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "command": command,
            "job": job.to_dict(),
        }
