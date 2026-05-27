"""Build fixed ChEMBL pIC50 snapshots for reproducible evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from src.data.loader import ChEMBLDataLoader
from src.utils.config import target_config

DEFAULT_SNAPSHOT_COLUMNS = [
    "snapshot_id",
    "target",
    "target_name",
    "molecule_chembl_id",
    "canonical_smiles",
    "pIC50",
    "split",
    "scaffold_smiles",
    "source",
]


class ChEMBLPic50Loader(Protocol):
    """Minimal loader protocol used by the snapshot builder."""

    def load_chembl(self, target_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """Return preprocessed ChEMBL rows with molecule id, SMILES, and pIC50."""


@dataclass
class SnapshotBuildResult:
    """Paths and manifest from a snapshot build."""

    csv_path: Path
    manifest_path: Path
    row_count: int
    csv_sha256: str


def build_chembl_pic50_snapshot(
    *,
    targets: list[str],
    output_path: str | Path,
    manifest_path: str | Path,
    loader: ChEMBLPic50Loader | None = None,
    force_refresh: bool = False,
    max_rows_per_target: int | None = None,
    snapshot_id: str | None = None,
    random_seed: int = 42,
    scaffold_test_fraction: float = 0.15,
    external_fraction: float = 0.15,
) -> SnapshotBuildResult:
    """Build a deterministic ChEMBL pIC50 CSV and JSON manifest."""
    if not targets:
        raise ValueError("At least one ChEMBL target is required.")
    if scaffold_test_fraction < 0 or external_fraction < 0:
        raise ValueError("Split fractions must be non-negative.")
    if scaffold_test_fraction + external_fraction >= 1:
        raise ValueError("Split fractions must leave room for a train split.")

    snapshot_id = snapshot_id or datetime.now(timezone.utc).strftime("chembl-pic50-%Y%m%dT%H%M%SZ")
    loader = loader or ChEMBLDataLoader()
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)

    target_frames = []
    target_manifest: dict[str, Any] = {}
    for target in targets:
        target_df = loader.load_chembl(target, force_refresh=force_refresh)
        normalized_df = _normalize_target_frame(
            target_df,
            target=target,
            snapshot_id=snapshot_id,
            random_seed=random_seed,
            scaffold_test_fraction=scaffold_test_fraction,
            external_fraction=external_fraction,
            max_rows=max_rows_per_target,
        )
        target_frames.append(normalized_df)
        target_manifest[target] = {
            "target_name": target_config.get_target_name(target),
            "row_count": int(len(normalized_df)),
            "split_counts": {
                key: int(value)
                for key, value in normalized_df["split"].value_counts().sort_index().items()
            },
            "source": "ChEMBL",
        }

    snapshot_df = pd.concat(target_frames, ignore_index=True)
    snapshot_df = snapshot_df.sort_values(
        ["target", "split", "molecule_chembl_id", "canonical_smiles"],
        kind="mergesort",
    ).reset_index(drop=True)
    snapshot_df = snapshot_df[DEFAULT_SNAPSHOT_COLUMNS]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_df.to_csv(output_path, index=False, encoding="utf-8", lineterminator="\n")
    csv_sha256 = hashlib.sha256(output_path.read_bytes()).hexdigest()

    manifest = {
        "snapshot_id": snapshot_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "ChEMBL webresource client",
        "row_count": int(len(snapshot_df)),
        "csv_path": str(output_path.as_posix()),
        "csv_sha256": csv_sha256,
        "schema": DEFAULT_SNAPSHOT_COLUMNS,
        "targets": target_manifest,
        "filters": {
            "standard_type": "IC50",
            "standard_units": "nM",
            "pIC50_range": [0, 15],
            "max_rows_per_target": max_rows_per_target,
        },
        "split_policy": {
            "method": "stable_scaffold_hash",
            "random_seed": random_seed,
            "scaffold_test_fraction": scaffold_test_fraction,
            "external_fraction": external_fraction,
            "unit": "Murcko scaffold",
        },
        "context_of_use": {
            "intended_use": "fixed benchmark for model evaluation and pharma due diligence",
            "decision_role": "research_triage_only",
            "not_for": [
                "clinical_decision",
                "regulatory_submission",
                "manufacturing_release",
                "patient_care",
            ],
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return SnapshotBuildResult(
        csv_path=output_path,
        manifest_path=manifest_path,
        row_count=int(len(snapshot_df)),
        csv_sha256=csv_sha256,
    )


def _normalize_target_frame(
    df: pd.DataFrame,
    *,
    target: str,
    snapshot_id: str,
    random_seed: int,
    scaffold_test_fraction: float,
    external_fraction: float,
    max_rows: int | None,
) -> pd.DataFrame:
    required = {"molecule_chembl_id", "canonical_smiles", "pIC50"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Target {target} is missing columns: {sorted(missing)}")

    work_df = df[list(required)].copy()
    work_df = work_df.dropna(subset=["molecule_chembl_id", "canonical_smiles", "pIC50"])
    work_df["pIC50"] = pd.to_numeric(work_df["pIC50"], errors="coerce")
    work_df = work_df.dropna(subset=["pIC50"])
    work_df = work_df[(work_df["pIC50"] >= 0) & (work_df["pIC50"] <= 15)]
    work_df["canonical_smiles"] = work_df["canonical_smiles"].astype(str)
    work_df["molecule_chembl_id"] = work_df["molecule_chembl_id"].astype(str)
    work_df = work_df.drop_duplicates(subset=["canonical_smiles"], keep="first")
    work_df["scaffold_smiles"] = work_df["canonical_smiles"].map(_murcko_scaffold_smiles)
    work_df = work_df.dropna(subset=["scaffold_smiles"])
    work_df = work_df.sort_values(["molecule_chembl_id", "canonical_smiles"], kind="mergesort")

    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows_per_target must be positive when provided.")
        work_df = work_df.head(max_rows)

    if work_df.empty:
        raise ValueError(f"Target {target} has no valid rows after filtering.")

    work_df["snapshot_id"] = snapshot_id
    work_df["target"] = target
    work_df["target_name"] = target_config.get_target_name(target)
    work_df["split"] = _assign_scaffold_splits(
        work_df["scaffold_smiles"].tolist(),
        target=target,
        random_seed=random_seed,
        scaffold_test_fraction=scaffold_test_fraction,
        external_fraction=external_fraction,
    )
    work_df["source"] = "ChEMBL"
    return work_df


def _assign_scaffold_splits(
    scaffold_values: list[str],
    *,
    target: str,
    random_seed: int,
    scaffold_test_fraction: float,
    external_fraction: float,
) -> list[str]:
    scaffold_set = sorted(set(scaffold_values))
    ranked_scaffolds = sorted(
        scaffold_set,
        key=lambda value: _stable_hash_float(f"{random_seed}:{target}:{value}"),
    )

    n_scaffolds = len(ranked_scaffolds)
    if n_scaffolds == 1:
        split_by_scaffold = {ranked_scaffolds[0]: "train"}
    else:
        external_count = _bounded_holdout_count(n_scaffolds, external_fraction)
        remaining_after_external = max(0, n_scaffolds - external_count)
        scaffold_test_count = _bounded_holdout_count(
            remaining_after_external,
            scaffold_test_fraction,
        )
        split_by_scaffold = {}
        for index, scaffold in enumerate(ranked_scaffolds):
            if index < external_count:
                split_by_scaffold[scaffold] = "external"
            elif index < external_count + scaffold_test_count:
                split_by_scaffold[scaffold] = "scaffold_test"
            else:
                split_by_scaffold[scaffold] = "train"

    return [split_by_scaffold[value] for value in scaffold_values]


def _bounded_holdout_count(n_items: int, fraction: float) -> int:
    if n_items < 3 or fraction == 0:
        return 0
    return min(max(1, round(n_items * fraction)), n_items - 2)


def _stable_hash_float(value: str) -> float:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _murcko_scaffold_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    scaffold_smiles = Chem.MolToSmiles(scaffold)
    return scaffold_smiles or Chem.MolToSmiles(mol)
