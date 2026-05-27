"""Tests for fixed ChEMBL snapshot generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.chembl_snapshot import build_chembl_pic50_snapshot


class FakeChEMBLLoader:
    """Deterministic test double for ChEMBLDataLoader."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, bool]] = []

    def load_chembl(self, target_id: str, force_refresh: bool = False) -> pd.DataFrame:
        self.calls.append((target_id, force_refresh))
        return pd.DataFrame(
            {
                "molecule_chembl_id": [f"{target_id}_M{i}" for i in range(1, 7)],
                "canonical_smiles": [
                    "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                    "NCCc1ccc(O)c(O)c1",
                    "NCCc1c[nH]c2ccccc12",
                    "CN(C)CCOC(c1ccccc1)c1ccccc1",
                    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
                ],
                "pIC50": [4.3, 4.8, 5.2, 5.7, 6.1, 3.9],
            }
        )


def test_build_chembl_snapshot_writes_csv_manifest_and_checksum(tmp_path: Path):
    output_path = tmp_path / "chembl_pic50_snapshot.csv"
    manifest_path = tmp_path / "chembl_pic50_snapshot.manifest.json"
    loader = FakeChEMBLLoader()

    result = build_chembl_pic50_snapshot(
        targets=["CHEMBL238", "CHEMBL224"],
        output_path=output_path,
        manifest_path=manifest_path,
        loader=loader,
        force_refresh=True,
        snapshot_id="test-snapshot",
    )

    assert result.csv_path == output_path
    assert result.manifest_path == manifest_path
    assert loader.calls == [("CHEMBL238", True), ("CHEMBL224", True)]

    snapshot = pd.read_csv(output_path)
    assert len(snapshot) == 12
    assert {
        "snapshot_id",
        "target",
        "target_name",
        "molecule_chembl_id",
        "canonical_smiles",
        "pIC50",
        "split",
        "scaffold_smiles",
        "source",
    }.issubset(snapshot.columns)
    assert set(snapshot["split"]) == {"train", "scaffold_test", "external"}
    assert snapshot["snapshot_id"].nunique() == 1

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert manifest["snapshot_id"] == "test-snapshot"
    assert manifest["csv_sha256"] == csv_checksum
    assert manifest["row_count"] == 12
    assert manifest["targets"]["CHEMBL238"]["row_count"] == 6
    assert manifest["split_policy"]["method"] == "stable_scaffold_hash"


def test_build_chembl_snapshot_applies_per_target_row_limit(tmp_path: Path):
    output_path = tmp_path / "limited.csv"
    manifest_path = tmp_path / "limited.manifest.json"

    build_chembl_pic50_snapshot(
        targets=["CHEMBL238"],
        output_path=output_path,
        manifest_path=manifest_path,
        loader=FakeChEMBLLoader(),
        max_rows_per_target=4,
        snapshot_id="limited",
    )

    snapshot = pd.read_csv(output_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(snapshot) == 4
    assert manifest["targets"]["CHEMBL238"]["row_count"] == 4
    assert manifest["filters"]["max_rows_per_target"] == 4
