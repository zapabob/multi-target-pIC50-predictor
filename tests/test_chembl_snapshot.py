"""Tests for fixed ChEMBL snapshot generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.data.chembl_snapshot import (
    build_chembl_endpoint_snapshot,
    build_chembl_pic50_snapshot,
)


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


class FakeEndpointActivityLoader:
    """Deterministic test double for endpoint ChEMBL activity loading."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, bool]] = []

    def load_activity(
        self,
        target_id: str,
        endpoint: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        self.calls.append((target_id, endpoint, force_refresh))
        offset = 0.0 if endpoint == "pIC50" else 1.0
        standard_type = "IC50" if endpoint == "pIC50" else "Ki"
        return pd.DataFrame(
            {
                "molecule_chembl_id": [f"{target_id}_{endpoint}_M{i}" for i in range(1, 7)],
                "canonical_smiles": [
                    "CC(=O)OC1=CC=CC=C1C(=O)O",
                    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                    "NCCc1ccc(O)c(O)c1",
                    "NCCc1c[nH]c2ccccc12",
                    "CN(C)CCOC(c1ccccc1)c1ccccc1",
                    "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
                ],
                "endpoint": endpoint,
                "standard_type": standard_type,
                "standard_value_nM": [1000.0, 500.0, 250.0, 125.0, 62.5, 2000.0],
                "p_value": [4.3 + offset, 4.8 + offset, 5.2 + offset, 5.7 + offset, 6.1 + offset, 3.9 + offset],
            }
        )


def test_build_chembl_endpoint_snapshot_writes_pic50_and_pki(tmp_path: Path):
    output_path = tmp_path / "chembl_endpoint_snapshot.csv"
    manifest_path = tmp_path / "chembl_endpoint_snapshot.manifest.json"
    loader = FakeEndpointActivityLoader()

    result = build_chembl_endpoint_snapshot(
        targets=["CHEMBL238"],
        endpoints=["pIC50", "pKi"],
        output_path=output_path,
        manifest_path=manifest_path,
        loader=loader,
        force_refresh=True,
        snapshot_id="endpoint-test",
    )

    assert result.csv_path == output_path
    assert loader.calls == [("CHEMBL238", "pIC50", True), ("CHEMBL238", "pKi", True)]

    snapshot = pd.read_csv(output_path)
    assert len(snapshot) == 12
    assert {"pIC50", "pKi"} == set(snapshot["endpoint"])
    assert {"IC50", "Ki"} == set(snapshot["standard_type"])
    assert {"train", "scaffold_test", "external"} <= set(snapshot["split"])
    assert {"activity_class", "diqr_outlier", "training_eligible"}.issubset(snapshot.columns)
    assert set(snapshot["training_eligible"]) == {True}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    csv_checksum = hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert manifest["snapshot_id"] == "endpoint-test"
    assert manifest["csv_sha256"] == csv_checksum
    assert manifest["targets"]["CHEMBL238"]["endpoints"]["pKi"]["row_count"] == 6
    assert manifest["split_policy"]["method"] == "sklearn_group_shuffle"
    assert manifest["filters"]["inactive_threshold_uM"] == 1000.0


class FakeEndpointQualityLoader:
    """Endpoint loader with a clear dIQR outlier and inactive high-value row."""

    def load_activity(
        self,
        target_id: str,
        endpoint: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        del target_id, force_refresh
        return pd.DataFrame(
            {
                "molecule_chembl_id": [f"M{i}" for i in range(1, 7)],
                "canonical_smiles": [
                    "CCO",
                    "CCCO",
                    "CCCCO",
                    "CCN",
                    "CCCN",
                    "c1ccccc1",
                ],
                "endpoint": endpoint,
                "standard_type": "IC50",
                "standard_value_nM": [
                    1000.0,
                    794.3,
                    631.0,
                    501.2,
                    398.1,
                    100_000_000.0,
                ],
                "p_value": [6.0, 6.1, 6.2, 6.3, 6.4, 1.0],
            }
        )


def test_build_chembl_endpoint_snapshot_flags_diqr_and_inactive_rows(tmp_path: Path):
    output_path = tmp_path / "quality.csv"
    manifest_path = tmp_path / "quality.manifest.json"

    build_chembl_endpoint_snapshot(
        targets=["CHEMBL238"],
        endpoints=["pIC50"],
        output_path=output_path,
        manifest_path=manifest_path,
        loader=FakeEndpointQualityLoader(),
        snapshot_id="quality",
        split_method="stable_hash",
    )

    snapshot = pd.read_csv(output_path)
    inactive = snapshot[snapshot["activity_class"] == "inactive_ge_threshold"]
    outliers = snapshot[snapshot["diqr_outlier"]]

    assert len(snapshot) == 6
    assert len(inactive) == 1
    assert len(outliers) == 1
    assert outliers.iloc[0]["molecule_chembl_id"] == "M6"
    assert snapshot["training_eligible"].sum() == 5

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    endpoint_manifest = manifest["targets"]["CHEMBL238"]["endpoints"]["pIC50"]
    assert endpoint_manifest["activity_class_counts"]["inactive_ge_threshold"] == 1
    assert endpoint_manifest["diqr_outlier_count"] == 1
    assert endpoint_manifest["training_eligible_count"] == 5


class FakeEndpointContextLoader:
    """Endpoint loader with repeated measurements in separate assay contexts."""

    def load_activity(
        self,
        target_id: str,
        endpoint: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        del target_id, endpoint, force_refresh
        return pd.DataFrame(
            {
                "molecule_chembl_id": ["M1", "M1", "M1"],
                "canonical_smiles": ["CCO", "CCO", "CCO"],
                "endpoint": ["pIC50", "pIC50", "pIC50"],
                "standard_type": ["IC50", "IC50", "IC50"],
                "standard_value_nM": [10_000.0, 1_000.0, 100.0],
                "p_value": [5.0, 6.0, 7.0],
                "assay_chembl_id": ["A1", "A1", "A2"],
                "assay_type": ["B", "B", "B"],
                "assay_type_description": ["Binding", "Binding", "Binding"],
                "assay_description": ["DAT uptake assay", "DAT uptake assay", "DAT binding assay"],
                "assay_organism": ["Homo sapiens", "Homo sapiens", "Homo sapiens"],
                "assay_cell_type": ["HEK293", "HEK293", "HEK293"],
                "assay_tissue": ["", "", ""],
                "bao_format": ["BAO_0000219", "BAO_0000219", "BAO_0000219"],
                "bao_label": ["cell-based format", "cell-based format", "cell-based format"],
                "assay_modality": ["uptake", "uptake", "binding"],
            }
        )


def test_build_chembl_endpoint_snapshot_aggregates_within_assay_context(tmp_path: Path):
    output_path = tmp_path / "context.csv"
    manifest_path = tmp_path / "context.manifest.json"

    build_chembl_endpoint_snapshot(
        targets=["CHEMBL238"],
        endpoints=["pIC50"],
        output_path=output_path,
        manifest_path=manifest_path,
        loader=FakeEndpointContextLoader(),
        snapshot_id="context",
        split_method="stable_hash",
        aggregation_method="median",
    )

    snapshot = pd.read_csv(output_path)
    assert len(snapshot) == 2
    assert {"uptake", "binding"} == set(snapshot["assay_modality"])

    uptake = snapshot[snapshot["assay_modality"] == "uptake"].iloc[0]
    assert uptake["measurement_count"] == 2
    assert uptake["p_value"] == 5.5
    assert uptake["aggregation_method"] == "median"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    endpoint_manifest = manifest["targets"]["CHEMBL238"]["endpoints"]["pIC50"]
    assert endpoint_manifest["measurement_count"] == 3
    assert endpoint_manifest["assay_modality_counts"] == {"binding": 1, "uptake": 1}
