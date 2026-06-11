"""Build fixed ChEMBL pIC50 snapshots for reproducible evaluation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
from chembl_webresource_client.new_client import new_client
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

from src.data.loader import ChEMBLDataLoader
from src.utils.config import target_config

DEFAULT_DIQ_MULTIPLIER = 2.0
DEFAULT_INACTIVE_THRESHOLD_UM = 1000.0
ENDPOINT_SPLIT_METHODS = {"stable_hash", "sklearn_group_shuffle"}
ENDPOINT_AGGREGATION_METHODS = {"none", "median", "robust_mean"}

ASSAY_CONTEXT_COLUMNS = [
    "assay_chembl_id",
    "assay_type",
    "assay_type_description",
    "assay_description",
    "assay_organism",
    "assay_cell_type",
    "assay_tissue",
    "bao_format",
    "bao_label",
    "assay_modality",
]

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

ENDPOINT_TO_STANDARD_TYPE = {
    "pIC50": "IC50",
    "pKi": "Ki",
}

DEFAULT_ENDPOINT_SNAPSHOT_COLUMNS = [
    "snapshot_id",
    "target",
    "target_name",
    "endpoint",
    "standard_type",
    "molecule_chembl_id",
    "canonical_smiles",
    *ASSAY_CONTEXT_COLUMNS,
    "assay_context_key",
    "p_value",
    "standard_value_nM",
    "measurement_count",
    "aggregation_method",
    "activity_class",
    "diqr_outlier",
    "training_eligible",
    "split",
    "scaffold_smiles",
    "source",
]


class ChEMBLPic50Loader(Protocol):
    """Minimal loader protocol used by the snapshot builder."""

    def load_chembl(self, target_id: str, force_refresh: bool = False) -> pd.DataFrame:
        """Return preprocessed ChEMBL rows with molecule id, SMILES, and pIC50."""


class ChEMBLEndpointActivityLoader(Protocol):
    """Minimal loader protocol used by the endpoint snapshot builder."""

    def load_activity(
        self,
        target_id: str,
        endpoint: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Return ChEMBL rows with molecule id, SMILES, endpoint, and p_value."""


@dataclass
class SnapshotBuildResult:
    """Paths and manifest from a snapshot build."""

    csv_path: Path
    manifest_path: Path
    row_count: int
    csv_sha256: str


class ChEMBLEndpointDataLoader:
    """Load endpoint-specific ChEMBL activity rows for CPU snapshot builds."""

    def __init__(
        self,
        max_records: int | None = None,
        *,
        include_assay_details: bool = True,
    ) -> None:
        self.max_records = max_records
        self.include_assay_details = include_assay_details

    def load_activity(
        self,
        target_id: str,
        endpoint: str,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch IC50 or Ki rows in nM from the ChEMBL webresource client."""
        del force_refresh
        standard_type = _standard_type_for_endpoint(endpoint)
        activities = new_client.activity.filter(
            target_chembl_id=target_id,
            standard_type=standard_type,
            standard_units="nM",
            standard_relation="=",
        ).only(
            [
                "activity_id",
                "molecule_chembl_id",
                "canonical_smiles",
                "standard_value",
                "standard_relation",
                "pchembl_value",
                "assay_chembl_id",
                "assay_description",
                "assay_type",
                "bao_format",
                "bao_label",
                "target_organism",
                "document_chembl_id",
                "document_year",
                "molecule_pref_name",
                "parent_molecule_chembl_id",
            ]
        )
        if self.max_records is not None:
            activities = activities[: self.max_records]
        activity_records = list(activities)
        if not activity_records:
            raise ValueError(f"No {standard_type} nM rows found for target {target_id}")
        raw_df = pd.DataFrame(activity_records)
        if self.include_assay_details:
            raw_df = _attach_assay_details(raw_df)
        return _preprocess_endpoint_activity(raw_df, endpoint=endpoint)


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


def build_chembl_endpoint_snapshot(
    *,
    targets: list[str],
    endpoints: list[str],
    output_path: str | Path,
    manifest_path: str | Path,
    loader: ChEMBLEndpointActivityLoader | None = None,
    force_refresh: bool = False,
    max_rows_per_target_endpoint: int | None = None,
    snapshot_id: str | None = None,
    random_seed: int = 42,
    scaffold_test_fraction: float = 0.15,
    external_fraction: float = 0.15,
    split_method: str = "sklearn_group_shuffle",
    diq_multiplier: float = DEFAULT_DIQ_MULTIPLIER,
    inactive_threshold_uM: float = DEFAULT_INACTIVE_THRESHOLD_UM,
    aggregation_method: str = "median",
) -> SnapshotBuildResult:
    """Build a deterministic ChEMBL pIC50/pKi endpoint CSV and JSON manifest."""
    if not targets:
        raise ValueError("At least one ChEMBL target is required.")
    if not endpoints:
        raise ValueError("At least one endpoint is required.")
    unsupported = sorted(set(endpoints).difference(ENDPOINT_TO_STANDARD_TYPE))
    if unsupported:
        raise ValueError(f"Unsupported endpoints: {unsupported}")
    if scaffold_test_fraction < 0 or external_fraction < 0:
        raise ValueError("Split fractions must be non-negative.")
    if scaffold_test_fraction + external_fraction >= 1:
        raise ValueError("Split fractions must leave room for a train split.")
    if split_method not in ENDPOINT_SPLIT_METHODS:
        raise ValueError(f"Unsupported split_method: {split_method}")
    if diq_multiplier <= 0:
        raise ValueError("diq_multiplier must be positive.")
    if inactive_threshold_uM <= 0:
        raise ValueError("inactive_threshold_uM must be positive.")
    if aggregation_method not in ENDPOINT_AGGREGATION_METHODS:
        raise ValueError(f"Unsupported aggregation_method: {aggregation_method}")

    snapshot_id = snapshot_id or datetime.now(timezone.utc).strftime(
        "chembl-endpoint-%Y%m%dT%H%M%SZ"
    )
    loader = loader or ChEMBLEndpointDataLoader(max_records=max_rows_per_target_endpoint)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path)

    frames = []
    target_manifest: dict[str, Any] = {}
    for target in targets:
        endpoint_manifest: dict[str, Any] = {}
        for endpoint in endpoints:
            target_df = loader.load_activity(target, endpoint, force_refresh=force_refresh)
            normalized_df = _normalize_endpoint_frame(
                target_df,
                target=target,
                endpoint=endpoint,
                snapshot_id=snapshot_id,
                random_seed=random_seed,
                scaffold_test_fraction=scaffold_test_fraction,
                external_fraction=external_fraction,
                split_method=split_method,
                diq_multiplier=diq_multiplier,
                inactive_threshold_uM=inactive_threshold_uM,
                aggregation_method=aggregation_method,
                max_rows=max_rows_per_target_endpoint,
            )
            frames.append(normalized_df)
            endpoint_manifest[endpoint] = {
                "standard_type": _standard_type_for_endpoint(endpoint),
                "row_count": int(len(normalized_df)),
                "split_counts": {
                    key: int(value)
                    for key, value in normalized_df["split"].value_counts().sort_index().items()
                },
                "activity_class_counts": {
                    key: int(value)
                    for key, value in normalized_df["activity_class"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "diqr_outlier_count": int(normalized_df["diqr_outlier"].sum()),
                "training_eligible_count": int(normalized_df["training_eligible"].sum()),
                "measurement_count": int(normalized_df["measurement_count"].sum()),
                "assay_modality_counts": {
                    key: int(value)
                    for key, value in normalized_df["assay_modality"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "assay_type_counts": {
                    key: int(value)
                    for key, value in normalized_df["assay_type"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "assay_organism_counts": {
                    key: int(value)
                    for key, value in normalized_df["assay_organism"]
                    .value_counts()
                    .sort_index()
                    .items()
                },
                "source": "ChEMBL",
            }

        target_manifest[target] = {
            "target_name": target_config.get_target_name(target),
            "row_count": int(sum(item["row_count"] for item in endpoint_manifest.values())),
            "endpoints": endpoint_manifest,
        }

    snapshot_df = pd.concat(frames, ignore_index=True)
    snapshot_df = snapshot_df.sort_values(
        ["target", "endpoint", "split", "molecule_chembl_id", "canonical_smiles"],
        kind="mergesort",
    ).reset_index(drop=True)
    snapshot_df = snapshot_df[DEFAULT_ENDPOINT_SNAPSHOT_COLUMNS]

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
        "schema": DEFAULT_ENDPOINT_SNAPSHOT_COLUMNS,
        "targets": target_manifest,
        "filters": {
            "standard_types": {
                endpoint: _standard_type_for_endpoint(endpoint) for endpoint in endpoints
            },
            "standard_relation": "=",
            "standard_units": "nM",
            "p_value_range": [0, 15],
            "max_rows_per_target_endpoint": max_rows_per_target_endpoint,
            "inactive_threshold_uM": inactive_threshold_uM,
            "inactive_threshold_nM": inactive_threshold_uM * 1000.0,
            "inactive_rule": "standard_value_nM >= inactive_threshold_uM * 1000",
            "outlier_rule": "dIQR on endpoint p_value within target/endpoint",
            "diq_multiplier": diq_multiplier,
            "aggregation_method": aggregation_method,
            "aggregation_unit": (
                "molecule_chembl_id + canonical_smiles + endpoint + assay context"
            ),
        },
        "split_policy": {
            "method": split_method,
            "random_seed": random_seed,
            "scaffold_test_fraction": scaffold_test_fraction,
            "external_fraction": external_fraction,
            "unit": "endpoint-specific Murcko scaffold",
        },
        "context_of_use": {
            "intended_use": "fixed endpoint benchmark for CPU pIC50/pKi model evaluation",
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


def _normalize_endpoint_frame(
    df: pd.DataFrame,
    *,
    target: str,
    endpoint: str,
    snapshot_id: str,
    random_seed: int,
    scaffold_test_fraction: float,
    external_fraction: float,
    split_method: str,
    diq_multiplier: float,
    inactive_threshold_uM: float,
    aggregation_method: str,
    max_rows: int | None,
) -> pd.DataFrame:
    standard_type = _standard_type_for_endpoint(endpoint)
    work_df = df.copy()
    if "standard_value_nM" not in work_df.columns and "standard_value" in work_df.columns:
        work_df = work_df.rename(columns={"standard_value": "standard_value_nM"})
    required = {"molecule_chembl_id", "canonical_smiles", "standard_value_nM"}
    missing = required.difference(work_df.columns)
    if missing:
        raise ValueError(f"Target {target} {endpoint} is missing columns: {sorted(missing)}")

    selected_columns = list(
        required.union(
            {
                "p_value",
                "endpoint",
                "standard_type",
                "document_chembl_id",
                "document_year",
                "molecule_pref_name",
                "parent_molecule_chembl_id",
                *ASSAY_CONTEXT_COLUMNS,
            }
        )
    )
    work_df = work_df[[column for column in selected_columns if column in work_df.columns]].copy()
    work_df = work_df.dropna(subset=["molecule_chembl_id", "canonical_smiles", "standard_value_nM"])
    work_df["standard_value_nM"] = pd.to_numeric(work_df["standard_value_nM"], errors="coerce")
    work_df = work_df.dropna(subset=["standard_value_nM"])
    work_df = work_df[work_df["standard_value_nM"] > 0]

    if "p_value" not in work_df.columns:
        work_df["p_value"] = -np.log10(work_df["standard_value_nM"].to_numpy(dtype=float) * 1e-9)
    else:
        work_df["p_value"] = pd.to_numeric(work_df["p_value"], errors="coerce")
    work_df = work_df.dropna(subset=["p_value"])
    work_df = work_df[(work_df["p_value"] >= 0) & (work_df["p_value"] <= 15)]

    work_df["canonical_smiles"] = work_df["canonical_smiles"].astype(str)
    work_df["molecule_chembl_id"] = work_df["molecule_chembl_id"].astype(str)
    work_df = _ensure_assay_context_columns(work_df, standard_type=standard_type)
    work_df = _aggregate_endpoint_replicates(work_df, method=aggregation_method)

    inactive_threshold_nM = inactive_threshold_uM * 1000.0
    work_df["activity_class"] = np.where(
        work_df["standard_value_nM"] >= inactive_threshold_nM,
        "inactive_ge_threshold",
        "measured_active_range",
    )
    work_df["diqr_outlier"] = _diqr_outlier_mask(work_df["p_value"], multiplier=diq_multiplier)
    work_df["training_eligible"] = ~work_df["diqr_outlier"]
    work_df["scaffold_smiles"] = work_df["canonical_smiles"].map(_murcko_scaffold_smiles)
    work_df = work_df.dropna(subset=["scaffold_smiles"])
    work_df = work_df.sort_values(
        ["molecule_chembl_id", "endpoint", "assay_modality", "assay_context_key"],
        kind="mergesort",
    )

    if max_rows is not None:
        if max_rows <= 0:
            raise ValueError("max_rows_per_target_endpoint must be positive when provided.")
        work_df = work_df.head(max_rows)

    if work_df.empty:
        raise ValueError(f"Target {target} {endpoint} has no valid rows after filtering.")

    work_df["snapshot_id"] = snapshot_id
    work_df["target"] = target
    work_df["target_name"] = target_config.get_target_name(target)
    work_df["endpoint"] = endpoint
    work_df["standard_type"] = standard_type
    if split_method == "sklearn_group_shuffle":
        work_df["split"] = _assign_sklearn_scaffold_splits(
            work_df["scaffold_smiles"].tolist(),
            random_seed=random_seed,
            scaffold_test_fraction=scaffold_test_fraction,
            external_fraction=external_fraction,
        )
    else:
        work_df["split"] = _assign_scaffold_splits(
            work_df["scaffold_smiles"].tolist(),
            target=f"{target}:{endpoint}",
            random_seed=random_seed,
            scaffold_test_fraction=scaffold_test_fraction,
            external_fraction=external_fraction,
        )
    work_df["source"] = "ChEMBL"
    return work_df


def _preprocess_endpoint_activity(df: pd.DataFrame, *, endpoint: str) -> pd.DataFrame:
    standard_type = _standard_type_for_endpoint(endpoint)
    if "standard_value_nM" not in df.columns and "standard_value" in df.columns:
        df = df.rename(columns={"standard_value": "standard_value_nM"})
    required = {"molecule_chembl_id", "canonical_smiles", "standard_value_nM"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required ChEMBL activity columns: {sorted(missing)}")

    optional_columns = [
        "document_chembl_id",
        "document_year",
        "molecule_pref_name",
        "parent_molecule_chembl_id",
        "pchembl_value",
        *ASSAY_CONTEXT_COLUMNS,
    ]
    selected = list(required) + [column for column in optional_columns if column in df.columns]
    work_df = df[selected].copy()
    work_df = _ensure_assay_context_columns(work_df, standard_type=standard_type)
    work_df = work_df.dropna(subset=["molecule_chembl_id", "canonical_smiles", "standard_value_nM"])
    work_df["standard_value_nM"] = pd.to_numeric(work_df["standard_value_nM"], errors="coerce")
    work_df = work_df.dropna(subset=["standard_value_nM"])
    work_df = work_df[work_df["standard_value_nM"] > 0]
    work_df["p_value"] = -np.log10(work_df["standard_value_nM"].to_numpy(dtype=float) * 1e-9)
    work_df = work_df[(work_df["p_value"] >= 0) & (work_df["p_value"] <= 15)]

    valid_rows = []
    for _, row in work_df.iterrows():
        smiles = str(row["canonical_smiles"])
        if Chem.MolFromSmiles(smiles) is None:
            continue
        payload = {
            "molecule_chembl_id": str(row["molecule_chembl_id"]),
            "canonical_smiles": smiles,
            "endpoint": endpoint,
            "standard_type": standard_type,
            "standard_value_nM": float(row["standard_value_nM"]),
            "p_value": float(row["p_value"]),
        }
        for column in optional_columns:
            if column in row.index:
                payload[column] = _clean_optional_text(row[column])
        valid_rows.append(payload)
    return pd.DataFrame(valid_rows)


def _attach_assay_details(df: pd.DataFrame) -> pd.DataFrame:
    """Attach assay organism/cell/tissue metadata in ChEMBL batches."""
    if "assay_chembl_id" not in df.columns or df.empty:
        return df

    assay_ids = sorted(
        {
            str(value)
            for value in df["assay_chembl_id"].dropna().tolist()
            if str(value).strip()
        }
    )
    if not assay_ids:
        return df

    detail_rows: list[dict[str, Any]] = []
    detail_fields = [
        "assay_chembl_id",
        "assay_cell_type",
        "assay_organism",
        "assay_tissue",
        "assay_type",
        "assay_type_description",
        "bao_format",
        "bao_label",
        "cell_chembl_id",
        "confidence_description",
        "confidence_score",
        "description",
        "tissue_chembl_id",
    ]
    for chunk in _chunks(assay_ids, 100):
        try:
            records = list(
                new_client.assay.filter(assay_chembl_id__in=chunk).only(detail_fields)
            )
        except Exception:
            records = []
            for assay_id in chunk:
                try:
                    records.extend(
                        list(new_client.assay.filter(assay_chembl_id=assay_id).only(detail_fields))
                    )
                except Exception:
                    continue
        detail_rows.extend(records)

    if not detail_rows:
        return df

    detail_df = pd.DataFrame(detail_rows).drop_duplicates(
        subset=["assay_chembl_id"],
        keep="first",
    )
    detail_df = detail_df.rename(
        columns={
            "description": "assay_description_detail",
            "bao_format": "bao_format_detail",
            "bao_label": "bao_label_detail",
            "assay_type": "assay_type_detail",
        }
    )
    merged = df.merge(detail_df, on="assay_chembl_id", how="left")
    for base, detail in [
        ("assay_description", "assay_description_detail"),
        ("bao_format", "bao_format_detail"),
        ("bao_label", "bao_label_detail"),
        ("assay_type", "assay_type_detail"),
    ]:
        if detail in merged.columns:
            if base not in merged.columns:
                merged[base] = merged[detail]
            else:
                merged[base] = merged[base].where(merged[base].notna(), merged[detail])
            merged = merged.drop(columns=[detail])
    return merged


def _ensure_assay_context_columns(df: pd.DataFrame, *, standard_type: str) -> pd.DataFrame:
    work_df = df.copy()
    if "assay_description" not in work_df.columns and "description" in work_df.columns:
        work_df["assay_description"] = work_df["description"]
    if "assay_organism" not in work_df.columns and "target_organism" in work_df.columns:
        work_df["assay_organism"] = work_df["target_organism"]

    for column in ASSAY_CONTEXT_COLUMNS:
        if column not in work_df.columns:
            work_df[column] = None
        work_df[column] = work_df[column].map(_clean_optional_text)

    fallback_descriptions = {
        "B": "Binding",
        "F": "Functional",
        "A": "ADME",
        "T": "Toxicity",
        "P": "Physicochemical",
        "U": "Unclassified",
    }
    work_df["assay_type_description"] = np.where(
        work_df["assay_type_description"].astype(str).str.len() > 0,
        work_df["assay_type_description"],
        work_df["assay_type"].map(fallback_descriptions).fillna("unknown"),
    )
    work_df["assay_modality"] = work_df.apply(
        lambda row: _infer_assay_modality(
            standard_type=standard_type,
            assay_type_description=str(row.get("assay_type_description", "")),
            assay_description=str(row.get("assay_description", "")),
            bao_label=str(row.get("bao_label", "")),
        ),
        axis=1,
    )
    work_df["assay_context_key"] = work_df.apply(_assay_context_key, axis=1)
    return work_df


def _aggregate_endpoint_replicates(df: pd.DataFrame, *, method: str) -> pd.DataFrame:
    if method == "none":
        work_df = df.copy()
        work_df["measurement_count"] = 1
        work_df["aggregation_method"] = "none"
        return work_df

    group_columns = [
        "molecule_chembl_id",
        "canonical_smiles",
        "endpoint",
        "standard_type",
        *ASSAY_CONTEXT_COLUMNS,
        "assay_context_key",
    ]
    rows: list[dict[str, Any]] = []
    for _, group in df.groupby(group_columns, dropna=False, sort=True):
        p_values = pd.to_numeric(group["p_value"], errors="coerce").dropna().to_numpy(dtype=float)
        if len(p_values) == 0:
            continue
        if method == "median":
            p_value = float(np.median(p_values))
        elif method == "robust_mean":
            p_value = _trimmed_mean(p_values)
        else:
            raise ValueError(f"Unsupported aggregation method: {method}")

        first = group.iloc[0]
        row = {column: first[column] for column in group_columns}
        row["p_value"] = p_value
        row["standard_value_nM"] = float(10 ** (9 - p_value))
        row["measurement_count"] = int(len(group))
        row["aggregation_method"] = method
        rows.append(row)
    return pd.DataFrame(rows)


def _trimmed_mean(values: np.ndarray) -> float:
    values = np.sort(values.astype(float))
    if len(values) < 5:
        return float(np.mean(values))
    trim = max(1, int(np.floor(len(values) * 0.2)))
    trimmed = values[trim:-trim]
    if len(trimmed) == 0:
        trimmed = values
    return float(np.mean(trimmed))


def _infer_assay_modality(
    *,
    standard_type: str,
    assay_type_description: str,
    assay_description: str,
    bao_label: str,
) -> str:
    text = " ".join([standard_type, assay_type_description, assay_description, bao_label]).lower()
    if "uptake" in text or "transport" in text or "reuptake" in text:
        return "uptake"
    if "binding" in text or "displacement" in text or "radioligand" in text:
        return "binding"
    if standard_type.upper() == "KI":
        return "binding"
    if "functional" in text or "inhibition" in text or "inhibitory" in text:
        return "functional"
    return "unknown"


def _assay_context_key(row: pd.Series) -> str:
    parts = [
        row.get("assay_modality"),
        row.get("assay_type"),
        row.get("assay_organism"),
        row.get("assay_cell_type"),
        row.get("assay_tissue"),
        row.get("bao_label"),
    ]
    return "|".join(_clean_optional_text(part) or "unknown" for part in parts)


def _clean_optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    text = str(value).strip()
    if text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _standard_type_for_endpoint(endpoint: str) -> str:
    try:
        return ENDPOINT_TO_STANDARD_TYPE[endpoint]
    except KeyError as exc:
        raise ValueError(f"Unsupported endpoint: {endpoint}") from exc


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


def _assign_sklearn_scaffold_splits(
    scaffold_values: list[str],
    *,
    random_seed: int,
    scaffold_test_fraction: float,
    external_fraction: float,
) -> list[str]:
    indices = np.arange(len(scaffold_values))
    groups = np.asarray(scaffold_values, dtype=object)
    split = np.asarray(["train"] * len(scaffold_values), dtype=object)

    train_indices, external_indices = _group_shuffle_holdout(
        indices,
        groups,
        fraction=external_fraction,
        random_seed=random_seed,
    )
    split[external_indices] = "external"

    train_indices, scaffold_test_indices = _group_shuffle_holdout(
        train_indices,
        groups[train_indices],
        fraction=scaffold_test_fraction,
        random_seed=random_seed + 1,
    )
    del train_indices
    split[scaffold_test_indices] = "scaffold_test"
    return [str(value) for value in split.tolist()]


def _group_shuffle_holdout(
    indices: np.ndarray,
    groups: np.ndarray,
    *,
    fraction: float,
    random_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(indices) < 3 or fraction == 0:
        return indices, np.asarray([], dtype=int)
    if len(set(groups.tolist())) < 3:
        return indices, np.asarray([], dtype=int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=random_seed)
    try:
        train_relative, holdout_relative = next(splitter.split(indices, groups=groups))
    except ValueError:
        return indices, np.asarray([], dtype=int)
    return indices[train_relative], indices[holdout_relative]


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


def _diqr_outlier_mask(values: pd.Series, *, multiplier: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.dropna().shape[0] < 4:
        return pd.Series(False, index=values.index)
    q1 = float(numeric.quantile(0.25))
    q3 = float(numeric.quantile(0.75))
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        return pd.Series(False, index=values.index)
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    return (numeric < lower) | (numeric > upper)
