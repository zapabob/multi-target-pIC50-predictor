"""CPU-only descriptor baseline for pharma MVP demonstrations."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import QED, Crippen, Descriptors, Lipinski
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "mol_wt",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rotatable_bonds",
    "aromatic_rings",
    "fraction_csp3",
    "qed",
]

CONTEXT_OF_USE = {
    "intended_use": "CPU-only early discovery demo for small-molecule pIC50 triage.",
    "decision_role": "research_triage_only",
    "not_for": [
        "clinical_decision",
        "regulatory_submission",
        "manufacturing_release",
        "patient_care",
    ],
    "endpoint": (
        "Target-specific pIC50 derived from IC50 nM values. The bundled dataset is a "
        "fixed demo fixture and must be replaced by a governed ChEMBL or sponsor "
        "snapshot before scientific or portfolio decisions."
    ),
}


@dataclass
class CPUDemoPrediction:
    """Serializable CPU baseline prediction output."""

    smiles: str
    target: str
    pIC50_prediction: float
    uncertainty: float
    applicability_domain: dict[str, Any]
    model_version: str
    model_kind: str
    device: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CPUDemoPIC50Model:
    """Small descriptor Ridge model that runs on CPU-only environments."""

    def __init__(self, payload: dict[str, Any]):
        self.payload = payload
        self.model_version = str(payload["model_version"])
        self.model_kind = str(payload["model_kind"])
        self.device = str(payload.get("device", "cpu"))
        self.feature_names = list(payload["feature_names"])
        self.targets = payload["targets"]
        self.context_of_use = payload["context_of_use"]

    @classmethod
    def from_file(cls, model_path: str | Path) -> CPUDemoPIC50Model:
        path = Path(model_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(payload)

    def predict(self, smiles: str, target: str = "CHEMBL238") -> CPUDemoPrediction:
        if target not in self.targets:
            raise ValueError(f"Unsupported target: {target}")

        feature_map = calculate_descriptor_features(smiles)
        features = np.array([feature_map[name] for name in self.feature_names], dtype=float)
        target_payload = self.targets[target]

        mean = np.array(target_payload["scaler_mean"], dtype=float)
        scale = np.array(target_payload["scaler_scale"], dtype=float)
        coefficients = np.array(target_payload["coefficients"], dtype=float)
        scaled_features = (features - mean) / scale
        prediction = float(target_payload["intercept"] + np.dot(scaled_features, coefficients))

        domain = _domain_check(feature_map, target_payload["applicability_domain"])
        base_uncertainty = float(target_payload["residual_rmse"])
        uncertainty = max(0.15, base_uncertainty * (1.0 + domain["distance"]))

        return CPUDemoPrediction(
            smiles=smiles,
            target=target,
            pIC50_prediction=round(prediction, 3),
            uncertainty=round(uncertainty, 3),
            applicability_domain=domain,
            model_version=self.model_version,
            model_kind=self.model_kind,
            device=self.device,
        )


class CPUDemoPredictorAdapter:
    """Adapter for CompoundAssessmentPipeline's legacy predictor interface."""

    def __init__(self, model: CPUDemoPIC50Model, target: str):
        self.model = model
        self.target = target
        self.last_result: CPUDemoPrediction | None = None

    def predict(self, smiles: str) -> tuple[float, dict[str, Any]]:
        self.last_result = self.model.predict(smiles, self.target)
        return self.last_result.pIC50_prediction, {
            "std": self.last_result.uncertainty,
            "applicability_domain": self.last_result.applicability_domain,
            "model_version": self.last_result.model_version,
            "model_kind": self.last_result.model_kind,
        }


def calculate_descriptor_features(smiles: str) -> dict[str, float]:
    """Calculate compact RDKit descriptors for the CPU baseline."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    return {
        "mol_wt": float(Descriptors.MolWt(mol)),
        "logp": float(Crippen.MolLogP(mol)),
        "tpsa": float(Descriptors.TPSA(mol)),
        "hbd": float(Lipinski.NumHDonors(mol)),
        "hba": float(Lipinski.NumHAcceptors(mol)),
        "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
        "aromatic_rings": float(Lipinski.NumAromaticRings(mol)),
        "fraction_csp3": float(Descriptors.FractionCSP3(mol)),
        "qed": float(QED.qed(mol)),
    }


def build_demo_cpu_artifacts(
    dataset_path: str | Path,
    model_path: str | Path,
    report_path: str | Path,
    *,
    alpha: float = 1.0,
) -> tuple[Path, Path]:
    """Build the checked-in CPU demo model and benchmark report."""
    dataset_path = Path(dataset_path)
    model_path = Path(model_path)
    report_path = Path(report_path)

    df = pd.read_csv(dataset_path)
    if "smiles" not in df.columns and "canonical_smiles" in df.columns:
        df = df.rename(columns={"canonical_smiles": "smiles"})
    required_columns = {"target", "target_name", "split", "smiles", "pIC50", "source"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required benchmark columns: {sorted(missing)}")

    feature_rows = []
    for smiles in df["smiles"]:
        feature_rows.append(calculate_descriptor_features(str(smiles)))
    feature_df = pd.DataFrame(feature_rows, columns=FEATURE_NAMES)
    work_df = pd.concat([df.reset_index(drop=True), feature_df], axis=1)

    model_payload: dict[str, Any] = {
        "model_version": "demo-cpu-ridge-v1",
        "model_kind": "cpu_descriptor_ridge",
        "device": "cpu",
        "feature_names": FEATURE_NAMES,
        "context_of_use": CONTEXT_OF_USE,
        "training": {
            "algorithm": "sklearn.linear_model.Ridge",
            "alpha": alpha,
            "dataset_path": str(dataset_path.as_posix()),
            "dataset_source": "checked-in demo fixture",
            "random_seed": None,
        },
        "targets": {},
    }
    report_payload: dict[str, Any] = {
        "model_version": model_payload["model_version"],
        "model_kind": model_payload["model_kind"],
        "device": "cpu",
        "context_of_use": CONTEXT_OF_USE,
        "benchmark_dataset": {
            "path": str(dataset_path.as_posix()),
            "source": _dataset_source_label(work_df),
            "rows": int(len(work_df)),
            "splits": sorted(work_df["split"].unique().tolist()),
        },
        "targets": {},
    }

    for target, target_df in work_df.groupby("target", sort=True):
        train_df = target_df[target_df["split"] == "train"].copy()
        if train_df.empty:
            raise ValueError(f"Target {target} has no train split")

        scaler = StandardScaler()
        x_train = scaler.fit_transform(train_df[FEATURE_NAMES].to_numpy(dtype=float))
        y_train = train_df["pIC50"].to_numpy(dtype=float)
        regressor = Ridge(alpha=alpha)
        regressor.fit(x_train, y_train)

        domain = _build_domain(train_df)
        split_metrics = {}
        for split_name, split_df in target_df.groupby("split", sort=True):
            split_metrics[split_name] = _evaluate_split(regressor, scaler, split_df)

        residual_rmse = split_metrics.get("scaffold_test", split_metrics["train"])["rmse"]
        if residual_rmse is None or math.isnan(residual_rmse):
            residual_rmse = split_metrics["train"]["rmse"]
        residual_rmse = max(float(residual_rmse or 0.0), 0.25)

        target_name = str(train_df["target_name"].iloc[0])
        model_payload["targets"][target] = {
            "target_name": target_name,
            "coefficients": [float(value) for value in regressor.coef_],
            "intercept": float(regressor.intercept_),
            "scaler_mean": [float(value) for value in scaler.mean_],
            "scaler_scale": [
                float(value) if float(value) != 0.0 else 1.0 for value in scaler.scale_
            ],
            "residual_rmse": round(residual_rmse, 4),
            "applicability_domain": domain,
            "metrics": split_metrics,
        }
        report_payload["targets"][target] = {
            "target_name": target_name,
            "n_train": int(len(train_df)),
            "metrics": split_metrics,
            "applicability_domain": domain,
        }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model_payload, indent=2), encoding="utf-8")
    report_path.write_text(json.dumps(report_payload, indent=2), encoding="utf-8")
    return model_path, report_path


def _build_domain(train_df: pd.DataFrame) -> dict[str, Any]:
    ranges = {}
    for name in FEATURE_NAMES:
        values = train_df[name].to_numpy(dtype=float)
        ranges[name] = {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return {
        "method": "descriptor_min_max",
        "training_split": "train",
        "features": ranges,
        "warning": (
            "Predictions outside the descriptor min/max envelope should be treated as "
            "low-confidence research triage only."
        ),
    }


def _dataset_source_label(df: pd.DataFrame) -> str:
    sources = sorted(str(value) for value in df["source"].dropna().unique())
    if len(sources) == 1:
        return sources[0]
    return "mixed"


def _domain_check(
    feature_map: dict[str, float],
    domain_payload: dict[str, Any],
) -> dict[str, Any]:
    out_of_domain = []
    total_distance = 0.0

    for name, bounds in domain_payload["features"].items():
        value = float(feature_map[name])
        lower = float(bounds["min"])
        upper = float(bounds["max"])
        width = max(upper - lower, 1.0)
        if value < lower:
            out_of_domain.append(name)
            total_distance += (lower - value) / width
        elif value > upper:
            out_of_domain.append(name)
            total_distance += (value - upper) / width

    distance = total_distance / max(len(domain_payload["features"]), 1)
    return {
        "method": domain_payload["method"],
        "in_domain": not out_of_domain,
        "out_of_domain_features": out_of_domain,
        "distance": round(float(distance), 4),
        "warning": domain_payload["warning"] if out_of_domain else None,
    }


def _evaluate_split(
    regressor: Ridge,
    scaler: StandardScaler,
    split_df: pd.DataFrame,
) -> dict[str, float | int | None]:
    x_values = scaler.transform(split_df[FEATURE_NAMES].to_numpy(dtype=float))
    y_true = split_df["pIC50"].to_numpy(dtype=float)
    y_pred = regressor.predict(x_values)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred)) if len(split_df) > 1 else None
    return {
        "r2": round(r2, 4) if r2 is not None else None,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "n": int(len(split_df)),
    }
