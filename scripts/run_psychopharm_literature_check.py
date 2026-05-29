"""Compare psychopharmacology reference values with local pIC50 predictions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


CONTEXT_OF_USE = {
    "intended_use": (
        "Research-only psychopharmacology sanity check comparing curated ChEMBL "
        "activity values with local endpoint-specific pIC50 and pKi predictions."
    ),
    "decision_role": "research_triage_only",
    "not_for": [
        "clinical_decision",
        "regulatory_submission",
        "patient_care",
        "controlled-substance handling decisions",
    ],
}

STANDARD_TYPE_TO_ENDPOINT = {
    "IC50": "pIC50",
    "KI": "pKi",
    "EC50": "pEC50",
}

PREDICTION_ENDPOINTS = ("pIC50", "pKi")


class _Prediction(Protocol):
    smiles: str
    target: str
    endpoint: str
    endpoint_prediction: float
    uncertainty: float
    applicability_domain: dict[str, Any]
    model_version: str
    model_kind: str
    device: str


class _Predictor(Protocol):
    def predict(self, smiles: str, target: str, endpoint: str = "pIC50") -> _Prediction:
        ...


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def _reference_px(row: pd.Series) -> float | None:
    pchembl_value = row.get("pchembl_value")
    if pd.notna(pchembl_value) and str(pchembl_value).strip():
        return float(pchembl_value)
    standard_value = row.get("standard_value_nM")
    if pd.isna(standard_value) or float(standard_value) <= 0:
        return None
    return -math.log10(float(standard_value) * 1e-9)


def _literature_summary(group: pd.DataFrame) -> dict[str, Any]:
    reference_values = [_reference_px(row) for _, row in group.iterrows()]
    values = np.asarray([value for value in reference_values if value is not None], dtype=float)
    if len(values) == 0:
        mean = median = sd = sem = None
    else:
        mean = float(np.mean(values))
        median = float(np.median(values))
        sd = float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0
        sem = float(sd / math.sqrt(len(values))) if len(values) >= 2 else 0.0
    endpoints = sorted({str(value) for value in group["standard_type"].dropna()})
    return {
        "n": int(len(values)),
        "mean_pX": _round(mean),
        "median_pX": _round(median),
        "sd_pX": _round(sd),
        "sem_pX": _round(sem),
        "endpoint_types": endpoints,
        "standard_values_nM": [
            _round(float(value)) for value in group["standard_value_nM"].dropna().tolist()
        ],
        "pX_values": [_round(value) for value in values.tolist()],
        "document_chembl_ids": sorted({str(value) for value in group["document_chembl_id"].dropna()}),
        "years": sorted({int(value) for value in group["year"].dropna()}),
        "dois": sorted({str(value) for value in group["doi"].dropna() if str(value).strip()}),
        "pubmed_ids": sorted({str(value) for value in group["pubmed_id"].dropna() if str(value).strip()}),
        "comparability_notes": sorted({str(value) for value in group["comparability"].dropna()}),
    }


def _endpoint_for_standard_type(standard_type: str) -> str:
    return STANDARD_TYPE_TO_ENDPOINT.get(str(standard_type).upper(), f"p{standard_type}")


def _endpoint_literature_summary(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "sd": None,
            "sem": None,
            "standard_values_nM": [],
            "values": [],
            "document_chembl_ids": [],
            "years": [],
            "dois": [],
            "pubmed_ids": [],
            "comparability_notes": [],
        }

    reference_values = [_reference_px(row) for _, row in group.iterrows()]
    values = np.asarray([value for value in reference_values if value is not None], dtype=float)
    if len(values) == 0:
        mean = median = sd = sem = None
    else:
        mean = float(np.mean(values))
        median = float(np.median(values))
        sd = float(np.std(values, ddof=1)) if len(values) >= 2 else 0.0
        sem = float(sd / math.sqrt(len(values))) if len(values) >= 2 else 0.0
    return {
        "n": int(len(values)),
        "mean": _round(mean),
        "median": _round(median),
        "sd": _round(sd),
        "sem": _round(sem),
        "standard_values_nM": [
            _round(float(value)) for value in group["standard_value_nM"].dropna().tolist()
        ],
        "values": [_round(value) for value in values.tolist()],
        "document_chembl_ids": sorted({str(value) for value in group["document_chembl_id"].dropna()}),
        "years": sorted({int(value) for value in group["year"].dropna()}),
        "dois": sorted({str(value) for value in group["doi"].dropna() if str(value).strip()}),
        "pubmed_ids": sorted({str(value) for value in group["pubmed_id"].dropna() if str(value).strip()}),
        "comparability_notes": sorted({str(value) for value in group["comparability"].dropna()}),
    }


def _literature_by_endpoint(group: pd.DataFrame) -> dict[str, Any]:
    work_df = group.copy()
    work_df["_endpoint"] = work_df["standard_type"].map(_endpoint_for_standard_type)
    summaries = {
        endpoint: _endpoint_literature_summary(endpoint_df)
        for endpoint, endpoint_df in work_df.groupby("_endpoint", sort=True)
    }
    for endpoint in PREDICTION_ENDPOINTS:
        summaries.setdefault(endpoint, _endpoint_literature_summary(work_df.iloc[0:0]))
    return summaries


def _prediction_value(prediction: _Prediction, endpoint: str) -> float:
    if hasattr(prediction, "endpoint_prediction"):
        return float(prediction.endpoint_prediction)
    if endpoint == "pIC50" and hasattr(prediction, "pIC50_prediction"):
        return float(prediction.pIC50_prediction)
    raise AttributeError(f"Prediction does not expose {endpoint}")


def _prediction_payload(prediction: _Prediction, endpoint: str) -> dict[str, Any]:
    prediction_value = _prediction_value(prediction, endpoint)
    return {
        "target": prediction.target,
        "endpoint": endpoint,
        "value": _round(prediction_value),
        "uncertainty": _round(float(prediction.uncertainty)),
        "model_version": prediction.model_version,
        "model_kind": prediction.model_kind,
        "device": prediction.device,
        "applicability_domain": prediction.applicability_domain,
    }


def _predict_endpoint(
    predictor: _Predictor,
    smiles: str,
    model_target: str,
    endpoint: str,
) -> dict[str, Any]:
    try:
        prediction = predictor.predict(smiles, str(model_target), endpoint=endpoint)
    except TypeError:
        if endpoint != "pIC50":
            return {"endpoint": endpoint, "error": "predictor does not support endpoint argument"}
        prediction = predictor.predict(smiles, str(model_target))
    except ValueError as exc:
        return {"endpoint": endpoint, "error": str(exc)}
    return _prediction_payload(prediction, endpoint)


def _descriptor_payload(smiles: str) -> dict[str, float]:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from src.models.demo_cpu import calculate_descriptor_features

    return {
        name: _round(value, digits=6)
        for name, value in calculate_descriptor_features(smiles).items()
    }


def _comparison_rows(reference_df: pd.DataFrame, predictor: _Predictor) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    grouped = reference_df.groupby(["compound_label", "model_target"], sort=True)
    for (compound_label, model_target), group in grouped:
        first = group.iloc[0]
        literature = _literature_summary(group)
        endpoint_literature = _literature_by_endpoint(group)
        smiles = str(first["canonical_smiles"])
        predictions = {
            endpoint: _predict_endpoint(predictor, smiles, str(model_target), endpoint)
            for endpoint in PREDICTION_ENDPOINTS
        }
        prediction_px = predictions["pIC50"].get("value")
        mean_px = literature["mean_pX"]
        delta = float(prediction_px) - float(mean_px) if prediction_px is not None and mean_px is not None else None
        fold_error = 10 ** abs(delta) if delta is not None else None
        endpoint_deltas = {}
        for endpoint, prediction_payload in predictions.items():
            endpoint_mean = endpoint_literature.get(endpoint, {}).get("mean")
            prediction_value = prediction_payload.get("value")
            endpoint_delta = (
                float(prediction_value) - float(endpoint_mean)
                if prediction_value is not None and endpoint_mean is not None
                else None
            )
            endpoint_deltas[endpoint] = {
                "prediction_minus_literature_mean": _round(endpoint_delta),
                "fold_error_vs_literature_mean": _round(
                    10 ** abs(endpoint_delta) if endpoint_delta is not None else None
                ),
            }
        comparisons.append(
            {
                "compound_label": str(compound_label),
                "compound_proxy": str(first["compound_proxy"]),
                "compound_chembl_id": str(first["compound_chembl_id"]),
                "canonical_smiles": smiles,
                "rdkit_features": _descriptor_payload(smiles),
                "literature_targets": sorted({str(value) for value in group["literature_target"]}),
                "model_target": str(model_target),
                "target_label": str(first["target_label"]),
                "literature": literature,
                "literature_by_endpoint": endpoint_literature,
                "predictions": predictions,
                "prediction": predictions["pIC50"],
                "endpoint_deltas": endpoint_deltas,
                "prediction_minus_literature_mean": _round(delta),
                "fold_error_vs_literature_mean": _round(fold_error),
            }
        )
    return comparisons


def run_psychopharm_literature_check(
    *,
    reference_path: str | Path = "data/psychopharm_literature_reference.csv",
    model_path: str | Path = "models/chembl_endpoint_cpu_model.json",
    output_path: str | Path = "artifacts/psychopharm_literature_prediction_check.json",
    predictor: _Predictor | None = None,
) -> dict[str, Any]:
    """Write a JSON comparison between curated reference values and predictions."""

    reference_path = Path(reference_path)
    output_path = Path(output_path)
    reference_df = pd.read_csv(reference_path)
    required = {
        "compound_label",
        "compound_proxy",
        "canonical_smiles",
        "compound_chembl_id",
        "literature_target",
        "model_target",
        "target_label",
        "standard_type",
        "standard_value_nM",
        "document_chembl_id",
        "year",
        "comparability",
    }
    missing = required.difference(reference_df.columns)
    if missing:
        raise ValueError(f"Missing required reference columns: {sorted(missing)}")

    if predictor is None:
        if str(REPO_ROOT) not in sys.path:
            sys.path.insert(0, str(REPO_ROOT))
        from src.models.demo_cpu import CPUDemoEndpointModel, CPUDemoPIC50Model

        model_payload = json.loads(Path(model_path).read_text(encoding="utf-8"))
        if "endpoints" in model_payload:
            predictor = CPUDemoEndpointModel(model_payload)
        else:
            predictor = CPUDemoPIC50Model(model_payload)

    comparisons = _comparison_rows(reference_df, predictor)
    deltas = [
        abs(float(row["prediction_minus_literature_mean"]))
        for row in comparisons
        if row["prediction_minus_literature_mean"] is not None
    ]
    report = {
        "context_of_use": CONTEXT_OF_USE,
        "reference_path": reference_path.as_posix(),
        "model_path": Path(model_path).as_posix(),
        "summary": {
            "compound_count": int(reference_df["compound_label"].nunique()),
            "comparison_count": len(comparisons),
            "reference_row_count": int(len(reference_df)),
            "literature_endpoints": sorted(
                {_endpoint_for_standard_type(value) for value in reference_df["standard_type"]}
            ),
            "prediction_endpoints": list(PREDICTION_ENDPOINTS),
            "mean_abs_delta_pX": _round(float(np.mean(deltas)) if deltas else None),
        },
        "comparisons": comparisons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare curated psychopharmacology activity values with local predictions."
    )
    parser.add_argument("--reference", default="data/psychopharm_literature_reference.csv")
    parser.add_argument("--model", default="models/chembl_endpoint_cpu_model.json")
    parser.add_argument("--output", default="artifacts/psychopharm_literature_prediction_check.json")
    args = parser.parse_args()
    report = run_psychopharm_literature_check(
        reference_path=args.reference,
        model_path=args.model,
        output_path=args.output,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
