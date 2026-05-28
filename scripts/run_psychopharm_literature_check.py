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
        "activity values with local pIC50 predictions."
    ),
    "decision_role": "research_triage_only",
    "not_for": [
        "clinical_decision",
        "regulatory_submission",
        "patient_care",
        "controlled-substance handling decisions",
    ],
}


class _Prediction(Protocol):
    smiles: str
    target: str
    pIC50_prediction: float
    uncertainty: float
    applicability_domain: dict[str, Any]
    model_version: str
    model_kind: str
    device: str


class _Predictor(Protocol):
    def predict(self, smiles: str, target: str) -> _Prediction:
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


def _prediction_payload(prediction: _Prediction) -> dict[str, Any]:
    return {
        "target": prediction.target,
        "pIC50": _round(float(prediction.pIC50_prediction)),
        "uncertainty": _round(float(prediction.uncertainty)),
        "model_version": prediction.model_version,
        "model_kind": prediction.model_kind,
        "device": prediction.device,
        "applicability_domain": prediction.applicability_domain,
    }


def _comparison_rows(reference_df: pd.DataFrame, predictor: _Predictor) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    grouped = reference_df.groupby(["compound_label", "model_target"], sort=True)
    for (compound_label, model_target), group in grouped:
        first = group.iloc[0]
        literature = _literature_summary(group)
        smiles = str(first["canonical_smiles"])
        prediction = predictor.predict(smiles, str(model_target))
        prediction_px = float(prediction.pIC50_prediction)
        mean_px = literature["mean_pX"]
        delta = prediction_px - float(mean_px) if mean_px is not None else None
        fold_error = 10 ** abs(delta) if delta is not None else None
        comparisons.append(
            {
                "compound_label": str(compound_label),
                "compound_proxy": str(first["compound_proxy"]),
                "compound_chembl_id": str(first["compound_chembl_id"]),
                "canonical_smiles": smiles,
                "literature_targets": sorted({str(value) for value in group["literature_target"]}),
                "model_target": str(model_target),
                "target_label": str(first["target_label"]),
                "literature": literature,
                "prediction": _prediction_payload(prediction),
                "prediction_minus_literature_mean": _round(delta),
                "fold_error_vs_literature_mean": _round(fold_error),
            }
        )
    return comparisons


def run_psychopharm_literature_check(
    *,
    reference_path: str | Path = "data/psychopharm_literature_reference.csv",
    model_path: str | Path = "models/chembl_category_cpu_pic50_model.json",
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
        from src.models.demo_cpu import CPUDemoPIC50Model

        predictor = CPUDemoPIC50Model.from_file(model_path)

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
    parser.add_argument("--model", default="models/chembl_category_cpu_pic50_model.json")
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
