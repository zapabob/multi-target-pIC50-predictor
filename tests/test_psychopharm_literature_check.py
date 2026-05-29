from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from scripts.run_psychopharm_literature_check import run_psychopharm_literature_check


@dataclass
class _FakePrediction:
    smiles: str
    target: str
    endpoint: str
    endpoint_prediction: float
    uncertainty: float
    applicability_domain: dict[str, object]
    model_version: str = "fake-v1"
    model_kind: str = "fake"
    device: str = "cpu"


class _FakePredictor:
    def predict(self, smiles: str, target: str, endpoint: str = "pIC50") -> _FakePrediction:
        predictions = {"pIC50": 7.0, "pKi": 8.0}
        return _FakePrediction(
            smiles=smiles,
            target=target,
            endpoint=endpoint,
            endpoint_prediction=predictions[endpoint],
            uncertainty=0.25,
            applicability_domain={"in_domain": True},
        )


def test_psychopharm_literature_check_groups_reference_values(tmp_path: Path):
    reference_path = tmp_path / "reference.csv"
    reference_path.write_text(
        "\n".join(
            [
                "compound_label,compound_proxy,canonical_smiles,compound_chembl_id,literature_target,model_target,target_label,standard_type,standard_relation,standard_value_nM,pchembl_value,document_chembl_id,year,doi,pubmed_id,assay_note,comparability",
                "LSD,Lysergide,CCN,CHEMBL263881,CHEMBL224,CHEMBL224,5HT2A,Ki,=,3.162,8.50,CHEMBL1145901,2003,10.1021/jm0341204,14613313,human receptor,direct target",
                "LSD,Lysergide,CCN,CHEMBL263881,CHEMBL224,CHEMBL224,5HT2A,EC50,=,12.9,7.89,CHEMBL5113537,2022,10.1021/acs.jmedchem.2c00702,36099411,human receptor,functional endpoint",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "report.json"

    report = run_psychopharm_literature_check(
        reference_path=reference_path,
        output_path=output_path,
        predictor=_FakePredictor(),
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == report
    assert report["context_of_use"]["decision_role"] == "research_triage_only"
    assert report["summary"]["comparison_count"] == 1
    comparison = report["comparisons"][0]
    assert comparison["compound_label"] == "LSD"
    assert comparison["model_target"] == "CHEMBL224"
    assert comparison["literature"]["n"] == 2
    assert comparison["literature"]["mean_pX"] == 8.195
    assert comparison["literature_by_endpoint"]["pKi"]["n"] == 1
    assert comparison["literature_by_endpoint"]["pKi"]["mean"] == 8.5
    assert comparison["literature_by_endpoint"]["pIC50"]["n"] == 0
    assert comparison["literature_by_endpoint"]["pEC50"]["mean"] == 7.89
    assert comparison["predictions"]["pIC50"]["value"] == 7.0
    assert comparison["predictions"]["pKi"]["value"] == 8.0
    assert comparison["endpoint_deltas"]["pKi"]["prediction_minus_literature_mean"] == -0.5
    assert comparison["endpoint_deltas"]["pKi"]["fold_error_vs_literature_mean"] > 3
    assert "mol_wt" in comparison["rdkit_features"]
    assert "CHEMBL5113537" in comparison["literature"]["document_chembl_ids"]
