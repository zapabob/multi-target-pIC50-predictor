"""CPU-only pharma MVP demo path tests."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.models.demo_cpu import (
    CPUDemoEndpointModel,
    CPUDemoPIC50Model,
    build_demo_cpu_artifacts,
    build_demo_endpoint_cpu_artifacts,
)

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"
DATASET_PATH = Path("data/demo_pic50_benchmark.csv")
ENDPOINT_SNAPSHOT_ROWS = [
    "snapshot_id,target,target_name,endpoint,standard_type,molecule_chembl_id,canonical_smiles,p_value,split,scaffold_smiles,source",
    "s1,CHEMBL238,DAT,pIC50,IC50,M1,CCO,4.1,train,CCO,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M2,CCCO,4.4,train,CCCO,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M3,CCCCO,4.7,train,CCCCO,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M4,c1ccccc1,4.0,train,c1ccccc1,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M5,CCN(CC)CC,5.0,scaffold_test,CCN,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M6,CC(=O)O,3.8,external,CC(=O)O,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K1,CCO,5.1,train,CCO,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K2,CCCO,5.4,train,CCCO,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K3,CCCCO,5.7,train,CCCCO,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K4,c1ccccc1,5.0,train,c1ccccc1,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K5,CCN(CC)CC,6.0,scaffold_test,CCN,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K6,CC(=O)O,4.8,external,CC(=O)O,ChEMBL",
]


def _write_endpoint_snapshot(path: Path) -> None:
    path.write_text("\n".join(ENDPOINT_SNAPSHOT_ROWS), encoding="utf-8")


def test_build_demo_artifacts_from_fixed_benchmark(tmp_path: Path):
    model_path = tmp_path / "demo_cpu_pic50_model.json"
    report_path = tmp_path / "demo_cpu_benchmark.json"

    build_demo_cpu_artifacts(DATASET_PATH, model_path, report_path)

    model_payload = json.loads(model_path.read_text(encoding="utf-8"))
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert model_payload["model_kind"] == "cpu_descriptor_ridge"
    assert model_payload["device"] == "cpu"
    assert report_payload["context_of_use"]["decision_role"] == "research_triage_only"
    assert {"CHEMBL238", "CHEMBL224"}.issubset(report_payload["targets"])

    for target_metrics in report_payload["targets"].values():
        assert {"train", "scaffold_test", "external"}.issubset(target_metrics["metrics"])
        for split_metrics in target_metrics["metrics"].values():
            assert {"r2", "rmse", "mae", "n"}.issubset(split_metrics)


def test_build_demo_artifacts_accepts_chembl_snapshot_schema(tmp_path: Path):
    snapshot_path = tmp_path / "snapshot.csv"
    snapshot_path.write_text(
        "\n".join(
            [
                "snapshot_id,target,target_name,molecule_chembl_id,canonical_smiles,pIC50,split,scaffold_smiles,source",
                "s1,CHEMBL238,DAT,M1,CCO,4.1,train,CCO,ChEMBL",
                "s1,CHEMBL238,DAT,M2,CCCO,4.4,train,CCCO,ChEMBL",
                "s1,CHEMBL238,DAT,M3,CCCCO,4.7,train,CCCCO,ChEMBL",
                "s1,CHEMBL238,DAT,M4,c1ccccc1,4.0,train,c1ccccc1,ChEMBL",
                "s1,CHEMBL238,DAT,M5,CCN(CC)CC,5.0,scaffold_test,CCN,ChEMBL",
                "s1,CHEMBL238,DAT,M6,CC(=O)O,3.8,external,CC(=O)O,ChEMBL",
            ]
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "snapshot_model.json"
    report_path = tmp_path / "snapshot_report.json"

    build_demo_cpu_artifacts(snapshot_path, model_path, report_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["targets"]["CHEMBL238"]["n_train"] == 4
    assert report_payload["benchmark_dataset"]["source"] == "ChEMBL"


def test_build_endpoint_artifacts_predicts_pic50_and_pki(tmp_path: Path):
    dataset_path = tmp_path / "endpoint_snapshot.csv"
    _write_endpoint_snapshot(dataset_path)
    model_path = tmp_path / "endpoint_model.json"
    report_path = tmp_path / "endpoint_report.json"

    build_demo_endpoint_cpu_artifacts(dataset_path, model_path, report_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert {"pIC50", "pKi"}.issubset(report_payload["endpoints"])
    assert report_payload["endpoints"]["pKi"]["targets"]["CHEMBL238"]["n_train"] == 4

    model = CPUDemoEndpointModel.from_file(model_path)
    pic50 = model.predict(ASPIRIN, target="CHEMBL238", endpoint="pIC50")
    pki = model.predict(ASPIRIN, target="CHEMBL238", endpoint="pKi")

    assert pic50.endpoint == "pIC50"
    assert pki.endpoint == "pKi"
    assert pic50.endpoint_prediction != pki.endpoint_prediction
    assert pki.applicability_domain["method"] == "descriptor_min_max"


def test_build_endpoint_artifacts_honors_training_eligible_flag(tmp_path: Path):
    dataset_path = tmp_path / "endpoint_snapshot.csv"
    dataset_path.write_text(
        "\n".join(
            [
                "snapshot_id,target,target_name,endpoint,standard_type,molecule_chembl_id,canonical_smiles,p_value,training_eligible,split,scaffold_smiles,source",
                "s1,CHEMBL238,DAT,pIC50,IC50,M1,CCO,4.1,true,train,CCO,ChEMBL",
                "s1,CHEMBL238,DAT,pIC50,IC50,M2,CCCO,4.4,true,train,CCCO,ChEMBL",
                "s1,CHEMBL238,DAT,pIC50,IC50,M3,CCCCO,4.7,true,train,CCCCO,ChEMBL",
                "s1,CHEMBL238,DAT,pIC50,IC50,M4,c1ccccc1,4.0,true,train,c1ccccc1,ChEMBL",
                "s1,CHEMBL238,DAT,pIC50,IC50,M5,CCN(CC)CC,5.0,true,scaffold_test,CCN,ChEMBL",
                "s1,CHEMBL238,DAT,pIC50,IC50,M6,CC(=O)O,3.8,true,external,CC(=O)O,ChEMBL",
                "s1,CHEMBL238,DAT,pIC50,IC50,M7,CCCCCC,1.0,false,train,CCCCCC,ChEMBL",
            ]
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "endpoint_model.json"
    report_path = tmp_path / "endpoint_report.json"

    build_demo_endpoint_cpu_artifacts(dataset_path, model_path, report_path)

    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["benchmark_dataset"]["rows"] == 6
    assert report_payload["endpoints"]["pIC50"]["targets"]["CHEMBL238"]["n_train"] == 4


def test_cpu_demo_model_predicts_with_uncertainty_and_domain(tmp_path: Path):
    model_path = tmp_path / "demo_cpu_pic50_model.json"
    report_path = tmp_path / "demo_cpu_benchmark.json"
    build_demo_cpu_artifacts(DATASET_PATH, model_path, report_path)

    model = CPUDemoPIC50Model.from_file(model_path)
    result = model.predict(ASPIRIN, target="CHEMBL238")

    assert result.target == "CHEMBL238"
    assert result.pIC50_prediction is not None
    assert result.uncertainty is not None
    assert result.uncertainty > 0
    assert result.applicability_domain["method"] == "descriptor_min_max"
    assert isinstance(result.applicability_domain["in_domain"], bool)


def test_cpu_demo_model_rejects_invalid_smiles(tmp_path: Path):
    model_path = tmp_path / "demo_cpu_pic50_model.json"
    report_path = tmp_path / "demo_cpu_benchmark.json"
    build_demo_cpu_artifacts(DATASET_PATH, model_path, report_path)

    model = CPUDemoPIC50Model.from_file(model_path)

    with pytest.raises(ValueError, match="Invalid SMILES"):
        model.predict("not-a-smiles", target="CHEMBL238")


def test_fastapi_predict_and_assess_use_cpu_demo_model(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    model_path = tmp_path / "demo_cpu_pic50_model.json"
    report_path = tmp_path / "demo_cpu_benchmark.json"
    build_demo_cpu_artifacts(DATASET_PATH, model_path, report_path)

    from src.api.app import create_app

    client = TestClient(create_app(model_path=model_path))

    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["model"]["device"] == "cpu"

    prediction = client.post(
        "/predict",
        json={"smiles": ASPIRIN, "target": "CHEMBL238"},
    )
    assert prediction.status_code == 200
    prediction_payload = prediction.json()
    assert prediction_payload["pIC50_prediction"] is not None
    assert prediction_payload["uncertainty"] is not None
    assert "applicability_domain" in prediction_payload

    assessment = client.post(
        "/assess",
        json={
            "smiles": ASPIRIN,
            "target": "CHEMBL238",
            "include_3d": False,
            "include_reactions": False,
            "include_image": False,
        },
    )
    assert assessment.status_code == 200
    assessment_payload = assessment.json()
    assert assessment_payload["pIC50_prediction"] is not None
    assert assessment_payload["admet"]["success"]


def test_fastapi_predicts_endpoint_model_pic50_and_pki(tmp_path: Path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    dataset_path = tmp_path / "endpoint_snapshot.csv"
    model_path = tmp_path / "endpoint_model.json"
    report_path = tmp_path / "endpoint_report.json"
    _write_endpoint_snapshot(dataset_path)
    build_demo_endpoint_cpu_artifacts(dataset_path, model_path, report_path)

    from src.api.app import create_app

    client = TestClient(create_app(model_path=model_path))

    health = client.get("/health")
    assert health.status_code == 200
    assert set(health.json()["model"]["endpoints"]) == {"pIC50", "pKi"}

    prediction = client.post(
        "/predict",
        json={"smiles": ASPIRIN, "target": "CHEMBL238", "endpoints": ["pIC50", "pKi"]},
    )
    assert prediction.status_code == 200
    prediction_payload = prediction.json()
    assert set(prediction_payload["predictions"]) == {"pIC50", "pKi"}
    assert prediction_payload["predictions"]["pIC50"]["pIC50_prediction"] is not None
    assert prediction_payload["predictions"]["pKi"]["pKi_prediction"] is not None

    assessment = client.post(
        "/assess",
        json={
            "smiles": ASPIRIN,
            "target": "CHEMBL238",
            "include_3d": False,
            "include_reactions": False,
            "include_image": False,
        },
    )
    assert assessment.status_code == 200
    assessment_payload = assessment.json()
    assert assessment_payload["model"]["endpoint"] == "pIC50"
    assert assessment_payload["endpoint_prediction"] is not None


def test_cli_predict_accepts_cpu_demo_json_model(tmp_path: Path):
    model_path = tmp_path / "demo_cpu_pic50_model.json"
    report_path = tmp_path / "demo_cpu_benchmark.json"
    build_demo_cpu_artifacts(DATASET_PATH, model_path, report_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "cli.py",
            "predict",
            "--model",
            str(model_path),
            "--target",
            "CHEMBL238",
            "--smiles",
            ASPIRIN,
            "--uncertainty",
        ],
        cwd=Path.cwd(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=60,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Predicted_pIC50" in completed.stdout
    assert "ApplicabilityDomain" in completed.stdout
    assert ASPIRIN in completed.stdout
