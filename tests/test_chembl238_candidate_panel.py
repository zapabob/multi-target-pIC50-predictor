from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from scripts.run_chembl238_candidate_panel import (
    run_chembl238_candidate_panel,
    run_chembl238_qsar_comparison,
)
from src.models.demo_cpu import build_demo_endpoint_cpu_artifacts


SNAPSHOT_ROWS = [
    "snapshot_id,target,target_name,endpoint,standard_type,molecule_chembl_id,canonical_smiles,p_value,standard_value_nM,training_eligible,diqr_outlier,activity_class,split,scaffold_smiles,source",
    "s1,CHEMBL238,DAT,pIC50,IC50,M1,CCO,4.1,79432.8,true,false,measured_active_range,train,CCO,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M2,CCCO,4.4,39810.7,true,false,measured_active_range,train,CCCO,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M3,CCCCO,4.7,19952.6,true,false,measured_active_range,train,CCCCO,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M4,c1ccccc1,4.0,100000.0,true,false,measured_active_range,train,c1ccccc1,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M5,CCN(CC)CC,5.0,10000.0,true,false,measured_active_range,scaffold_test,CCN,ChEMBL",
    "s1,CHEMBL238,DAT,pIC50,IC50,M6,CC(=O)O,3.8,158489.3,true,false,measured_active_range,external,CC(=O)O,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K1,CCO,5.1,7943.3,true,false,measured_active_range,train,CCO,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K2,CCCO,5.4,3981.1,true,false,measured_active_range,train,CCCO,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K3,CCCCO,5.7,1995.3,true,false,measured_active_range,train,CCCCO,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K4,c1ccccc1,5.0,10000.0,true,false,measured_active_range,train,c1ccccc1,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K5,CCN(CC)CC,6.0,1000.0,true,false,measured_active_range,scaffold_test,CCN,ChEMBL",
    "s1,CHEMBL238,DAT,pKi,Ki,K6,CC(=O)O,4.8,15848.9,true,false,measured_active_range,external,CC(=O)O,ChEMBL",
]


REFERENCE_ROWS = [
    "compound_label,compound_proxy,canonical_smiles,compound_chembl_id,literature_target,model_target,target_label,standard_type,standard_relation,standard_value_nM,pchembl_value,document_chembl_id,year,doi,pubmed_id,assay_note,comparability",
    "Adderall,d-amphetamine,C[C@H](N)Cc1ccccc1,CHEMBL612,CHEMBL238,CHEMBL238,DAT,IC50,=,288.4,6.54,CHEMBL2390840,2013,10.1016/j.bmcl.2013.03.066,23602445,human DAT uptake assay,d-amphetamine proxy",
    "Methylphenidate,Methylphenidate,COC(=O)C(c1ccccc1)C1CCCCN1,CHEMBL796,CHEMBL238,CHEMBL238,DAT,Ki,=,34.0,7.47,CHEMBL3351632,2014,10.1021/ml500053b,25050161,human DAT binding,direct target binding",
    "Cocaine,Cocaine,COC(=O)[C@H]1[C@@H](OC(=O)c2ccccc2)C[C@@H]2CC[C@H]1N2C,CHEMBL370805,CHEMBL238,CHEMBL238,DAT,IC50,=,240.0,6.62,CHEMBL1139231,2007,10.1021/jm0608614,17228864,DAT uptake inhibition,cocaine comparator",
]


def _write_panel_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    snapshot_path = tmp_path / "endpoint_snapshot.csv"
    snapshot_path.write_text("\n".join(SNAPSHOT_ROWS), encoding="utf-8")
    reference_path = tmp_path / "reference.csv"
    reference_path.write_text("\n".join(REFERENCE_ROWS), encoding="utf-8")
    model_path = tmp_path / "endpoint_model.json"
    report_path = tmp_path / "endpoint_report.json"
    build_demo_endpoint_cpu_artifacts(snapshot_path, model_path, report_path)
    return snapshot_path, reference_path, model_path


def test_candidate_panel_writes_cpu_endpoint_predictions_and_comparators(tmp_path: Path):
    snapshot_path, reference_path, model_path = _write_panel_inputs(tmp_path)
    output_path = tmp_path / "candidate.json"

    report = run_chembl238_candidate_panel(
        snapshot_path=snapshot_path,
        reference_path=reference_path,
        model_path=model_path,
        output_path=output_path,
        run_deep=False,
    )

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved == report
    assert report["candidate"]["label"] == "4B-MAR"
    assert "pIC50" in report["models"]["cpu_endpoint_ridge"]["predictions"]
    assert "pKi" in report["models"]["cpu_endpoint_ridge"]["predictions"]
    assert "Cocaine" in report["reference_panel"]["compounds"]
    assert report["models"]["compact_deep"]["status"] == "not_requested"
    assert report["candidate"]["input_representations"]["smiles_token_sequence"]["valid"]
    assert report["candidate"]["input_representations"]["rdkit_node_graph"]["valid_node_graph"]
    assert report["consensus"]["pIC50"]["model_count"] == 1


def test_candidate_panel_runs_optuna_after_baseline_on_cpu(tmp_path: Path):
    snapshot_path, reference_path, model_path = _write_panel_inputs(tmp_path)
    output_path = tmp_path / "candidate_deep.json"

    report = run_chembl238_candidate_panel(
        snapshot_path=snapshot_path,
        reference_path=reference_path,
        model_path=model_path,
        output_path=output_path,
        endpoints=("pIC50",),
        run_deep=True,
        deep_models=("transformer",),
        deep_epochs=1,
        optuna_trials=1,
        hidden_dim=32,
        batch_size=4,
        device="cpu",
    )

    transformer = report["models"]["compact_deep"]["models"]["transformer"]["pIC50"]
    assert "baseline_run" in transformer
    assert transformer["optuna"]["status"] == "completed"
    assert transformer["optuna"]["trials_completed"] == 1
    assert transformer["optuna"]["best_refit"]["value"] is not None
    assert transformer["optuna"]["best_refit"]["training"]["input_representation"] == "smiles_token_sequence"


def test_candidate_panel_requires_cuda_when_cuda_is_requested(tmp_path: Path):
    if torch.cuda.is_available():
        pytest.skip("CUDA is available on this host.")
    snapshot_path, reference_path, model_path = _write_panel_inputs(tmp_path)

    with pytest.raises(ValueError, match="CUDA was requested"):
        run_chembl238_candidate_panel(
            snapshot_path=snapshot_path,
            reference_path=reference_path,
            model_path=model_path,
            output_path=tmp_path / "candidate_cuda.json",
            endpoints=("pIC50",),
            run_deep=True,
            deep_models=("transformer",),
            deep_epochs=1,
            optuna_trials=0,
            device="cuda",
        )


def test_qsar_comparison_writes_candidate_table(tmp_path: Path):
    snapshot_path, reference_path, model_path = _write_panel_inputs(tmp_path)
    candidate_set = tmp_path / "candidates.csv"
    candidate_set.write_text(
        "\n".join(
            [
                "label,smiles,chemotype,source",
                "Phenethylamine,C1=CC=C(C=C1)CCN,phenethylamine_core,test",
                "Aminorex,C1C(OC(=N1)N)C2=CC=CC=C2,aminorex_core,test",
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "comparison.json"
    table_path = tmp_path / "comparison.csv"

    report = run_chembl238_qsar_comparison(
        candidate_set_path=candidate_set,
        snapshot_path=snapshot_path,
        reference_path=reference_path,
        model_path=model_path,
        output_path=output_path,
        table_output_path=table_path,
        run_deep=False,
    )

    assert output_path.exists()
    assert table_path.exists()
    assert len(report["candidate_reports"]) == 2
    assert len(report["comparison_table"]) == 4
    assert {"Phenethylamine", "Aminorex"} == set(report["candidate_reports"])
