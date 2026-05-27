"""Smoke-run tests for CHEMBL238 ELT evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_elt_chembl238_smoke import run_elt_chembl238_smoke


def test_run_elt_chembl238_smoke_writes_loop_predictions(tmp_path: Path):
    snapshot_path = tmp_path / "chembl238_snapshot.csv"
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
    analysis_path = tmp_path / "methylphenidate_analysis.json"
    analysis_path.write_text(
        json.dumps(
            {
                "pIC50_mean": 7.3719,
                "model_prediction": {"pIC50_prediction": 6.04},
            }
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "elt_report.json"

    report = run_elt_chembl238_smoke(
        snapshot_path=snapshot_path,
        report_path=report_path,
        analysis_path=analysis_path,
        epochs=1,
        hidden_dim=16,
        token_count=2,
        loop_count=2,
        batch_size=3,
        learning_rate=1e-3,
        random_seed=7,
    )

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == report
    assert report["model_kind"] == "elastic_looped_transformer_pic50"
    assert report["reference"]["github_repo"] == "zapabob/elastic-looped-transformer"
    assert report["dataset"]["rows"] == 6
    assert {"train", "scaffold_test", "external"}.issubset(report["metrics"])
    assert set(report["methylphenidate_loop_predictions"]) == {"1", "2"}
    assert report["methylphenidate_literature_comparison"]["literature_mean_pIC50"] == 7.3719
    assert "2" in report["methylphenidate_literature_comparison"]["loops"]
    for prediction in report["methylphenidate_loop_predictions"].values():
        assert prediction["pIC50_prediction"] is not None
        assert prediction["uncertainty"] > 0
