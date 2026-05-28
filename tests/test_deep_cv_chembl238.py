"""Cross-validation tests for compact deep CHEMBL238 model comparisons."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.run_deep_cv_chembl238 import run_deep_cv_chembl238


def test_run_deep_cv_chembl238_compares_gnn_and_multimodal_elt(tmp_path: Path):
    snapshot_path = tmp_path / "chembl238_snapshot.csv"
    snapshot_path.write_text(
        "\n".join(
            [
                "snapshot_id,target,target_name,molecule_chembl_id,canonical_smiles,pIC50,split,scaffold_smiles,source",
                "s1,CHEMBL224,5HT2A,M1,NCCc1ccccc1,7.1,train,c1ccccc1,ChEMBL",
                "s1,CHEMBL224,5HT2A,M2,COc1ccc(CCN)cc1,7.4,train,c1ccccc1,ChEMBL",
                "s1,CHEMBL218,CB1,M3,CCCCCc1ccccc1O,6.7,train,c1ccccc1,ChEMBL",
                "s1,CHEMBL253,CB2,M4,CCCCOc1ccccc1,6.4,train,c1ccccc1,ChEMBL",
                "s1,CHEMBL233,mu-opioid,M5,CCN(CC)C(=O)c1ccccc1,8.0,train,c1ccccc1,ChEMBL",
                "s1,CHEMBL236,delta-opioid,M6,CN1CCC(CC1)c1ccccc1,7.8,train,C1CCCCC1,ChEMBL",
                "s1,CHEMBL237,kappa-opioid,M7,c1ccccc1C2CCNCC2,7.5,train,C1CCCCC1,ChEMBL",
                "s1,CHEMBL238,DAT,M8,CC(N)Cc1ccccc1,6.2,train,c1ccccc1,ChEMBL",
                "s1,CHEMBL238,DAT,M9,CC(C)NCCc1ccccc1,6.5,scaffold_test,c1ccccc1,ChEMBL",
                "s1,CHEMBL238,DAT,M10,CCO,4.3,train,CCO,ChEMBL",
                "s1,CHEMBL224,5HT2A,M11,c1ccncc1,5.3,train,c1ccncc1,ChEMBL",
                "s1,CHEMBL218,CB1,M12,C1CCCCC1,5.8,external,C1CCCCC1,ChEMBL",
            ]
        ),
        encoding="utf-8",
    )
    report_path = tmp_path / "deep_cv_report.json"

    report = run_deep_cv_chembl238(
        snapshot_path=snapshot_path,
        report_path=report_path,
        target="ALL",
        models=("multimodal_elt", "gnn"),
        folds=2,
        epochs=1,
        hidden_dim=16,
        descriptor_token_count=2,
        image_grid_size=8,
        image_patch_size=4,
        batch_size=4,
        learning_rate=1e-3,
        random_seed=9,
        max_rows=0,
    )

    saved = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved == report
    assert report["target"] == "ALL"
    assert report["dataset"]["cv_rows"] == 11
    assert report["fold_policy"]["method"] == "stable_scaffold_hash_modulo"
    assert report["fold_policy"]["max_rows"] is None
    assert report["category_rules"]["mode"] == "target_and_structure_multilabel"
    assert report["training"]["target_standardization"] == "train_fold_zscore_inverse_transform"
    assert report["external_references"]["github_repo"] == "zapabob/elastic-looped-transformer"
    assert report["external_references"]["hf_author"] == "zapabobouj"
    assert report["external_references"]["arxiv_id"] == "2604.09168"
    assert set(report["models"]) == {"multimodal_elt", "gnn"}
    expected_categories = {"psychedelic", "cannabinoid", "opioid", "phenethylamine"}

    for model_report in report["models"].values():
        assert len(model_report["folds"]) == 2
        assert {"r2", "rmse", "mae", "mse_loss", "n"}.issubset(model_report["mean_metrics"])
        assert expected_categories.issubset(model_report["category_metrics"])
        for fold in model_report["folds"]:
            assert fold["test_metrics"]["n"] > 0
            assert "mse_loss" in fold["test_metrics"]
            assert "train_size" in fold
            assert "test_size" in fold
            assert "train_loss_final" in fold

    elt_fold = report["models"]["multimodal_elt"]["folds"][0]
    assert "graph_summary" in elt_fold["evidence_channels"]
    for category in expected_categories:
        assert {"r2", "rmse", "mae", "mse_loss", "n"}.issubset(
            report["models"]["multimodal_elt"]["category_metrics"][category]
        )
