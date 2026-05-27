"""README asset generation tests for the pharma MVP evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.reporting.pharma_mvp_assets import build_pharma_mvp_assets


def test_build_pharma_mvp_assets_writes_stats_and_errorbar_plot(tmp_path: Path):
    analysis_path = tmp_path / "methylphenidate_analysis.json"
    benchmark_path = tmp_path / "chembl238_benchmark.json"
    stats_output = tmp_path / "readme_stats.json"
    figure_output = tmp_path / "errorbar.png"

    analysis_path.write_text(
        json.dumps(
            {
                "compound": "methylphenidate plus methylphenidate hydrochloride records",
                "target": "CHEMBL238 DAT",
                "literature_ic50_nM": [17.0, 19.9, 121.7, 79.0],
                "literature_pIC50": [7.7696, 7.7011, 6.9147, 7.1024],
                "pIC50_mean": 7.3719,
                "pIC50_sd": 0.4275,
                "pIC50_sem": 0.2137,
                "pIC50_95ci": [6.6917, 8.0521],
                "geometric_mean_ic50_nM": 42.4673,
                "methylphenidate_inactive_by_rule_rows": 0,
                "model_prediction": {
                    "pIC50_prediction": 6.04,
                    "uncertainty": 0.87,
                    "applicability_domain": {"in_domain": True},
                    "device": "cpu",
                },
                "model_predicted_ic50_nM": 912.0108,
                "model_minus_literature_mean_pIC50": -1.3319,
                "model_vs_literature_one_sample_t": -6.2317,
                "model_vs_literature_two_sided_p": 0.008333,
            }
        ),
        encoding="utf-8",
    )
    benchmark_path.write_text(
        json.dumps(
            {
                "benchmark_dataset": {"path": "data/chembl238_pic50_snapshot.csv", "rows": 2382},
                "targets": {
                    "CHEMBL238": {
                        "metrics": {
                            "train": {"r2": 0.2450, "rmse": 1.0474, "mae": 0.8553, "n": 1762},
                            "scaffold_test": {
                                "r2": 0.3263,
                                "rmse": 0.8699,
                                "mae": 0.7090,
                                "n": 359,
                            },
                            "external": {
                                "r2": 0.2062,
                                "rmse": 1.0197,
                                "mae": 0.8295,
                                "n": 261,
                            },
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    summary = build_pharma_mvp_assets(
        analysis_path=analysis_path,
        benchmark_path=benchmark_path,
        stats_output_path=stats_output,
        figure_output_path=figure_output,
    )

    saved_summary = json.loads(stats_output.read_text(encoding="utf-8"))
    assert saved_summary == summary
    assert summary["effect_size"]["cohen_dz"] == pytest.approx(-3.1159, abs=0.001)
    assert summary["power"]["observed_two_sided_alpha_0_05"] == pytest.approx(
        0.9754,
        abs=0.001,
    )
    assert summary["benchmark_metrics"]["CHEMBL238"]["external"]["rmse"] == 1.0197
    assert figure_output.read_bytes().startswith(b"\x89PNG")
