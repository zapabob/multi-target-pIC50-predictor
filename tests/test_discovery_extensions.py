"""Tests for integrated drug-discovery extension modules."""

import pytest

pytest.importorskip("rdkit")

from src.admet import ADMETPredictor
from src.features.structure3d import ETKDGConformerGenerator
from src.pipeline.compound_assessment import CompoundAssessmentPipeline
from src.synthesis import SyntheticAccessibilityScorer

ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"


def test_admet_profile_for_valid_smiles():
    profile = ADMETPredictor().predict(ASPIRIN)

    assert profile.success
    assert profile.descriptors["mol_wt"] > 100
    assert "developability_proxy" in profile.scores


def test_synthetic_accessibility_profile_for_valid_smiles():
    profile = SyntheticAccessibilityScorer().score(ASPIRIN)

    assert profile.success
    assert 1.0 <= profile.scores["sa_score_proxy"] <= 10.0
    assert 1.0 <= profile.scores["scscore_proxy"] <= 5.0


def test_etkdg_conformer_generation():
    result = ETKDGConformerGenerator(num_conformers=3).generate(ASPIRIN)

    assert result.success
    assert result.conformer_count >= 1
    assert "radius_of_gyration" in result.descriptors


def test_compound_assessment_pipeline_without_pic50_model():
    result = CompoundAssessmentPipeline(target="CHEMBL238").assess(
        ASPIRIN,
        include_3d=True,
        include_reactions=True,
        include_image=False,
    )

    assert result.target == "CHEMBL238"
    assert result.pIC50_prediction is None
    assert result.admet["success"]
    assert result.synthesis["success"]
    assert result.structure3d["success"]
    assert isinstance(result.retrosynthesis, list)
