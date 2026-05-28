"""Tests for ViT-style multimodal elastic-looped pIC50 models."""

from __future__ import annotations

import pytest
import torch

from src.models.elastic_looped_transformer import MultimodalElasticLoopedPIC50Model


def test_multimodal_elt_uses_vit_image_patches_and_graph_summary():
    model = MultimodalElasticLoopedPIC50Model(
        descriptor_dim=12,
        image_feature_dim=64,
        image_grid_size=8,
        image_patch_size=4,
        graph_feature_dim=5,
        hidden_dim=32,
        descriptor_token_count=2,
        num_heads=4,
        dropout=0.0,
        default_num_loops=3,
    )

    output = model(
        descriptor_features=torch.randn(4, 12),
        image_features=torch.randn(4, 64),
        graph_features=torch.randn(4, 5),
        loop_steps=(1.0, 2.0, 1.0),
    )

    assert model.image_patch_count == 4
    assert output.pic50.shape == (4, 1)
    assert output.uncertainty.shape == (4, 1)
    assert output.loop_count == 3
    assert output.normalized_loop_steps == (0.25, 0.5, 0.25)
    assert output.evidence_channels == (
        "molecular_descriptor",
        "molecule_render_vit_patch",
        "graph_summary",
        "elastic_looped_transformer",
    )
    assert torch.all(output.uncertainty > 0)


def test_multimodal_elt_rejects_image_width_that_cannot_form_patches():
    model = MultimodalElasticLoopedPIC50Model(
        descriptor_dim=12,
        image_feature_dim=64,
        image_grid_size=8,
        image_patch_size=4,
        hidden_dim=32,
        descriptor_token_count=2,
        num_heads=4,
    )

    with pytest.raises(ValueError, match="image_features width"):
        model(
            descriptor_features=torch.randn(2, 12),
            image_features=torch.randn(2, 63),
        )
