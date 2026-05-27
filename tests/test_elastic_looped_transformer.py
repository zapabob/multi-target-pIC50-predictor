"""Tests for the elastic-looped transformer pIC50 model path."""

from __future__ import annotations

import torch

from src.models.elastic_looped_transformer import (
    ElasticLoopedPIC50Model,
    LitElasticLoopedPIC50,
    default_loop_steps,
)


def test_default_loop_steps_are_budget_normalized():
    steps = default_loop_steps(num_loops=4)

    assert steps == (0.25, 0.25, 0.25, 0.25)
    assert sum(steps) == 1.0


def test_elastic_looped_pic50_model_supports_variable_compute_budget():
    batch_size = 8
    input_dim = 32
    model = ElasticLoopedPIC50Model(
        input_dim=input_dim,
        hidden_dim=64,
        token_count=4,
        num_heads=4,
        dropout=0.0,
    )

    short = model(torch.randn(batch_size, input_dim), loop_steps=(0.5, 0.5))
    long = model(torch.randn(batch_size, input_dim), loop_steps=(0.25, 0.25, 0.25, 0.25))

    assert short.pic50.shape == (batch_size, 1)
    assert long.pic50.shape == (batch_size, 1)
    assert short.uncertainty.shape == (batch_size, 1)
    assert torch.all(short.uncertainty > 0)
    assert short.loop_count == 2
    assert long.loop_count == 4
    assert short.evidence_channels == ("molecular_descriptor", "elastic_looped_transformer")


def test_elastic_looped_pic50_model_rejects_bad_loop_schedule():
    model = ElasticLoopedPIC50Model(input_dim=16, hidden_dim=32, token_count=2, num_heads=4)

    try:
        model(torch.randn(2, 16), loop_steps=(0.5, 0.0))
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("Expected invalid loop schedule to fail")


def test_lit_elastic_looped_pic50_training_step_accepts_structured_batch():
    batch_size = 6
    input_dim = 20
    model = LitElasticLoopedPIC50(
        input_dim=input_dim,
        hidden_dim=32,
        token_count=4,
        num_heads=4,
        dropout=0.0,
    )

    batch = {
        "features": torch.randn(batch_size, input_dim),
        "y": torch.randn(batch_size, 1),
        "loop_steps": (0.5, 0.5),
    }

    loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor)
    assert loss.item() >= 0
    assert not torch.isnan(loss)
