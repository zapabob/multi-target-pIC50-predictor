"""Elastic-looped Transformer model for molecular pIC50 regression.

This module adapts the core idea from `zapabob/elastic-looped-transformer`:
share one Transformer block across a selectable number of loop iterations, so
the same checkpoint can trade compute budget for iterative refinement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim


def default_loop_steps(num_loops: int) -> tuple[float, ...]:
    """Return a normalized loop-step schedule for an inference budget."""

    if num_loops < 1:
        raise ValueError("num_loops must be at least 1")
    step = 1.0 / num_loops
    return tuple(step for _ in range(num_loops))


def _validate_loop_steps(loop_steps: tuple[float, ...]) -> tuple[float, ...]:
    if not loop_steps:
        raise ValueError("loop_steps must not be empty")
    if any(step <= 0 for step in loop_steps):
        raise ValueError("loop_steps must contain only positive values")
    total = sum(loop_steps)
    if total <= 0:
        raise ValueError("loop_steps must sum to a positive value")
    return tuple(float(step) / total for step in loop_steps)


@dataclass(frozen=True)
class ElasticLoopedPIC50Output:
    """Output bundle for pIC50 prediction with loop telemetry."""

    pic50: torch.Tensor
    uncertainty: torch.Tensor
    loop_count: int
    normalized_loop_steps: tuple[float, ...]
    evidence_channels: tuple[str, ...]


class LoopStepEmbedding(nn.Module):
    """Embed scalar loop-time and loop-step values into hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, elapsed: float, step_size: float, batch_size: int, device: torch.device):
        step = torch.tensor([[elapsed, step_size]], dtype=torch.float32, device=device)
        return self.net(step).unsqueeze(1).expand(batch_size, 1, -1)


class ElasticLoopedPIC50Model(nn.Module):
    """Descriptor regressor with a weight-shared elastic Transformer loop.

    The model converts flat molecular descriptors into a short token sequence,
    applies the same Transformer block for each selected loop step, then returns
    pIC50 and a positive uncertainty proxy.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        token_count: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        default_num_loops: int = 4,
    ):
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if token_count < 1:
            raise ValueError("token_count must be positive")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.token_count = token_count
        self.num_heads = num_heads
        self.default_num_loops = default_num_loops

        self.input_projection = nn.Linear(input_dim, hidden_dim * token_count)
        self.position_embedding = nn.Parameter(torch.randn(1, token_count, hidden_dim) * 0.02)
        self.step_embedding = LoopStepEmbedding(hidden_dim)
        self.shared_block = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),
        )

    def forward(
        self,
        features: torch.Tensor,
        loop_steps: tuple[float, ...] | None = None,
    ) -> ElasticLoopedPIC50Output:
        """Predict pIC50 with a selectable loop budget."""

        if features.dim() != 2:
            raise ValueError("features must have shape [batch, input_dim]")
        if features.size(1) != self.input_dim:
            raise ValueError(f"features width must be {self.input_dim}")

        normalized_steps = _validate_loop_steps(
            loop_steps or default_loop_steps(self.default_num_loops)
        )
        batch_size = features.size(0)
        x = self.input_projection(features)
        x = x.view(batch_size, self.token_count, self.hidden_dim)
        x = x + self.position_embedding

        elapsed = 0.0
        for step in normalized_steps:
            x = x + self.step_embedding(elapsed, step, batch_size, features.device)
            x = self.shared_block(x)
            elapsed += step

        pooled = self.norm(x).mean(dim=1)
        pic50 = self.regression_head(pooled)
        uncertainty = self.uncertainty_head(pooled) + 1e-6
        return ElasticLoopedPIC50Output(
            pic50=pic50,
            uncertainty=uncertainty,
            loop_count=len(normalized_steps),
            normalized_loop_steps=normalized_steps,
            evidence_channels=("molecular_descriptor", "elastic_looped_transformer"),
        )


class LitElasticLoopedPIC50(pl.LightningModule):
    """PyTorch Lightning wrapper for the elastic-looped pIC50 model."""

    def __init__(
        self,
        input_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        hidden_dim: int = 256,
        token_count: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        default_num_loops: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = ElasticLoopedPIC50Model(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            token_count=token_count,
            num_heads=num_heads,
            dropout=dropout,
            default_num_loops=default_num_loops,
        )
        self.criterion = nn.MSELoss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

    def forward(
        self,
        features: torch.Tensor,
        loop_steps: tuple[float, ...] | None = None,
    ) -> ElasticLoopedPIC50Output:
        """Forward pass."""

        return self.model(features, loop_steps=loop_steps)

    def _step(self, batch: dict[str, Any] | tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        if isinstance(batch, dict):
            features = batch["features"]
            y = batch["y"]
            loop_steps = batch.get("loop_steps")
        else:
            features, y = batch
            loop_steps = None
        if y.dim() == 1:
            y = y.unsqueeze(1)
        output = self(features, loop_steps=loop_steps)
        loss = self.criterion(output.pic50, y)
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            f"{stage}_loop_count",
            float(output.loop_count),
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        return loss

    def training_step(
        self,
        batch: dict[str, Any] | tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Training step."""

        return self._step(batch, "train")

    def validation_step(
        self,
        batch: dict[str, Any] | tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        """Validation step."""

        return self._step(batch, "val")

    def configure_optimizers(self) -> dict[str, Any]:
        """Configure optimizer and scheduler."""

        optimizer = optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
