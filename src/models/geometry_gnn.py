"""3D geometry-aware GNN model adapters for SchNet and DimeNet++."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

GeometryModelName = Literal["schnet", "dimenet++"]


@dataclass
class GeometryGNNConfig:
    """Configuration for a geometry-aware graph model."""

    model_name: GeometryModelName = "schnet"
    hidden_channels: int = 128
    num_filters: int = 128
    num_interactions: int = 6
    num_gaussians: int = 50
    cutoff: float = 10.0
    out_channels: int = 1
    dropout: float = 0.1


class GeometryGNNRegressor:
    """Factory wrapper around PyTorch Geometric SchNet/DimeNet++ models."""

    def __init__(self, config: GeometryGNNConfig | None = None):
        self.config = config or GeometryGNNConfig()
        self.model = self._build_model()

    def _build_model(self):
        try:
            from torch_geometric.nn.models import DimeNetPlusPlus, SchNet
        except ImportError as exc:
            raise ImportError(
                "torch-geometric with geometry model support is required for SchNet or DimeNet++."
            ) from exc

        cfg = self.config
        if cfg.model_name == "schnet":
            return SchNet(
                hidden_channels=cfg.hidden_channels,
                num_filters=cfg.num_filters,
                num_interactions=cfg.num_interactions,
                num_gaussians=cfg.num_gaussians,
                cutoff=cfg.cutoff,
                out_channels=cfg.out_channels,
            )
        if cfg.model_name == "dimenet++":
            return DimeNetPlusPlus(
                hidden_channels=cfg.hidden_channels,
                out_channels=cfg.out_channels,
                num_blocks=cfg.num_interactions,
                int_emb_size=64,
                basis_emb_size=8,
                out_emb_channels=cfg.hidden_channels,
                num_spherical=7,
                num_radial=6,
                cutoff=cfg.cutoff,
            )
        raise ValueError(f"Unsupported geometry model: {cfg.model_name}")

    def __call__(self, z, pos, batch=None):
        """Run the wrapped model.

        Args:
            z: Atomic numbers.
            pos: 3D coordinates.
            batch: Optional graph batch assignment.
        """
        return self.model(z=z, pos=pos, batch=batch)
