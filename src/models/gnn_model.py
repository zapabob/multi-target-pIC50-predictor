"""
Graph Neural Network models for molecular pIC50 prediction.
RTX3060 optimized implementation using PyTorch Geometric.
"""

import logging

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error, r2_score
from torch_geometric.nn import GATConv, GCNConv, global_add_pool, global_max_pool, global_mean_pool


class GNNModel(nn.Module):
    """Graph Neural Network for pIC50 prediction.

    Lightweight architecture optimized for RTX3060:
    - 2-3 Graph Attention/Convolution layers
    - 128-256 hidden dimensions
    - Global pooling + MLP head
    """

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        pool_method: str = "mean",
        use_edge_features: bool = True,
        gnn_type: str = "gat",  # 'gat' or 'gcn'
    ):
        """Initialize GNN model.

        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            num_heads: Number of attention heads (GAT only)
            dropout: Dropout rate
            pool_method: Global pooling method ('mean', 'max', 'add')
            use_edge_features: Whether to use edge features
            gnn_type: Type of GNN ('gat' or 'gcn')
        """
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.edge_feature_dim = edge_feature_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.pool_method = pool_method
        self.use_edge_features = use_edge_features
        self.gnn_type = gnn_type

        # Input projection
        self.input_proj = nn.Linear(node_feature_dim, hidden_dim)

        # Edge projection (if using edge features)
        if use_edge_features and gnn_type == "gat":
            self.edge_proj = nn.Linear(edge_feature_dim, hidden_dim)

        # GNN layers
        self.gnn_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for i in range(num_layers):
            if gnn_type == "gat":
                # Graph Attention Network
                in_channels = hidden_dim if i == 0 else hidden_dim * num_heads
                out_channels = hidden_dim

                self.gnn_layers.append(
                    GATConv(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        heads=num_heads,
                        dropout=dropout,
                        concat=True if i < num_layers - 1 else False,
                        edge_dim=hidden_dim if use_edge_features else None,
                    )
                )
            elif gnn_type == "gcn":
                # Graph Convolutional Network
                in_channels = hidden_dim
                self.gnn_layers.append(GCNConv(in_channels, hidden_dim))

            # Batch normalization
            bn_dim = (
                hidden_dim * num_heads if (gnn_type == "gat" and i < num_layers - 1) else hidden_dim
            )
            self.batch_norms.append(nn.BatchNorm1d(bn_dim))

        # Global pooling
        if pool_method == "mean":
            self.pool = global_mean_pool
        elif pool_method == "max":
            self.pool = global_max_pool
        elif pool_method == "add":
            self.pool = global_add_pool
        else:
            raise ValueError(f"Unknown pooling method: {pool_method}")

        # Output MLP
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, data):
        """Forward pass.

        Args:
            data: PyTorch Geometric Batch object or Data object

        Returns:
            Graph-level predictions [batch_size, 1]
        """
        # Extract components from data object
        if hasattr(data, "batch"):
            # Batch object
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
            batch = data.batch
        else:
            # Single Data object
            x = data.x
            edge_index = data.edge_index
            edge_attr = data.edge_attr if hasattr(data, "edge_attr") else None
            batch = None
        # Project input
        x = self.input_proj(x)
        x = F.relu(x)

        # Project edge features
        if self.use_edge_features and edge_attr is not None and self.gnn_type == "gat":
            edge_attr = self.edge_proj(edge_attr)

        # Apply GNN layers
        for i, (gnn_layer, batch_norm) in enumerate(
            zip(self.gnn_layers, self.batch_norms, strict=True)
        ):
            if self.gnn_type == "gat" and self.use_edge_features and edge_attr is not None:
                x_new = gnn_layer(x, edge_index, edge_attr=edge_attr)
            else:
                x_new = gnn_layer(x, edge_index)

            x_new = batch_norm(x_new)

            # Apply activation (except for last layer)
            if i < self.num_layers - 1:
                x_new = F.relu(x_new)
                x_new = F.dropout(x_new, p=self.dropout, training=self.training)

            # Residual connection (if dimensions match)
            if x.size(-1) == x_new.size(-1):
                x = x + x_new
            else:
                x = x_new

        # Global pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = self.pool(x, batch)

        # Output MLP
        out = self.mlp(x)

        return out


class LitGNN(pl.LightningModule):
    """PyTorch Lightning module for GNN pIC50 prediction."""

    def __init__(
        self,
        node_feature_dim: int,
        edge_feature_dim: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.2,
        pool_method: str = "mean",
        use_edge_features: bool = True,
        gnn_type: str = "gat",
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1,
    ):
        """Initialize Lightning GNN module.

        Args:
            node_feature_dim: Dimension of node features
            edge_feature_dim: Dimension of edge features
            learning_rate: Learning rate
            weight_decay: Weight decay
            hidden_dim: Hidden layer dimension
            num_layers: Number of GNN layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            pool_method: Global pooling method
            use_edge_features: Whether to use edge features
            gnn_type: Type of GNN ('gat' or 'gcn')
            scheduler_step_size: Learning rate scheduler step size
            scheduler_gamma: Learning rate scheduler gamma
        """
        super().__init__()

        self.save_hyperparameters()

        # Model
        self.model = GNNModel(
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            pool_method=pool_method,
            use_edge_features=use_edge_features,
            gnn_type=gnn_type,
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma

        self.logger_instance = logging.getLogger(__name__)

    def forward(self, batch):
        """Forward pass.

        Args:
            batch: PyTorch Geometric batch

        Returns:
            Predictions
        """
        return self.model(batch)

    def training_step(self, batch, batch_idx):
        """Training step.

        Args:
            batch: PyTorch Geometric batch with 'y' attribute
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        y_hat = self(batch)
        y = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
        loss = self.criterion(y_hat, y)

        # Log metrics
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        # Calculate R² score
        y_np = y.cpu().numpy().flatten()
        y_hat_np = y_hat.detach().cpu().numpy().flatten()
        r2 = r2_score(y_np, y_hat_np)
        self.log(
            "train_r2", r2, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs
        )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step.

        Args:
            batch: PyTorch Geometric batch with 'y' attribute
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        y_hat = self(batch)
        y = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
        loss = self.criterion(y_hat, y)

        # Log metrics
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch.num_graphs,
        )

        # Calculate R² score
        y_np = y.cpu().numpy().flatten()
        y_hat_np = y_hat.detach().cpu().numpy().flatten()
        r2 = r2_score(y_np, y_hat_np)
        self.log(
            "val_r2", r2, on_step=False, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs
        )

        return loss

    def test_step(self, batch, batch_idx):
        """Test step.

        Args:
            batch: PyTorch Geometric batch with 'y' attribute
            batch_idx: Batch index

        Returns:
            Dictionary with test metrics
        """
        y_hat = self(batch)
        y = batch.y.unsqueeze(1) if batch.y.dim() == 1 else batch.y
        loss = self.criterion(y_hat, y)

        # Calculate metrics
        y_np = y.cpu().numpy().flatten()
        y_hat_np = y_hat.detach().cpu().numpy().flatten()
        r2 = r2_score(y_np, y_hat_np)
        rmse = np.sqrt(mean_squared_error(y_np, y_hat_np))

        # Log metrics
        self.log("test_loss", loss, batch_size=batch.num_graphs)
        self.log("test_r2", r2, batch_size=batch.num_graphs)
        self.log("test_rmse", rmse, batch_size=batch.num_graphs)

        return {"test_loss": loss, "test_r2": torch.tensor(r2), "test_rmse": torch.tensor(rmse)}

    def configure_optimizers(self):
        """Configure optimizers and schedulers.

        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.scheduler_step_size, gamma=self.scheduler_gamma
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
        }

    def predict_step(self, batch, batch_idx):
        """Prediction step.

        Args:
            batch: PyTorch Geometric batch
            batch_idx: Batch index

        Returns:
            Predictions tensor
        """
        return self(batch)
