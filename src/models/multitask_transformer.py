"""
Multi-task learning for simultaneous prediction of multiple targets.
Optimized for 7 targets: DAT, 5HT2A, CB1, CB2, μ/δ/κ-opioid receptors.
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import r2_score, mean_squared_error
import pandas as pd


class MultiTaskTransformer(nn.Module):
    """Multi-task Transformer for simultaneous pIC50 prediction across multiple targets.
    
    Architecture:
    - Shared Transformer encoder
    - Target-specific regression heads
    - Uncertainty weighting for task balancing
    """
    
    def __init__(
        self,
        input_dim: int,
        target_names: List[str],
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_uncertainty_weighting: bool = True
    ):
        """Initialize multi-task Transformer.
        
        Args:
            input_dim: Input feature dimension
            target_names: List of target names
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_uncertainty_weighting: Whether to use uncertainty weighting
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.target_names = target_names
        self.num_targets = len(target_names)
        self.hidden_dim = hidden_dim
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding (for sequence-like input)
        self.pos_encoding = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Shared Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Target-specific regression heads
        self.target_heads = nn.ModuleDict()
        for target in target_names:
            self.target_heads[target] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)
            )
        
        # Uncertainty weighting parameters (learnable)
        if use_uncertainty_weighting:
            self.log_vars = nn.Parameter(torch.zeros(self.num_targets))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Dictionary of target predictions
        """
        batch_size = x.size(0)
        
        # Project input
        x = self.input_projection(x)  # [batch_size, hidden_dim]
        
        # Add sequence dimension and positional encoding
        x = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        x = x + self.pos_encoding[:, :x.size(1), :]
        
        # Apply transformer encoder
        x = self.transformer_encoder(x)  # [batch_size, 1, hidden_dim]
        x = x.squeeze(1)  # [batch_size, hidden_dim]
        
        # Apply dropout
        x = self.dropout(x)
        
        # Target-specific predictions
        predictions = {}
        for target in self.target_names:
            predictions[target] = self.target_heads[target](x)
        
        return predictions
    
    def get_uncertainty_weights(self) -> torch.Tensor:
        """Get uncertainty weights for multi-task loss.
        
        Returns:
            Uncertainty weights tensor
        """
        if self.use_uncertainty_weighting:
            return 1.0 / (2.0 * torch.exp(self.log_vars))
        else:
            return torch.ones(self.num_targets, device=next(self.parameters()).device)


class LitMultiTaskPIC50(pl.LightningModule):
    """PyTorch Lightning module for multi-task pIC50 prediction."""
    
    def __init__(
        self,
        input_dim: int,
        target_names: List[str],
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_uncertainty_weighting: bool = True,
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.1
    ):
        """Initialize Lightning multi-task module.
        
        Args:
            input_dim: Input feature dimension
            target_names: List of target names
            learning_rate: Learning rate
            weight_decay: Weight decay
            hidden_dim: Hidden layer dimension
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_uncertainty_weighting: Whether to use uncertainty weighting
            scheduler_step_size: Learning rate scheduler step size
            scheduler_gamma: Learning rate scheduler gamma
        """
        super().__init__()
        
        self.save_hyperparameters()
        
        # Model
        self.model = MultiTaskTransformer(
            input_dim=input_dim,
            target_names=target_names,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            use_uncertainty_weighting=use_uncertainty_weighting
        )
        
        # Loss function
        self.criterion = nn.MSELoss(reduction='none')
        
        # Hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_step_size = scheduler_step_size
        self.scheduler_gamma = scheduler_gamma
        self.target_names = target_names
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Metrics storage
        self.train_losses = []
        self.val_losses = []
        self.train_r2_scores = {target: [] for target in target_names}
        self.val_r2_scores = {target: [] for target in target_names}
        
        self.logger_instance = logging.getLogger(__name__)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of target predictions
        """
        return self.model(x)
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        mask: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        """Compute multi-task loss with uncertainty weighting.
        
        Args:
            predictions: Model predictions
            targets: True targets
            mask: Optional mask for missing targets
            
        Returns:
            Total loss
        """
        total_loss = 0.0
        num_valid_targets = 0
        
        # Get uncertainty weights
        if self.use_uncertainty_weighting:
            uncertainty_weights = self.model.get_uncertainty_weights()
        else:
            uncertainty_weights = torch.ones(len(self.target_names), device=self.device)
        
        for i, target in enumerate(self.target_names):
            if target not in predictions or target not in targets:
                continue
            
            pred = predictions[target]
            true = targets[target]
            
            # Apply mask if provided
            if mask is not None and target in mask:
                valid_mask = mask[target].bool()
                if not valid_mask.any():
                    continue
                pred = pred[valid_mask]
                true = true[valid_mask]
            
            # Compute loss for this target
            target_loss = self.criterion(pred.squeeze(), true)
            target_loss = target_loss.mean()
            
            # Apply uncertainty weighting
            if self.use_uncertainty_weighting:
                weight = uncertainty_weights[i]
                precision = 1.0 / (2.0 * torch.exp(self.model.log_vars[i]))
                weighted_loss = weight * target_loss + precision
            else:
                weighted_loss = target_loss
            
            total_loss += weighted_loss
            num_valid_targets += 1
        
        # Normalize by number of valid targets
        if num_valid_targets > 0:
            total_loss = total_loss / num_valid_targets
        
        return total_loss
    
    def training_step(self, batch, batch_idx):
        """Training step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        x, y_dict, mask = batch
        
        predictions = self(x)
        loss = self.compute_loss(predictions, y_dict, mask)
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log individual target losses
        for target in self.target_names:
            if target in predictions and target in y_dict:
                pred = predictions[target].detach().cpu().numpy().flatten()
                true = y_dict[target].detach().cpu().numpy()
                
                # Apply mask if available
                if mask is not None and target in mask:
                    valid_mask = mask[target].bool().cpu().numpy()
                    pred = pred[valid_mask]
                    true = true[valid_mask]
                
                if len(pred) > 0:
                    r2 = r2_score(true, pred)
                    self.log(f'train_r2_{target}', r2, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Loss tensor
        """
        x, y_dict, mask = batch
        
        predictions = self(x)
        loss = self.compute_loss(predictions, y_dict, mask)
        
        # Log metrics
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log individual target losses
        for target in self.target_names:
            if target in predictions and target in y_dict:
                pred = predictions[target].detach().cpu().numpy().flatten()
                true = y_dict[target].detach().cpu().numpy()
                
                # Apply mask if available
                if mask is not None and target in mask:
                    valid_mask = mask[target].bool().cpu().numpy()
                    pred = pred[valid_mask]
                    true = true[valid_mask]
                
                if len(pred) > 0:
                    r2 = r2_score(true, pred)
                    self.log(f'val_r2_{target}', r2, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Dictionary with test metrics
        """
        x, y_dict, mask = batch
        
        predictions = self(x)
        loss = self.compute_loss(predictions, y_dict, mask)
        
        # Calculate metrics for each target
        test_metrics = {'test_loss': loss}
        
        for target in self.target_names:
            if target in predictions and target in y_dict:
                pred = predictions[target].detach().cpu().numpy().flatten()
                true = y_dict[target].detach().cpu().numpy()
                
                # Apply mask if available
                if mask is not None and target in mask:
                    valid_mask = mask[target].bool().cpu().numpy()
                    pred = pred[valid_mask]
                    true = true[valid_mask]
                
                if len(pred) > 0:
                    r2 = r2_score(true, pred)
                    mse = mean_squared_error(true, pred)
                    rmse = np.sqrt(mse)
                    
                    test_metrics[f'test_r2_{target}'] = r2
                    test_metrics[f'test_rmse_{target}'] = rmse
        
        # Log metrics
        for key, value in test_metrics.items():
            self.log(key, value)
        
        return test_metrics
    
    def configure_optimizers(self):
        """Configure optimizers and schedulers.
        
        Returns:
            Dictionary with optimizer and scheduler configuration
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.scheduler_step_size,
            gamma=self.scheduler_gamma
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    
    def predict_step(self, batch, batch_idx):
        """Prediction step.
        
        Args:
            batch: Batch data
            batch_idx: Batch index
            
        Returns:
            Predictions dictionary
        """
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        return self(x)


class MultiTaskDataModule(pl.LightningDataModule):
    """Data module for multi-task learning."""
    
    def __init__(
        self,
        X: np.ndarray,
        y_dict: Dict[str, np.ndarray],
        target_names: List[str],
        batch_size: int = 32,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        random_state: int = 42
    ):
        """Initialize multi-task data module.
        
        Args:
            X: Input features
            y_dict: Dictionary of target arrays
            target_names: List of target names
            batch_size: Batch size
            train_ratio: Training data ratio
            val_ratio: Validation data ratio
            random_state: Random state
        """
        super().__init__()
        
        self.X = X
        self.y_dict = y_dict
        self.target_names = target_names
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.random_state = random_state
        
        # Create masks for missing values
        self.mask = {}
        for target in target_names:
            if target in y_dict:
                self.mask[target] = ~np.isnan(y_dict[target])
            else:
                self.mask[target] = np.zeros(len(X), dtype=bool)
    
    def setup(self, stage: Optional[str] = None):
        """Setup data splits."""
        n_samples = len(self.X)
        indices = np.arange(n_samples)
        np.random.seed(self.random_state)
        np.random.shuffle(indices)
        
        # Calculate split indices
        n_train = int(n_samples * self.train_ratio)
        n_val = int(n_samples * self.val_ratio)
        
        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        
        # Split data
        self.X_train = self.X[train_indices]
        self.X_val = self.X[val_indices]
        self.X_test = self.X[test_indices]
        
        # Split targets and masks
        self.y_train = {}
        self.y_val = {}
        self.y_test = {}
        self.mask_train = {}
        self.mask_val = {}
        self.mask_test = {}
        
        for target in self.target_names:
            if target in self.y_dict:
                y_target = self.y_dict[target]
                self.y_train[target] = y_target[train_indices]
                self.y_val[target] = y_target[val_indices]
                self.y_test[target] = y_target[test_indices]
                
                mask_target = self.mask[target]
                self.mask_train[target] = mask_target[train_indices]
                self.mask_val[target] = mask_target[val_indices]
                self.mask_test[target] = mask_target[test_indices]
    
    def train_dataloader(self):
        """Training dataloader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Convert to tensors
        X_tensor = torch.tensor(self.X_train, dtype=torch.float32)
        y_tensors = {target: torch.tensor(self.y_train[target], dtype=torch.float32) 
                    for target in self.target_names if target in self.y_train}
        mask_tensors = {target: torch.tensor(self.mask_train[target], dtype=torch.bool)
                       for target in self.target_names if target in self.mask_train}
        
        dataset = MultiTaskDataset(X_tensor, y_tensors, mask_tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        """Validation dataloader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Convert to tensors
        X_tensor = torch.tensor(self.X_val, dtype=torch.float32)
        y_tensors = {target: torch.tensor(self.y_val[target], dtype=torch.float32) 
                    for target in self.target_names if target in self.y_val}
        mask_tensors = {target: torch.tensor(self.mask_val[target], dtype=torch.bool)
                       for target in self.target_names if target in self.mask_val}
        
        dataset = MultiTaskDataset(X_tensor, y_tensors, mask_tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
    
    def test_dataloader(self):
        """Test dataloader."""
        from torch.utils.data import DataLoader, TensorDataset
        
        # Convert to tensors
        X_tensor = torch.tensor(self.X_test, dtype=torch.float32)
        y_tensors = {target: torch.tensor(self.y_test[target], dtype=torch.float32) 
                    for target in self.target_names if target in self.y_test}
        mask_tensors = {target: torch.tensor(self.mask_test[target], dtype=torch.bool)
                       for target in self.target_names if target in self.mask_test}
        
        dataset = MultiTaskDataset(X_tensor, y_tensors, mask_tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False)


class MultiTaskDataset(torch.utils.data.Dataset):
    """Dataset for multi-task learning."""
    
    def __init__(
        self,
        X: torch.Tensor,
        y_dict: Dict[str, torch.Tensor],
        mask_dict: Dict[str, torch.Tensor]
    ):
        """Initialize multi-task dataset.
        
        Args:
            X: Input features
            y_dict: Dictionary of target tensors
            mask_dict: Dictionary of mask tensors
        """
        self.X = X
        self.y_dict = y_dict
        self.mask_dict = mask_dict
    
    def __len__(self):
        """Dataset length."""
        return len(self.X)
    
    def __getitem__(self, idx):
        """Get item by index.
        
        Args:
            idx: Index
            
        Returns:
            Tuple of (features, targets_dict, mask_dict)
        """
        return self.X[idx], self.y_dict, self.mask_dict
