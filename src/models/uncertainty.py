"""
Uncertainty estimation for molecular pIC50 prediction.
Implements Monte Carlo Dropout and Deep Ensemble methods.
"""

import logging

import numpy as np
import torch
import torch.nn as nn


class MCDropoutPredictor:
    """Uncertainty estimation for pIC50 predictions.

    Supports:
    - Monte Carlo Dropout (lightweight, single model)
    - Deep Ensemble (multiple models, higher accuracy)
    """

    def __init__(
        self, method: str = "mc_dropout", n_samples: int = 30, confidence_level: float = 0.95
    ):
        """Initialize uncertainty estimator.

        Args:
            method: Uncertainty estimation method ('mc_dropout' or 'ensemble')
            n_samples: Number of MC Dropout samples or ensemble size
            confidence_level: Confidence level for prediction intervals
        """
        self.method = method
        self.n_samples = n_samples
        self.confidence_level = confidence_level
        self.logger = logging.getLogger(__name__)

        # Calculate quantiles for prediction intervals
        alpha = 1 - confidence_level
        self.lower_quantile = alpha / 2
        self.upper_quantile = 1 - alpha / 2

        self.logger.info(
            f"Uncertainty estimator initialized: method={method}, n_samples={n_samples}, CI={confidence_level}"
        )

    def enable_dropout(self, model: nn.Module) -> None:
        """Enable dropout layers for MC Dropout.

        Args:
            model: PyTorch model
        """
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
        self, model: nn.Module, X: torch.Tensor, device: torch.device, batch_size: int = 32
    ) -> dict[str, np.ndarray]:
        """Predict with uncertainty using Monte Carlo Dropout.

        Args:
            model: PyTorch model
            X: Input features
            device: Device to run on
            batch_size: Batch size for prediction

        Returns:
            Dictionary with prediction statistics
        """
        if self.method == "mc_dropout":
            return self._mc_dropout_predict(model, X, device, batch_size)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented for single model")

    def _mc_dropout_predict(
        self, model: nn.Module, X: torch.Tensor, device: torch.device, batch_size: int
    ) -> dict[str, np.ndarray]:
        """Monte Carlo Dropout prediction.

        Args:
            model: PyTorch model
            X: Input features
            device: Device to run on
            batch_size: Batch size

        Returns:
            Dictionary with prediction statistics
        """
        model.eval()
        self.enable_dropout(model)

        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                batch_preds = []

                for i in range(0, len(X), batch_size):
                    batch = X[i : i + batch_size].to(device)
                    pred = model(batch)
                    batch_preds.append(pred.cpu().numpy())

                predictions.append(np.concatenate(batch_preds, axis=0))

        predictions = np.array(predictions)  # [n_samples, n_data, 1]

        # Calculate statistics
        mean = np.mean(predictions, axis=0).flatten()
        std = np.std(predictions, axis=0).flatten()
        lower = np.quantile(predictions, self.lower_quantile, axis=0).flatten()
        upper = np.quantile(predictions, self.upper_quantile, axis=0).flatten()

        return {"mean": mean, "std": std, "lower": lower, "upper": upper, "samples": predictions}

    def predict_with_uncertainty_graph(
        self, model: nn.Module, data_list: list, device: torch.device, batch_size: int = 32
    ) -> dict[str, np.ndarray]:
        """Predict with uncertainty for graph data (GNN models).

        Args:
            model: PyTorch Geometric model
            data_list: List of PyTorch Geometric Data objects
            device: Device to run on
            batch_size: Batch size

        Returns:
            Dictionary with prediction statistics
        """

        if self.method == "mc_dropout":
            return self._mc_dropout_predict_graph(model, data_list, device, batch_size)
        else:
            raise NotImplementedError(f"Method {self.method} not implemented for graph models")

    def _mc_dropout_predict_graph(
        self, model: nn.Module, data_list: list, device: torch.device, batch_size: int
    ) -> dict[str, np.ndarray]:
        """Monte Carlo Dropout prediction for graph data.

        Args:
            model: PyTorch Geometric model
            data_list: List of Data objects
            device: Device
            batch_size: Batch size

        Returns:
            Dictionary with prediction statistics
        """
        from torch_geometric.data import Batch

        model.eval()
        self.enable_dropout(model)

        predictions = []

        with torch.no_grad():
            for _ in range(self.n_samples):
                batch_preds = []

                for i in range(0, len(data_list), batch_size):
                    batch_data = data_list[i : i + batch_size]
                    batch = Batch.from_data_list(batch_data).to(device)
                    pred = model(batch)
                    batch_preds.append(pred.cpu().numpy())

                predictions.append(np.concatenate(batch_preds, axis=0))

        predictions = np.array(predictions)  # [n_samples, n_data, 1]

        # Calculate statistics
        mean = np.mean(predictions, axis=0).flatten()
        std = np.std(predictions, axis=0).flatten()
        lower = np.quantile(predictions, self.lower_quantile, axis=0).flatten()
        upper = np.quantile(predictions, self.upper_quantile, axis=0).flatten()

        return {"mean": mean, "std": std, "lower": lower, "upper": upper, "samples": predictions}


class DeepEnsemblePredictor:
    """Deep Ensemble for uncertainty estimation.

    Trains multiple models with different initializations and aggregates predictions.
    """

    def __init__(self, n_models: int = 5, confidence_level: float = 0.95):
        """Initialize Deep Ensemble.

        Args:
            n_models: Number of ensemble models
            confidence_level: Confidence level for prediction intervals
        """
        self.n_models = n_models
        self.confidence_level = confidence_level
        self.models = []
        self.logger = logging.getLogger(__name__)

        # Calculate quantiles
        alpha = 1 - confidence_level
        self.lower_quantile = alpha / 2
        self.upper_quantile = 1 - alpha / 2

        self.logger.info(f"Deep Ensemble initialized: n_models={n_models}, CI={confidence_level}")

    def add_model(self, model: nn.Module) -> None:
        """Add a trained model to the ensemble.

        Args:
            model: Trained PyTorch model
        """
        self.models.append(model)
        self.logger.info(f"Model added to ensemble ({len(self.models)}/{self.n_models})")

    def predict_with_uncertainty(
        self, X: torch.Tensor, device: torch.device, batch_size: int = 32
    ) -> dict[str, np.ndarray]:
        """Predict with uncertainty using ensemble.

        Args:
            X: Input features
            device: Device to run on
            batch_size: Batch size

        Returns:
            Dictionary with prediction statistics
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models using add_model().")

        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                batch_preds = []
                for i in range(0, len(X), batch_size):
                    batch = X[i : i + batch_size].to(device)
                    pred = model(batch)
                    batch_preds.append(pred.cpu().numpy())
                predictions.append(np.concatenate(batch_preds, axis=0))

        predictions = np.array(predictions)  # [n_models, n_data, 1]

        # Calculate statistics
        mean = np.mean(predictions, axis=0).flatten()
        std = np.std(predictions, axis=0).flatten()
        lower = np.quantile(predictions, self.lower_quantile, axis=0).flatten()
        upper = np.quantile(predictions, self.upper_quantile, axis=0).flatten()

        return {"mean": mean, "std": std, "lower": lower, "upper": upper, "samples": predictions}

    def predict_with_uncertainty_graph(
        self, data_list: list, device: torch.device, batch_size: int = 32
    ) -> dict[str, np.ndarray]:
        """Predict with uncertainty for graph data.

        Args:
            data_list: List of PyTorch Geometric Data objects
            device: Device
            batch_size: Batch size

        Returns:
            Dictionary with prediction statistics
        """
        from torch_geometric.data import Batch

        if len(self.models) == 0:
            raise ValueError("No models in ensemble. Add models using add_model().")

        predictions = []

        for model in self.models:
            model.eval()
            with torch.no_grad():
                batch_preds = []
                for i in range(0, len(data_list), batch_size):
                    batch_data = data_list[i : i + batch_size]
                    batch = Batch.from_data_list(batch_data).to(device)
                    pred = model(batch)
                    batch_preds.append(pred.cpu().numpy())
                predictions.append(np.concatenate(batch_preds, axis=0))

        predictions = np.array(predictions)  # [n_models, n_data, 1]

        # Calculate statistics
        mean = np.mean(predictions, axis=0).flatten()
        std = np.std(predictions, axis=0).flatten()
        lower = np.quantile(predictions, self.lower_quantile, axis=0).flatten()
        upper = np.quantile(predictions, self.upper_quantile, axis=0).flatten()

        return {"mean": mean, "std": std, "lower": lower, "upper": upper, "samples": predictions}

    def calibrate(
        self, X_cal: torch.Tensor, y_cal: np.ndarray, device: torch.device, batch_size: int = 32
    ) -> dict[str, float]:
        """Calibrate uncertainty estimates on validation data.

        Args:
            X_cal: Calibration features
            y_cal: Calibration targets
            device: Device
            batch_size: Batch size

        Returns:
            Calibration metrics
        """
        predictions = self.predict_with_uncertainty(X_cal, device, batch_size)

        lower = predictions["lower"]
        upper = predictions["upper"]

        # Calculate coverage (how many true values fall within prediction intervals)
        in_interval = (y_cal >= lower) & (y_cal <= upper)
        coverage = np.mean(in_interval)

        # Calculate average interval width
        interval_width = np.mean(upper - lower)

        # Calculate calibration error
        calibration_error = abs(coverage - self.confidence_level)

        metrics = {
            "coverage": coverage,
            "target_coverage": self.confidence_level,
            "calibration_error": calibration_error,
            "avg_interval_width": interval_width,
        }

        self.logger.info(
            f"Calibration metrics: coverage={coverage:.3f}, target={self.confidence_level}, error={calibration_error:.3f}"
        )

        return metrics
