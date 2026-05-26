"""
Ensemble learning manager for molecular pIC50 prediction.
Combines Transformer, GNN, XGBoost, and Random Forest models.
"""

import logging

import numpy as np
import torch
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


class EnsembleManager:
    """Ensemble learning manager for combining multiple models.

    Supports:
    - Weighted averaging
    - Stacking (meta-learner)
    - Voting (for classification)
    - Uncertainty integration
    """

    def __init__(
        self,
        models: dict[str, torch.nn.Module] | None = None,
        method: str = "weighted_average",
        use_uncertainty: bool = True,
    ):
        """Initialize ensemble manager.

        Args:
            models: Dictionary of trained models
            method: Ensemble method ('weighted_average', 'stacking', 'voting')
            use_uncertainty: Whether to use uncertainty for weighting
        """
        self.models = models or {}
        self.method = method
        self.use_uncertainty = use_uncertainty
        self.logger = logging.getLogger(__name__)

        # Meta-learner for stacking
        self.meta_learner = None
        self.is_fitted = False

        # Model weights (for weighted averaging)
        self.model_weights = {}

        self.logger.info(
            f"Ensemble manager initialized: method={method}, models={list(self.models.keys())}"
        )

    def add_model(self, name: str, model: torch.nn.Module) -> None:
        """Add a model to the ensemble.

        Args:
            name: Model name
            model: Trained model
        """
        self.models[name] = model
        self.logger.info(f"Added model '{name}' to ensemble")

    def remove_model(self, name: str) -> None:
        """Remove a model from the ensemble.

        Args:
            name: Model name
        """
        if name in self.models:
            del self.models[name]
            self.logger.info(f"Removed model '{name}' from ensemble")

    def predict_single_model(
        self,
        model_name: str,
        X: torch.Tensor | list,
        device: torch.device,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Get predictions from a single model.

        Args:
            model_name: Name of the model
            X: Input features (tensor or graph data list)
            device: Device to run on
            batch_size: Batch size

        Returns:
            Predictions array
        """
        model = self.models[model_name]
        model.eval()

        with torch.no_grad():
            if model_name.startswith("gnn"):
                # Graph model
                from torch_geometric.data import Batch

                if isinstance(X, list):
                    batch = Batch.from_data_list(X).to(device)
                    predictions = model(batch).cpu().numpy().flatten()
                else:
                    raise ValueError("GNN models require graph data list")
            else:
                # Standard model (Transformer, etc.)
                if isinstance(X, torch.Tensor):
                    X = X.to(device)
                else:
                    X = torch.tensor(X, dtype=torch.float32).to(device)

                batch_preds = []
                for i in range(0, len(X), batch_size):
                    batch = X[i : i + batch_size]
                    pred = model(batch)
                    batch_preds.append(pred.cpu().numpy())

                predictions = np.concatenate(batch_preds, axis=0).flatten()

        return predictions

    def predict_all_models(
        self, X: torch.Tensor | list, device: torch.device, batch_size: int = 32
    ) -> dict[str, np.ndarray]:
        """Get predictions from all models.

        Args:
            X: Input features
            device: Device to run on
            batch_size: Batch size

        Returns:
            Dictionary of model predictions
        """
        predictions = {}

        for name in self.models:
            try:
                pred = self.predict_single_model(name, X, device, batch_size)
                predictions[name] = pred
            except Exception as e:
                self.logger.error(f"Error predicting with model '{name}': {e}")
                continue

        return predictions

    def fit_weighted_average(
        self,
        X_val: torch.Tensor | list,
        y_val: np.ndarray,
        device: torch.device,
        batch_size: int = 32,
    ) -> None:
        """Fit weights for weighted averaging based on validation performance.

        Args:
            X_val: Validation features
            y_val: Validation targets
            device: Device to run on
            batch_size: Batch size
        """
        predictions = self.predict_all_models(X_val, device, batch_size)

        if not predictions:
            raise ValueError("No valid predictions for weight fitting")

        # Calculate R² scores for each model
        scores = {}
        for name, pred in predictions.items():
            try:
                r2 = r2_score(y_val, pred)
                scores[name] = max(0, r2)  # Ensure non-negative weights
            except Exception as e:
                self.logger.warning(f"Error calculating R² for model '{name}': {e}")
                scores[name] = 0.0

        # Normalize weights
        total_score = sum(scores.values())
        if total_score > 0:
            self.model_weights = {name: score / total_score for name, score in scores.items()}
        else:
            # Equal weights if all models perform poorly
            n_models = len(scores)
            self.model_weights = {name: 1.0 / n_models for name in scores.keys()}

        self.logger.info(f"Model weights: {self.model_weights}")
        self.is_fitted = True

    def fit_stacking(
        self,
        X_train: torch.Tensor | list,
        y_train: np.ndarray,
        X_val: torch.Tensor | list,
        y_val: np.ndarray,
        device: torch.device,
        batch_size: int = 32,
    ) -> None:
        """Fit stacking meta-learner.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            device: Device to run on
            batch_size: Batch size
        """
        # Get base model predictions on training set
        train_predictions = self.predict_all_models(X_train, device, batch_size)

        # Get base model predictions on validation set
        val_predictions = self.predict_all_models(X_val, device, batch_size)

        if not train_predictions or not val_predictions:
            raise ValueError("No valid predictions for stacking")

        # Prepare meta-features
        train_meta = np.column_stack(
            [train_predictions[name] for name in sorted(train_predictions.keys())]
        )
        val_meta = np.column_stack(
            [val_predictions[name] for name in sorted(val_predictions.keys())]
        )

        # Fit meta-learner
        self.meta_learner = LinearRegression()
        self.meta_learner.fit(train_meta, y_train)

        # Evaluate meta-learner
        val_pred = self.meta_learner.predict(val_meta)
        val_r2 = r2_score(y_val, val_pred)

        self.logger.info(f"Stacking meta-learner fitted. Validation R²: {val_r2:.4f}")
        self.is_fitted = True

    def predict_weighted_average(
        self, X: torch.Tensor | list, device: torch.device, batch_size: int = 32
    ) -> np.ndarray:
        """Predict using weighted averaging.

        Args:
            X: Input features
            device: Device to run on
            batch_size: Batch size

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit_weighted_average() first.")

        predictions = self.predict_all_models(X, device, batch_size)

        if not predictions:
            raise ValueError("No valid predictions for ensemble")

        # Weighted average
        ensemble_pred = np.zeros(len(next(iter(predictions.values()))))

        for name, pred in predictions.items():
            weight = self.model_weights.get(name, 0.0)
            ensemble_pred += weight * pred

        return ensemble_pred

    def predict_stacking(
        self, X: torch.Tensor | list, device: torch.device, batch_size: int = 32
    ) -> np.ndarray:
        """Predict using stacking.

        Args:
            X: Input features
            device: Device to run on
            batch_size: Batch size

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted or self.meta_learner is None:
            raise ValueError("Ensemble not fitted. Call fit_stacking() first.")

        predictions = self.predict_all_models(X, device, batch_size)

        if not predictions:
            raise ValueError("No valid predictions for ensemble")

        # Prepare meta-features
        meta_features = np.column_stack([predictions[name] for name in sorted(predictions.keys())])

        # Meta-learner prediction
        ensemble_pred = self.meta_learner.predict(meta_features)

        return ensemble_pred

    def predict(
        self, X: torch.Tensor | list, device: torch.device, batch_size: int = 32
    ) -> np.ndarray:
        """Make ensemble predictions.

        Args:
            X: Input features
            device: Device to run on
            batch_size: Batch size

        Returns:
            Ensemble predictions
        """
        if self.method == "weighted_average":
            return self.predict_weighted_average(X, device, batch_size)
        elif self.method == "stacking":
            return self.predict_stacking(X, device, batch_size)
        else:
            raise ValueError(f"Unknown ensemble method: {self.method}")

    def predict_with_uncertainty(
        self,
        X: torch.Tensor | list,
        device: torch.device,
        batch_size: int = 32,
        uncertainty_method: str = "ensemble_variance",
    ) -> dict[str, np.ndarray]:
        """Make ensemble predictions with uncertainty.

        Args:
            X: Input features
            device: Device to run on
            batch_size: Batch size
            uncertainty_method: Uncertainty method ('ensemble_variance', 'individual_uncertainty')

        Returns:
            Dictionary with predictions and uncertainty
        """
        if uncertainty_method == "ensemble_variance":
            # Use variance across ensemble models
            predictions = self.predict_all_models(X, device, batch_size)

            if not predictions:
                raise ValueError("No valid predictions for uncertainty estimation")

            # Calculate ensemble statistics
            pred_matrix = np.column_stack(list(predictions.values()))
            mean_pred = np.mean(pred_matrix, axis=1)
            std_pred = np.std(pred_matrix, axis=1)

            return {
                "mean": mean_pred,
                "std": std_pred,
                "lower": mean_pred - 1.96 * std_pred,  # 95% CI
                "upper": mean_pred + 1.96 * std_pred,
                "individual_predictions": predictions,
            }

        elif uncertainty_method == "individual_uncertainty":
            # Use individual model uncertainties (if available)
            # This would require each model to support uncertainty estimation
            raise NotImplementedError("Individual uncertainty method not implemented yet")

        else:
            raise ValueError(f"Unknown uncertainty method: {uncertainty_method}")

    def evaluate(
        self,
        X: torch.Tensor | list,
        y: np.ndarray,
        device: torch.device,
        batch_size: int = 32,
    ) -> dict[str, float]:
        """Evaluate ensemble performance.

        Args:
            X: Input features
            y: True targets
            device: Device to run on
            batch_size: Batch size

        Returns:
            Dictionary with evaluation metrics
        """
        predictions = self.predict(X, device, batch_size)

        r2 = r2_score(y, predictions)
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)

        return {"r2": r2, "mse": mse, "rmse": rmse}

    def get_model_importance(self) -> dict[str, float]:
        """Get model importance scores.

        Returns:
            Dictionary with model importance scores
        """
        if self.method == "weighted_average":
            return self.model_weights.copy()
        elif self.method == "stacking" and self.meta_learner is not None:
            # Use meta-learner coefficients as importance
            feature_names = sorted(self.models.keys())
            importances = dict(zip(feature_names, self.meta_learner.coef_, strict=False))
            return importances
        else:
            return {name: 1.0 / len(self.models) for name in self.models.keys()}

    def save_ensemble(self, filepath: str) -> None:
        """Save ensemble configuration.

        Args:
            filepath: Path to save ensemble
        """
        import pickle

        ensemble_data = {
            "method": self.method,
            "use_uncertainty": self.use_uncertainty,
            "model_weights": self.model_weights,
            "is_fitted": self.is_fitted,
            "model_names": list(self.models.keys()),
        }

        with open(filepath, "wb") as f:
            pickle.dump(ensemble_data, f)

        self.logger.info(f"Ensemble configuration saved to {filepath}")

    def load_ensemble(self, filepath: str) -> None:
        """Load ensemble configuration.

        Args:
            filepath: Path to load ensemble from
        """
        import pickle

        with open(filepath, "rb") as f:
            ensemble_data = pickle.load(f)

        self.method = ensemble_data["method"]
        self.use_uncertainty = ensemble_data["use_uncertainty"]
        self.model_weights = ensemble_data["model_weights"]
        self.is_fitted = ensemble_data["is_fitted"]

        self.logger.info(f"Ensemble configuration loaded from {filepath}")


class XGBoostWrapper:
    """Wrapper for XGBoost model in ensemble."""

    def __init__(self, **kwargs):
        """Initialize XGBoost wrapper.

        Args:
            **kwargs: XGBoost parameters
        """
        self.model = xgb.XGBRegressor(**kwargs)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit XGBoost model.

        Args:
            X: Training features
            y: Training targets
        """
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)


class RandomForestWrapper:
    """Wrapper for Random Forest model in ensemble."""

    def __init__(self, **kwargs):
        """Initialize Random Forest wrapper.

        Args:
            **kwargs: Random Forest parameters
        """
        self.model = RandomForestRegressor(**kwargs)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit Random Forest model.

        Args:
            X: Training features
            y: Training targets
        """
        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Args:
            X: Input features

        Returns:
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        return self.model.predict(X)
