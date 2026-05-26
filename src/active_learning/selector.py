"""
Active Learning selection strategies for molecular pIC50 prediction.
Optimized for RTX3060 and efficient compound discovery.
"""

import logging

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class ActiveLearningSelector:
    """Active Learning selector for choosing next compounds to synthesize/test.

    Implements multiple selection strategies:
    - Uncertainty-based: Select compounds with highest prediction uncertainty
    - Diversity-based: Select diverse compounds from chemical space
    - Query-by-Committee: Select compounds where ensemble models disagree
    - Hybrid: Combination of uncertainty and diversity
    """

    def __init__(
        self, strategy: str = "uncertainty", batch_size: int = 10, diversity_weight: float = 0.3
    ):
        """Initialize Active Learning selector.

        Args:
            strategy: Selection strategy ('uncertainty', 'diversity', 'qbc', 'hybrid')
            batch_size: Number of compounds to select
            diversity_weight: Weight for diversity in hybrid strategy (0-1)
        """
        self.strategy = strategy
        self.batch_size = batch_size
        self.diversity_weight = diversity_weight
        self.logger = logging.getLogger(__name__)

        self.logger.info(
            f"Active Learning selector initialized: strategy={strategy}, batch_size={batch_size}"
        )

    def select_uncertainty(
        self, uncertainties: np.ndarray, n_select: int | None = None
    ) -> np.ndarray:
        """Select compounds with highest uncertainty.

        Args:
            uncertainties: Uncertainty scores [n_compounds]
            n_select: Number of compounds to select (default: batch_size)

        Returns:
            Indices of selected compounds
        """
        if n_select is None:
            n_select = self.batch_size

        # Select top uncertain compounds
        indices = np.argsort(uncertainties)[::-1][:n_select]

        self.logger.info(
            f"Selected {len(indices)} compounds by uncertainty (mean={uncertainties[indices].mean():.4f})"
        )
        return indices

    def select_diversity(
        self,
        features: np.ndarray,
        n_select: int | None = None,
        already_selected: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select diverse compounds using K-Means clustering.

        Args:
            features: Feature matrix [n_compounds, n_features]
            n_select: Number of compounds to select
            already_selected: Indices of already selected compounds

        Returns:
            Indices of selected compounds
        """
        if n_select is None:
            n_select = self.batch_size

        # If already selected compounds exist, remove them
        available_indices = np.arange(len(features))
        if already_selected is not None and len(already_selected) > 0:
            mask = np.ones(len(features), dtype=bool)
            mask[already_selected] = False
            available_indices = available_indices[mask]
            features_available = features[mask]
        else:
            features_available = features

        if len(available_indices) <= n_select:
            return available_indices

        # Use K-Means for diversity
        kmeans = KMeans(n_clusters=n_select, random_state=42, n_init=10)
        kmeans.fit(features_available)

        # Select compounds closest to cluster centers
        distances = cdist(features_available, kmeans.cluster_centers_)
        selected_local = np.argmin(distances, axis=0)
        selected_global = available_indices[selected_local]

        self.logger.info(f"Selected {len(selected_global)} diverse compounds using K-Means")
        return selected_global

    def select_qbc(
        self, ensemble_predictions: np.ndarray, n_select: int | None = None
    ) -> np.ndarray:
        """Select compounds using Query-by-Committee (ensemble disagreement).

        Args:
            ensemble_predictions: Predictions from ensemble [n_models, n_compounds]
            n_select: Number of compounds to select

        Returns:
            Indices of selected compounds
        """
        if n_select is None:
            n_select = self.batch_size

        # Calculate disagreement (standard deviation across models)
        disagreement = np.std(ensemble_predictions, axis=0)

        # Select compounds with highest disagreement
        indices = np.argsort(disagreement)[::-1][:n_select]

        self.logger.info(
            f"Selected {len(indices)} compounds by QbC (mean disagreement={disagreement[indices].mean():.4f})"
        )
        return indices

    def select_hybrid(
        self,
        uncertainties: np.ndarray,
        features: np.ndarray,
        n_select: int | None = None,
        already_selected: np.ndarray | None = None,
    ) -> np.ndarray:
        """Select compounds using hybrid uncertainty + diversity strategy.

        Args:
            uncertainties: Uncertainty scores [n_compounds]
            features: Feature matrix [n_compounds, n_features]
            n_select: Number of compounds to select
            already_selected: Indices of already selected compounds

        Returns:
            Indices of selected compounds
        """
        if n_select is None:
            n_select = self.batch_size

        # Normalize uncertainty scores
        uncertainties_norm = (uncertainties - uncertainties.min()) / (
            uncertainties.max() - uncertainties.min() + 1e-10
        )

        # Select candidates with top uncertainty
        n_candidates = min(n_select * 10, len(uncertainties))  # 10x oversampling
        candidate_indices = np.argsort(uncertainties)[::-1][:n_candidates]

        # Among candidates, select diverse compounds
        candidate_features = features[candidate_indices]
        candidate_uncertainties = uncertainties_norm[candidate_indices]

        # Greedy selection with uncertainty + diversity
        selected_local = []
        remaining = list(range(len(candidate_indices)))

        # Start with most uncertain
        first_idx = 0
        selected_local.append(first_idx)
        remaining.remove(first_idx)

        # Iteratively select compounds balancing uncertainty and diversity
        while len(selected_local) < n_select and len(remaining) > 0:
            best_score = -np.inf
            best_idx = None

            for idx in remaining:
                # Uncertainty score
                uncertainty_score = candidate_uncertainties[idx]

                # Diversity score (min distance to selected)
                if len(selected_local) > 0:
                    distances = pairwise_distances(
                        candidate_features[idx : idx + 1], candidate_features[selected_local]
                    )
                    diversity_score = distances.min()
                    # Normalize diversity
                    diversity_score = diversity_score / (distances.max() + 1e-10)
                else:
                    diversity_score = 1.0

                # Combined score
                score = (
                    1 - self.diversity_weight
                ) * uncertainty_score + self.diversity_weight * diversity_score

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected_local.append(best_idx)
                remaining.remove(best_idx)

        selected_global = candidate_indices[selected_local]

        self.logger.info(
            f"Selected {len(selected_global)} compounds using hybrid strategy (uncertainty_weight={1 - self.diversity_weight}, diversity_weight={self.diversity_weight})"
        )
        return selected_global

    def select(
        self,
        uncertainties: np.ndarray | None = None,
        features: np.ndarray | None = None,
        ensemble_predictions: np.ndarray | None = None,
        already_selected: np.ndarray | None = None,
        n_select: int | None = None,
    ) -> np.ndarray:
        """Select compounds using configured strategy.

        Args:
            uncertainties: Uncertainty scores (required for 'uncertainty' and 'hybrid')
            features: Feature matrix (required for 'diversity' and 'hybrid')
            ensemble_predictions: Ensemble predictions (required for 'qbc')
            already_selected: Already selected compound indices
            n_select: Number to select

        Returns:
            Indices of selected compounds
        """
        if self.strategy == "uncertainty":
            if uncertainties is None:
                raise ValueError("uncertainties required for uncertainty-based selection")
            return self.select_uncertainty(uncertainties, n_select)

        elif self.strategy == "diversity":
            if features is None:
                raise ValueError("features required for diversity-based selection")
            return self.select_diversity(features, n_select, already_selected)

        elif self.strategy == "qbc":
            if ensemble_predictions is None:
                raise ValueError("ensemble_predictions required for QbC selection")
            return self.select_qbc(ensemble_predictions, n_select)

        elif self.strategy == "hybrid":
            if uncertainties is None or features is None:
                raise ValueError("uncertainties and features required for hybrid selection")
            return self.select_hybrid(uncertainties, features, n_select, already_selected)

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def rank_compounds(
        self,
        uncertainties: np.ndarray | None = None,
        features: np.ndarray | None = None,
        ensemble_predictions: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Rank all compounds by selection priority.

        Args:
            uncertainties: Uncertainty scores
            features: Feature matrix
            ensemble_predictions: Ensemble predictions

        Returns:
            Tuple of (ranked_indices, scores)
        """
        if self.strategy == "uncertainty":
            if uncertainties is None:
                raise ValueError("uncertainties required")
            scores = uncertainties

        elif self.strategy == "diversity":
            raise NotImplementedError(
                "Diversity ranking not implemented (use select_diversity instead)"
            )

        elif self.strategy == "qbc":
            if ensemble_predictions is None:
                raise ValueError("ensemble_predictions required")
            scores = np.std(ensemble_predictions, axis=0)

        elif self.strategy == "hybrid":
            if uncertainties is None:
                raise ValueError("uncertainties required")
            # Normalize uncertainties as scores
            scores = (uncertainties - uncertainties.min()) / (
                uncertainties.max() - uncertainties.min() + 1e-10
            )

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        ranked_indices = np.argsort(scores)[::-1]
        return ranked_indices, scores[ranked_indices]

    def get_selection_report(
        self,
        selected_indices: np.ndarray,
        smiles_list: list[str],
        uncertainties: np.ndarray | None = None,
        predictions: np.ndarray | None = None,
    ) -> str:
        """Generate a report for selected compounds.

        Args:
            selected_indices: Indices of selected compounds
            smiles_list: List of SMILES strings
            uncertainties: Uncertainty scores
            predictions: pIC50 predictions

        Returns:
            Formatted report string
        """
        report = "\n=== Active Learning Selection Report ===\n"
        report += f"Strategy: {self.strategy}\n"
        report += f"Number of compounds selected: {len(selected_indices)}\n\n"

        for i, idx in enumerate(selected_indices):
            report += f"{i + 1}. SMILES: {smiles_list[idx]}\n"
            if predictions is not None:
                report += f"   Predicted pIC50: {predictions[idx]:.2f}\n"
            if uncertainties is not None:
                report += f"   Uncertainty: {uncertainties[idx]:.4f}\n"
            report += "\n"

        return report
