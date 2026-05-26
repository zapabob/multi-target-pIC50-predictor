"""Image-derived molecular features for multimodal models."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class MultimodalFeatureBundle:
    """Serializable image feature bundle."""

    smiles: str
    success: bool
    image_features: list[float]
    metadata: dict[str, Any]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MolecularImageFeaturizer:
    """Render a molecule and convert the image to compact numeric features."""

    def __init__(self, image_size: int = 224, feature_grid: int = 32):
        self.image_size = image_size
        self.feature_grid = feature_grid

    def featurize(self, smiles: str) -> MultimodalFeatureBundle:
        try:
            import numpy as np
            from rdkit import Chem
            from rdkit.Chem import Draw
        except ImportError as exc:
            return MultimodalFeatureBundle(
                smiles=smiles,
                success=False,
                image_features=[],
                metadata={},
                error=str(exc),
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return MultimodalFeatureBundle(
                smiles=smiles,
                success=False,
                image_features=[],
                metadata={},
                error="Invalid SMILES",
            )

        try:
            image = Draw.MolToImage(mol, size=(self.image_size, self.image_size))
            image = image.convert("L").resize((self.feature_grid, self.feature_grid))
            array = np.asarray(image, dtype=np.float32) / 255.0
            hist, _ = np.histogram(array, bins=16, range=(0.0, 1.0), density=True)
            stats = np.array(
                [
                    array.mean(),
                    array.std(),
                    array.min(),
                    array.max(),
                    np.quantile(array, 0.25),
                    np.quantile(array, 0.50),
                    np.quantile(array, 0.75),
                ],
                dtype=np.float32,
            )
            features = np.concatenate([stats, hist.astype(np.float32), array.flatten()])
            return MultimodalFeatureBundle(
                smiles=smiles,
                success=True,
                image_features=features.astype(float).tolist(),
                metadata={
                    "image_size": self.image_size,
                    "feature_grid": self.feature_grid,
                    "feature_dim": int(features.shape[0]),
                    "mode": "grayscale_render",
                },
            )
        except Exception as exc:
            return MultimodalFeatureBundle(
                smiles=smiles,
                success=False,
                image_features=[],
                metadata={},
                error=str(exc),
            )
