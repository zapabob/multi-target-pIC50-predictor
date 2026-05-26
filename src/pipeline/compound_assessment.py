"""Unified compound assessment pipeline for discovery triage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..admet import ADMETPredictor
from ..features.structure3d import ETKDGConformerGenerator
from ..multimodal import MolecularImageFeaturizer
from ..reactions import RetrosynthesisPlanner
from ..synthesis import SyntheticAccessibilityScorer


@dataclass
class CompoundAssessmentResult:
    """Serializable compound assessment output."""

    smiles: str
    target: str | None
    pIC50_prediction: float | None
    uncertainty: float | None
    admet: dict[str, Any]
    synthesis: dict[str, Any]
    structure3d: dict[str, Any] | None
    retrosynthesis: list[dict[str, Any]] | None
    multimodal: dict[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CompoundAssessmentPipeline:
    """Combine pIC50, 3D, ADMET, synthesis, reaction, and image triage."""

    def __init__(
        self,
        predictor: Any | None = None,
        target: str | None = None,
        include_coordinates: bool = False,
    ):
        self.predictor = predictor
        self.target = target
        self.admet_predictor = ADMETPredictor()
        self.synthesis_scorer = SyntheticAccessibilityScorer()
        self.conformer_generator = ETKDGConformerGenerator(include_coordinates=include_coordinates)
        self.retrosynthesis_planner = RetrosynthesisPlanner()
        self.image_featurizer = MolecularImageFeaturizer()

    def assess(
        self,
        smiles: str,
        include_3d: bool = True,
        include_reactions: bool = True,
        include_image: bool = False,
    ) -> CompoundAssessmentResult:
        prediction, uncertainty = self._predict_pic50(smiles)
        admet = self.admet_predictor.predict(smiles).to_dict()
        synthesis = self.synthesis_scorer.score(smiles).to_dict()
        structure3d = self.conformer_generator.generate(smiles).to_dict() if include_3d else None
        retrosynthesis = (
            [route.to_dict() for route in self.retrosynthesis_planner.plan(smiles)]
            if include_reactions
            else None
        )
        multimodal = self.image_featurizer.featurize(smiles).to_dict() if include_image else None

        return CompoundAssessmentResult(
            smiles=smiles,
            target=self.target,
            pIC50_prediction=prediction,
            uncertainty=uncertainty,
            admet=admet,
            synthesis=synthesis,
            structure3d=structure3d,
            retrosynthesis=retrosynthesis,
            multimodal=multimodal,
        )

    def assess_batch(
        self,
        smiles_list: list[str],
        include_3d: bool = True,
        include_reactions: bool = True,
        include_image: bool = False,
    ) -> list[CompoundAssessmentResult]:
        return [
            self.assess(
                smiles,
                include_3d=include_3d,
                include_reactions=include_reactions,
                include_image=include_image,
            )
            for smiles in smiles_list
        ]

    def _predict_pic50(self, smiles: str) -> tuple[float | None, float | None]:
        if self.predictor is None:
            return None, None
        try:
            output = self.predictor.predict(smiles)
        except Exception:
            return None, None

        if isinstance(output, tuple):
            prediction = output[0]
            confidence = output[1] if len(output) > 1 else None
            uncertainty = None
            if isinstance(confidence, dict):
                uncertainty = confidence.get("std")
            return (
                float(prediction) if prediction is not None else None,
                float(uncertainty) if uncertainty is not None else None,
            )
        return float(output), None
