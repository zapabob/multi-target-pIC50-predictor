"""Retrosynthesis and forward reaction planning interfaces."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ReactionStep:
    """One reaction or disconnection step."""

    name: str
    reactants: list[str]
    products: list[str]
    confidence: float
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReactionRoute:
    """Serializable reaction route."""

    target_smiles: str
    steps: list[ReactionStep]
    score: float
    backend: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_smiles": self.target_smiles,
            "steps": [step.to_dict() for step in self.steps],
            "score": self.score,
            "backend": self.backend,
            "error": self.error,
        }


class RetrosynthesisPlanner:
    """Template-based baseline retrosynthesis planner.

    This is intentionally conservative. It creates explainable baseline routes
    and defines a stable interface for AiZynthFinder, ASKCOS, IBM RXN, or other
    production retrosynthesis backends.
    """

    RETRO_TEMPLATES = {
        "amide_disconnection": "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]",
        "ester_disconnection": "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])O.[O:3][C:4]",
        "aryl_ether_disconnection": "[c:1][O:2][C:3]>>[c:1][O:2].[C:3][Br]",
    }

    def __init__(self, max_routes: int = 5):
        self.max_routes = max_routes

    def plan(self, target_smiles: str) -> list[ReactionRoute]:
        try:
            from rdkit import Chem
        except ImportError as exc:
            return [
                ReactionRoute(
                    target_smiles=target_smiles,
                    steps=[],
                    score=0.0,
                    backend="template",
                    error=f"RDKit is required for template retrosynthesis: {exc}",
                )
            ]

        mol = Chem.MolFromSmiles(target_smiles)
        if mol is None:
            return [
                ReactionRoute(
                    target_smiles=target_smiles,
                    steps=[],
                    score=0.0,
                    backend="template",
                    error="Invalid SMILES",
                )
            ]

        routes: list[ReactionRoute] = []
        for template_name, reaction_smarts in self.RETRO_TEMPLATES.items():
            product_sets = _apply_reaction(target_smiles, reaction_smarts)
            for precursors in product_sets:
                step = ReactionStep(
                    name=template_name,
                    reactants=list(precursors),
                    products=[target_smiles],
                    confidence=self._confidence(template_name, precursors),
                    notes="Template baseline; validate with a retrosynthesis engine and chemist review.",
                )
                routes.append(
                    ReactionRoute(
                        target_smiles=target_smiles,
                        steps=[step],
                        score=step.confidence,
                        backend="template",
                    )
                )

        if not routes:
            routes.append(
                ReactionRoute(
                    target_smiles=target_smiles,
                    steps=[],
                    score=0.1,
                    backend="template",
                    error="No simple template route found",
                )
            )
        return sorted(routes, key=lambda route: route.score, reverse=True)[: self.max_routes]

    @staticmethod
    def _confidence(template_name: str, precursors: Iterable[str]) -> float:
        base = {
            "amide_disconnection": 0.70,
            "ester_disconnection": 0.65,
            "aryl_ether_disconnection": 0.45,
        }.get(template_name, 0.3)
        precursor_penalty = max(0, len(list(precursors)) - 2) * 0.05
        return round(max(0.05, min(0.95, base - precursor_penalty)), 3)


class ForwardReactionPredictor:
    """Template-based forward reaction predictor."""

    FORWARD_TEMPLATES = {
        "amide_coupling": "[C:1](=[O:2])O.[N:3]>>[C:1](=[O:2])[N:3]",
        "esterification": "[C:1](=[O:2])O.[O:3][C:4]>>[C:1](=[O:2])[O:3][C:4]",
        "aryl_ether_formation": "[c:1][O:2].[C:3][Br]>>[c:1][O:2][C:3]",
    }

    def predict(
        self, reactant_smiles: list[str], template: str | None = None
    ) -> list[ReactionStep]:
        reactants = ".".join(reactant_smiles)
        templates = (
            {template: self.FORWARD_TEMPLATES[template]} if template else self.FORWARD_TEMPLATES
        )

        steps: list[ReactionStep] = []
        for name, reaction_smarts in templates.items():
            product_sets = _apply_reaction(reactants, reaction_smarts)
            for products in product_sets:
                steps.append(
                    ReactionStep(
                        name=name,
                        reactants=reactant_smiles,
                        products=list(products),
                        confidence=0.5,
                        notes="Template forward prediction; rank with an external reaction model for production.",
                    )
                )
        return steps


def _apply_reaction(input_smiles: str, reaction_smarts: str) -> list[tuple[str, ...]]:
    """Apply an RDKit reaction and return canonical product tuples."""
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError:
        return []

    reactants = [Chem.MolFromSmiles(part) for part in input_smiles.split(".")]
    if any(mol is None for mol in reactants):
        return []

    try:
        reaction = AllChem.ReactionFromSmarts(reaction_smarts)
        product_sets = reaction.RunReactants(tuple(reactants))
    except Exception:
        return []

    canonical: set[tuple[str, ...]] = set()
    for products in product_sets:
        smiles_products: list[str] = []
        for product in products:
            canonical_smiles = _canonical_product_smiles(product, Chem)
            if canonical_smiles is not None:
                smiles_products.append(canonical_smiles)
        if smiles_products:
            canonical.add(tuple(sorted(smiles_products)))
    return sorted(canonical)


def _canonical_product_smiles(product: Any, chem_module: Any) -> str | None:
    """Return canonical SMILES for valid RDKit products."""
    try:
        chem_module.SanitizeMol(product)
        return chem_module.MolToSmiles(product, canonical=True)
    except Exception:
        return None
