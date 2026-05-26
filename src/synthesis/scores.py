"""Synthetic accessibility and complexity scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class SynthesisProfile:
    """Serializable synthetic feasibility profile."""

    smiles: str
    success: bool
    scores: dict[str, float]
    drivers: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SyntheticAccessibilityScorer:
    """Heuristic SA/SCScore-style triage.

    The SA score is a bounded approximation for local ranking. Replace this
    class with the RDKit contrib SA scorer or a trained SCScore model when
    calibrated production scores are required.
    """

    def score(self, smiles: str) -> SynthesisProfile:
        try:
            from rdkit import Chem
            from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
        except ImportError as exc:
            return SynthesisProfile(
                smiles=smiles,
                success=False,
                scores={},
                drivers=[],
                error=str(exc),
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return SynthesisProfile(
                smiles=smiles,
                success=False,
                scores={},
                drivers=["invalid_smiles"],
                error="Invalid SMILES",
            )

        metrics = {
            "mol_wt": float(Descriptors.MolWt(mol)),
            "bertz_ct": float(Descriptors.BertzCT(mol)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            "stereo_centers": float(len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))),
            "spiro_atoms": float(rdMolDescriptors.CalcNumSpiroAtoms(mol)),
            "bridgehead_atoms": float(rdMolDescriptors.CalcNumBridgeheadAtoms(mol)),
            "rings": float(rdMolDescriptors.CalcNumRings(mol)),
            "aromatic_rings": float(Lipinski.NumAromaticRings(mol)),
            "hetero_atoms": float(rdMolDescriptors.CalcNumHeteroatoms(mol)),
            "heavy_atoms": float(mol.GetNumHeavyAtoms()),
        }

        drivers = self._drivers(metrics)
        sa_score = self._sa_score(metrics)
        sc_score = self._scscore_proxy(metrics, sa_score)

        return SynthesisProfile(
            smiles=smiles,
            success=True,
            scores={
                "sa_score_proxy": round(sa_score, 3),
                "scscore_proxy": round(sc_score, 3),
                "synthetic_feasibility": round(max(0.0, min(1.0, (10.0 - sa_score) / 9.0)), 3),
                **metrics,
            },
            drivers=drivers,
        )

    @staticmethod
    def _sa_score(metrics: dict[str, float]) -> float:
        complexity = 1.0
        complexity += min(metrics["bertz_ct"] / 400.0, 3.0)
        complexity += min(metrics["stereo_centers"] * 0.35, 1.5)
        complexity += min((metrics["spiro_atoms"] + metrics["bridgehead_atoms"]) * 0.4, 1.5)
        complexity += min(max(0.0, metrics["rings"] - 2.0) * 0.25, 1.2)
        complexity += min(max(0.0, metrics["rotatable_bonds"] - 6.0) * 0.15, 1.0)
        complexity += min(max(0.0, metrics["mol_wt"] - 450.0) / 150.0, 1.0)
        hetero_balance = abs(metrics["hetero_atoms"] - metrics["heavy_atoms"] * 0.25)
        complexity += min(hetero_balance / 12.0, 0.8)
        return max(1.0, min(10.0, complexity))

    @staticmethod
    def _scscore_proxy(metrics: dict[str, float], sa_score: float) -> float:
        score = 1.0 + (sa_score - 1.0) * (4.0 / 9.0)
        score += min(metrics["stereo_centers"] * 0.08, 0.4)
        score += min(metrics["bridgehead_atoms"] * 0.1, 0.3)
        return max(1.0, min(5.0, score))

    @staticmethod
    def _drivers(metrics: dict[str, float]) -> list[str]:
        drivers: list[str] = []
        if metrics["bertz_ct"] > 700:
            drivers.append("high_graph_complexity")
        if metrics["stereo_centers"] >= 2:
            drivers.append("multiple_stereocenters")
        if metrics["spiro_atoms"] + metrics["bridgehead_atoms"] > 0:
            drivers.append("spiro_or_bridgehead_atoms")
        if metrics["rotatable_bonds"] > 8:
            drivers.append("many_rotatable_bonds")
        if metrics["rings"] > 4:
            drivers.append("many_rings")
        if not drivers:
            drivers.append("straightforward_rule_based_profile")
        return drivers
