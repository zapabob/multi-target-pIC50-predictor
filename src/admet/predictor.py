"""Rule-based ADMET profiling for early discovery triage."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ADMETProfile:
    """Serializable ADMET profile."""

    smiles: str
    success: bool
    descriptors: dict[str, float]
    scores: dict[str, float]
    liabilities: list[str]
    recommendations: list[str]
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ADMETPredictor:
    """Lightweight ADMET estimator based on RDKit descriptors and rules."""

    def predict(self, smiles: str) -> ADMETProfile:
        try:
            from rdkit import Chem
            from rdkit.Chem import QED, Crippen, Descriptors, Lipinski, rdMolDescriptors
        except ImportError as exc:
            return ADMETProfile(
                smiles=smiles,
                success=False,
                descriptors={},
                scores={},
                liabilities=[],
                recommendations=["Install RDKit to enable ADMET descriptors."],
                error=str(exc),
            )

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ADMETProfile(
                smiles=smiles,
                success=False,
                descriptors={},
                scores={},
                liabilities=["invalid_smiles"],
                recommendations=["Check molecule syntax before ADMET triage."],
                error="Invalid SMILES",
            )

        descriptors = {
            "mol_wt": float(Descriptors.MolWt(mol)),
            "logp": float(Crippen.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "hbd": float(Lipinski.NumHDonors(mol)),
            "hba": float(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": float(Lipinski.NumRotatableBonds(mol)),
            "aromatic_rings": float(Lipinski.NumAromaticRings(mol)),
            "fraction_csp3": float(Descriptors.FractionCSP3(mol)),
            "heavy_atoms": float(mol.GetNumHeavyAtoms()),
            "formal_charge": float(sum(atom.GetFormalCharge() for atom in mol.GetAtoms())),
            "qed": float(QED.qed(mol)),
            "num_rings": float(rdMolDescriptors.CalcNumRings(mol)),
        }

        liabilities = self._liabilities(descriptors)
        scores = self._scores(descriptors, liabilities)
        recommendations = self._recommendations(descriptors, liabilities)

        return ADMETProfile(
            smiles=smiles,
            success=True,
            descriptors=descriptors,
            scores=scores,
            liabilities=liabilities,
            recommendations=recommendations,
        )

    @staticmethod
    def _liabilities(descriptors: dict[str, float]) -> list[str]:
        liabilities: list[str] = []
        if descriptors["mol_wt"] > 500:
            liabilities.append("lipinski_mw")
        if descriptors["logp"] > 5:
            liabilities.append("lipinski_logp")
        if descriptors["hbd"] > 5:
            liabilities.append("lipinski_hbd")
        if descriptors["hba"] > 10:
            liabilities.append("lipinski_hba")
        if descriptors["tpsa"] > 140:
            liabilities.append("low_passive_permeability")
        if descriptors["rotatable_bonds"] > 10:
            liabilities.append("high_flexibility")
        if descriptors["logp"] < -1:
            liabilities.append("low_membrane_partitioning")
        if abs(descriptors["formal_charge"]) > 1:
            liabilities.append("high_formal_charge")
        return liabilities

    @staticmethod
    def _scores(descriptors: dict[str, float], liabilities: list[str]) -> dict[str, float]:
        lipinski_violations = sum(1 for item in liabilities if item.startswith("lipinski"))
        permeability = 1.0
        permeability -= max(0.0, descriptors["tpsa"] - 90.0) / 100.0
        permeability -= max(0.0, descriptors["rotatable_bonds"] - 6.0) / 10.0
        permeability -= max(0.0, abs(descriptors["formal_charge"]) - 1.0) * 0.25

        solubility = 1.0
        solubility -= max(0.0, descriptors["logp"] - 2.5) / 4.0
        solubility -= max(0.0, descriptors["mol_wt"] - 400.0) / 300.0

        developability = (
            0.35 * descriptors["qed"]
            + 0.25 * max(0.0, 1.0 - lipinski_violations / 4.0)
            + 0.20 * max(0.0, min(1.0, permeability))
            + 0.20 * max(0.0, min(1.0, solubility))
        )

        return {
            "qed": round(descriptors["qed"], 4),
            "lipinski_violations": float(lipinski_violations),
            "permeability_proxy": round(max(0.0, min(1.0, permeability)), 4),
            "solubility_proxy": round(max(0.0, min(1.0, solubility)), 4),
            "developability_proxy": round(max(0.0, min(1.0, developability)), 4),
        }

    @staticmethod
    def _recommendations(descriptors: dict[str, float], liabilities: list[str]) -> list[str]:
        recommendations: list[str] = []
        if "lipinski_logp" in liabilities:
            recommendations.append(
                "Reduce lipophilicity or add polar surface without over-increasing TPSA."
            )
        if "low_passive_permeability" in liabilities:
            recommendations.append(
                "Lower TPSA or reduce exposed donors to improve passive permeability."
            )
        if "high_flexibility" in liabilities:
            recommendations.append(
                "Constrain rotatable bonds with ring closure or bioisosteric replacement."
            )
        if "lipinski_mw" in liabilities:
            recommendations.append("Trim nonessential substituents to reduce molecular weight.")
        if not recommendations:
            recommendations.append("No major rule-based ADMET liabilities detected.")
        return recommendations
