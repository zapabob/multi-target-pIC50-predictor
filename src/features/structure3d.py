"""3D conformer generation and geometry descriptors."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class ConformerGenerationResult:
    """Serializable result from ETKDG conformer generation."""

    smiles: str
    canonical_smiles: str | None
    success: bool
    method: str
    conformer_count: int
    best_conformer_id: int | None
    best_energy: float | None
    descriptors: dict[str, float]
    coordinates: list[dict[str, Any]]
    mol_block: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ETKDGConformerGenerator:
    """Generate low-energy 3D conformers using RDKit ETKDG."""

    def __init__(
        self,
        num_conformers: int = 20,
        prune_rms_thresh: float = 0.5,
        random_seed: int = 61453,
        optimize: str = "mmff",
        include_coordinates: bool = False,
    ):
        self.num_conformers = num_conformers
        self.prune_rms_thresh = prune_rms_thresh
        self.random_seed = random_seed
        self.optimize = optimize.lower()
        self.include_coordinates = include_coordinates
        self.logger = logging.getLogger(__name__)

    def generate(self, smiles: str) -> ConformerGenerationResult:
        """Generate conformers and return the best geometry."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError as exc:
            return self._failure(smiles, f"RDKit is required for ETKDG: {exc}")

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return self._failure(smiles, "Invalid SMILES")

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        mol = Chem.AddHs(mol)

        try:
            params = AllChem.ETKDGv3()
            params.randomSeed = int(self.random_seed)
            params.pruneRmsThresh = float(self.prune_rms_thresh)
            params.numThreads = 0
            if hasattr(params, "maxAttempts"):
                params.maxAttempts = 1000

            conformer_ids = list(
                AllChem.EmbedMultipleConfs(
                    mol,
                    numConfs=int(self.num_conformers),
                    params=params,
                )
            )
            if not conformer_ids:
                return self._failure(smiles, "ETKDG could not embed a conformer", canonical_smiles)

            energies = self._optimize_conformers(mol, conformer_ids)
            best_conformer_id = self._select_best_conformer(conformer_ids, energies)
            descriptors = self._geometry_descriptors(mol, best_conformer_id)
            coordinates = (
                self._coordinates(mol, best_conformer_id) if self.include_coordinates else []
            )
            mol_block = Chem.MolToMolBlock(mol, confId=best_conformer_id)

            return ConformerGenerationResult(
                smiles=smiles,
                canonical_smiles=canonical_smiles,
                success=True,
                method=f"ETKDGv3+{self.optimize.upper()}",
                conformer_count=len(conformer_ids),
                best_conformer_id=int(best_conformer_id),
                best_energy=energies.get(best_conformer_id),
                descriptors=descriptors,
                coordinates=coordinates,
                mol_block=mol_block,
            )
        except Exception as exc:
            self.logger.exception("3D conformer generation failed")
            return self._failure(smiles, str(exc), canonical_smiles)

    def generate_batch(self, smiles_list: Iterable[str]) -> list[ConformerGenerationResult]:
        """Generate conformers for a batch of SMILES."""
        return [self.generate(smiles) for smiles in smiles_list]

    def _optimize_conformers(self, mol: Any, conformer_ids: list[int]) -> dict[int, float | None]:
        from rdkit.Chem import AllChem

        energies: dict[int, float | None] = {}
        for conf_id in conformer_ids:
            try:
                if self.optimize == "mmff" and AllChem.MMFFHasAllMoleculeParams(mol):
                    props = AllChem.MMFFGetMoleculeProperties(mol)
                    ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
                elif self.optimize in {"uff", "mmff"}:
                    ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                else:
                    energies[conf_id] = None
                    continue

                if ff is None:
                    energies[conf_id] = None
                    continue
                ff.Minimize(maxIts=500)
                energies[conf_id] = float(ff.CalcEnergy())
            except Exception:
                energies[conf_id] = None
        return energies

    @staticmethod
    def _select_best_conformer(conformer_ids: list[int], energies: dict[int, float | None]) -> int:
        scored = [(conf_id, energies.get(conf_id)) for conf_id in conformer_ids]
        finite = [(conf_id, energy) for conf_id, energy in scored if energy is not None]
        if not finite:
            return int(conformer_ids[0])
        return int(min(finite, key=lambda item: item[1])[0])

    @staticmethod
    def _geometry_descriptors(mol: Any, conf_id: int) -> dict[str, float]:
        from rdkit.Chem import Descriptors3D, rdMolDescriptors

        descriptor_funcs = {
            "radius_of_gyration": rdMolDescriptors.CalcRadiusOfGyration,
            "asphericity": rdMolDescriptors.CalcAsphericity,
            "eccentricity": rdMolDescriptors.CalcEccentricity,
            "inertial_shape_factor": rdMolDescriptors.CalcInertialShapeFactor,
            "npr1": rdMolDescriptors.CalcNPR1,
            "npr2": rdMolDescriptors.CalcNPR2,
            "pmi1": rdMolDescriptors.CalcPMI1,
            "pmi2": rdMolDescriptors.CalcPMI2,
            "pmi3": rdMolDescriptors.CalcPMI3,
            "spherocity_index": Descriptors3D.SpherocityIndex,
        }

        descriptors: dict[str, float] = {}
        for name, func in descriptor_funcs.items():
            try:
                descriptors[name] = float(func(mol, confId=conf_id))
            except Exception:
                descriptors[name] = 0.0
        return descriptors

    @staticmethod
    def _coordinates(mol: Any, conf_id: int) -> list[dict[str, Any]]:
        conformer = mol.GetConformer(conf_id)
        coordinates: list[dict[str, Any]] = []
        for atom in mol.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            coordinates.append(
                {
                    "atom_index": atom.GetIdx(),
                    "symbol": atom.GetSymbol(),
                    "x": float(pos.x),
                    "y": float(pos.y),
                    "z": float(pos.z),
                }
            )
        return coordinates

    @staticmethod
    def _failure(
        smiles: str,
        error: str,
        canonical_smiles: str | None = None,
    ) -> ConformerGenerationResult:
        return ConformerGenerationResult(
            smiles=smiles,
            canonical_smiles=canonical_smiles,
            success=False,
            method="ETKDGv3",
            conformer_count=0,
            best_conformer_id=None,
            best_energy=None,
            descriptors={},
            coordinates=[],
            mol_block=None,
            error=error,
        )
