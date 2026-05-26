"""Smoke-check the UV-managed discovery environment."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


REQUIRED_IMPORTS = [
    "numpy",
    "pandas",
    "rdkit",
    "sklearn",
    "torch",
    "PIL",
    "pytest",
]

OPTIONAL_IMPORTS = [
    "prefect",
    "torch_geometric",
    "Bio",
    "gemmi",
]


def check_import(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
    except Exception as exc:
        print(f"{module_name}: missing ({exc.__class__.__name__}: {exc})")
        return False
    print(f"{module_name}: ok")
    return True


def main() -> int:
    print("Required libraries")
    required_ok = all(check_import(module_name) for module_name in REQUIRED_IMPORTS)

    print("\nOptional libraries")
    for module_name in OPTIONAL_IMPORTS:
        check_import(module_name)

    print("\nProject imports")
    from src.integrations import AlphaFold3JobSpec, ProteinTarget
    from src.pipeline import CompoundAssessmentPipeline

    result = CompoundAssessmentPipeline().assess(
        "CC(=O)OC1=CC=CC=C1C(=O)O",
        include_3d=True,
        include_reactions=True,
        include_image=True,
    )
    print(f"compound_assessment: {'ok' if result.admet['success'] else 'failed'}")

    spec = AlphaFold3JobSpec(
        name="smoke_test",
        protein=ProteinTarget(name="test", sequence="ACDE", chain_id="A"),
        ligand_smiles="CCO",
        model_seeds=[1],
    )
    print(f"alphafold3_json: {spec.to_alphafold3_json()['dialect']}")

    return 0 if required_ok and result.admet["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
