"""Tests for structure and docking integration contracts."""

from src.integrations import AlphaFold3JobSpec, DockingJobSpec, ProteinTarget


def test_alphafold3_job_spec_exports_expected_json_shape():
    spec = AlphaFold3JobSpec(
        name="dat_ligand_complex",
        protein=ProteinTarget(name="DAT", sequence="ACDE", chain_id="A"),
        ligand_smiles="CCO",
        model_seeds=[1],
    )

    payload = spec.to_alphafold3_json()

    assert payload["dialect"] == "alphafold3"
    assert payload["version"] == 1
    assert payload["modelSeeds"] == [1]
    assert payload["sequences"][0]["protein"]["id"] == "A"
    assert payload["sequences"][1]["ligand"]["smiles"] == "CCO"


def test_docking_job_spec_is_serializable():
    job = DockingJobSpec(
        name="dock_test",
        receptor_path="receptor.pdbqt",
        ligand_path="ligand.pdbqt",
        center=(1.0, 2.0, 3.0),
        box_size=(20.0, 20.0, 20.0),
    )

    assert job.to_dict()["engine"] == "vina"
