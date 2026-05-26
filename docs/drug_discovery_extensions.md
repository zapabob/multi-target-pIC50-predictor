# Drug Discovery Extension Architecture

This repository now has a modular layer for medicinal chemistry triage beyond
single-target pIC50 prediction.

## Implemented modules

- `src/features/structure3d.py`: RDKit ETKDGv3 conformer generation, MMFF/UFF
  optimization, and 3D geometry descriptors.
- `src/models/geometry_gnn.py`: SchNet and DimeNet++ model adapter using
  PyTorch Geometric geometry models.
- `src/admet/predictor.py`: rule-based ADMET/developability profile with QED,
  Lipinski, permeability, and solubility proxies.
- `src/synthesis/scores.py`: SA score and SCScore-style heuristic proxies for
  early synthetic feasibility ranking.
- `src/reactions/planner.py`: template retrosynthesis and forward reaction
  baseline with a stable interface for external retrosynthesis engines.
- `src/multimodal/image_featurizer.py`: rendered molecule image features for
  image + structure multimodal experiments.
- `src/pipeline/compound_assessment.py`: one orchestration point combining
  pIC50, 3D, ADMET, synthesis, reactions, and multimodal features.
- `src/pipeline/workflows.py`: batch runner plus optional Prefect and Airflow
  flow/DAG factories.
- `src/integrations/structure_pipeline.py`: project-level contracts for
  AlphaFold-style complex jobs and command-line docking jobs.

## CLI usage

```bash
python cli.py assess --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --output assessment.json
python cli.py assess --input compounds.smi --output assessment.csv --include-image
python cli.py assess --model models/dat_transformer_model.pt --target CHEMBL238 --smiles "CCN(CC)CC"
```

The `assess` command works without a pIC50 model; in that mode it returns ADMET,
synthetic feasibility, 3D, and retrosynthesis outputs while leaving pIC50 empty.

## How to extend next

1. Replace the rule-based ADMET module with calibrated endpoints or multitask
   ADMET models.
2. Replace retrosynthesis templates with AiZynthFinder, ASKCOS, IBM RXN, or an
   in-house transformer, keeping the `ReactionRoute` output contract.
3. Train SchNet/DimeNet++ on ETKDG conformers or docking poses and ensemble it
   with the existing Transformer/GNN predictors.
4. Use `AlphaFold3JobSpec` to create local AlphaFold3-style protein-ligand JSON
   input with protein sequence, ligand SMILES, `modelSeeds`, `dialect`, and
   `version`.
5. Use `DockingJobSpec` and `CommandLineDockingRunner` to connect Vina,
   Gnina, DiffDock, or a site-specific docking workflow.
