# Multi-Target pIC50 Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://www.docker.com/)

A drug-discovery research toolkit for multi-target pIC50 prediction, compound
triage, and early medicinal chemistry decision support.

The project started as a DAT activity predictor and now includes a modular
discovery pipeline for:

- multi-target pIC50 modeling across DAT, 5-HT2A, CB1, CB2, and opioid receptors
- RDKit descriptors, ECFP4/MACCS fingerprints, SMARTS flags, and graph features
- ETKDG 3D conformer generation with geometry descriptors
- ADMET and developability triage
- synthetic accessibility scoring with SA score and SCScore-style proxies
- retrosynthesis and forward-reaction baseline planning
- molecule image features for multimodal image + structure experiments
- optional Prefect/Airflow-style automation hooks
- future AlphaFold3 and docking simulation integration contracts

This code is intended for research and prioritization. It is not a clinical,
regulatory, or manufacturing decision system.

## Repository Layout

```text
.
|-- cli.py                         # Command-line entry point
|-- dat_predictor.py               # Legacy DAT predictor and GUI logic
|-- pyproject.toml                 # UV-managed project dependencies
|-- uv.lock                        # Reproducible dependency lockfile
|-- src/
|   |-- admet/                     # ADMET and developability profiling
|   |-- active_learning/           # Compound selection helpers
|   |-- data/                      # ChEMBL loading and dataset splitting
|   |-- features/                  # Molecular, graph, and 3D featurizers
|   |-- integrations/              # AlphaFold3 and docking job contracts
|   |-- models/                    # Transformer, GNN, ensemble, geometry GNNs
|   |-- multimodal/                # Molecule image feature extraction
|   |-- pipeline/                  # Integrated compound assessment workflows
|   |-- reactions/                 # Retro/forward reaction planning baseline
|   `-- synthesis/                 # Synthetic accessibility scoring
|-- tests/                         # Unit and integration tests
|-- docs/                          # Design and environment notes
`-- scripts/                       # Environment smoke checks and utilities
```

## Installation with UV

UV is the preferred environment manager for this repository.

```bash
uv sync
```

The default environment installs the core scientific stack:

- RDKit
- NumPy, pandas, SciPy, scikit-learn
- PyTorch
- pytest and development tools
- pIC50, ADMET, 3D conformer, synthesis, and image-feature dependencies

Optional extras are available for heavier workflows:

```bash
# Prefect orchestration
uv sync --extra workflow

# PyTorch Geometric model adapters
uv sync --extra gnn

# Protein structure and docking file helpers
uv sync --extra structure

# GUI dependencies
uv sync --extra gui

# API and production runtime dependencies
uv sync --extra prod
```

Airflow is intentionally managed separately because it should be installed with
the official Airflow constraints file:

```bash
uv pip install "apache-airflow==2.10.5" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"
```

See [docs/uv_environment.md](docs/uv_environment.md) for more details.

## Quick Start

Run a no-model compound assessment. This produces ADMET and synthesis outputs,
and can optionally include 3D, reaction, and image features.

```bash
uv run python cli.py assess --smiles "CC(=O)OC1=CC=CC=C1C(=O)O"
```

Write the assessment to JSON:

```bash
uv run python cli.py assess \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --output artifacts/assessment.json
```

Assess a file of SMILES strings:

```bash
uv run python cli.py assess --input compounds.smi --output artifacts/assessment.csv
```

Include rendered image features:

```bash
uv run python cli.py assess \
  --input compounds.smi \
  --include-image \
  --output artifacts/multimodal_assessment.json
```

Use a trained pIC50 model when available:

```bash
uv run python cli.py assess \
  --model models/dat_transformer_model.pt \
  --target CHEMBL238 \
  --smiles "CCN(CC)CC"
```

## Core Workflows

### pIC50 Prediction

The existing pIC50 workflow supports ChEMBL-based target data retrieval,
feature calculation, Transformer training, and prediction with optional
uncertainty reporting.

```bash
uv run python cli.py train --target CHEMBL238 --optimize
uv run python cli.py predict --model models/dat_transformer_model.pt --smiles "CCN(CC)CC"
```

Supported target examples:

| Target | ChEMBL ID | Description |
| --- | --- | --- |
| DAT | CHEMBL238 | Dopamine transporter |
| 5-HT2A | CHEMBL224 | Serotonin 2A receptor |
| CB1 | CHEMBL218 | Cannabinoid receptor 1 |
| CB2 | CHEMBL253 / CHEMBL1861 | Cannabinoid receptor 2 |
| mu opioid | CHEMBL233 | Mu opioid receptor |
| delta opioid | CHEMBL236 | Delta opioid receptor |
| kappa opioid | CHEMBL237 | Kappa opioid receptor |

### 3D Structure Features

`src/features/structure3d.py` generates RDKit ETKDGv3 conformers, optimizes
them with MMFF or UFF, and returns 3D descriptors such as radius of gyration,
asphericity, eccentricity, principal moments of inertia, and spherocity index.

These descriptors are available through the integrated `assess` command.

### Geometry-Aware GNNs

`src/models/geometry_gnn.py` defines a factory adapter for SchNet and DimeNet++.
Install the GNN extra before using these models:

```bash
uv sync --extra gnn
```

Some PyTorch Geometric operations may require compiled extensions such as
`torch-scatter` or `torch-sparse`. On Windows, install those from the PyG wheel
index that matches your local Torch and CUDA build.

### ADMET Integration

`src/admet/predictor.py` provides a lightweight rule-based ADMET profile using
RDKit descriptors:

- molecular weight
- LogP
- TPSA
- HBD/HBA
- rotatable bonds
- formal charge
- QED
- permeability and solubility proxies
- developability proxy score

This is a triage layer. Replace or ensemble it with calibrated ADMET models for
production-grade prediction.

### Synthetic Accessibility

`src/synthesis/scores.py` provides:

- SA score proxy
- SCScore-style proxy
- synthetic feasibility score
- complexity drivers such as stereocenters, ring count, graph complexity,
  spiro atoms, bridgehead atoms, and flexibility

Use these outputs to rank compounds before synthesis planning or docking.

### Reaction Route Prediction

`src/reactions/planner.py` provides a conservative baseline interface for:

- retrosynthetic template disconnections
- forward reaction templates
- route serialization through `ReactionRoute` and `ReactionStep`

The current templates are intentionally simple. The interface is ready for
AiZynthFinder, ASKCOS, IBM RXN, or an in-house reaction transformer.

### Multimodal Features

`src/multimodal/image_featurizer.py` renders molecule images with RDKit and
converts them into compact image-derived features. These can be combined with
graph, descriptor, or 3D features for image + structure experiments.

### Automation

`src/pipeline/workflows.py` includes:

- batch assessment runner
- JSON/CSV result writing
- optional Prefect flow factory
- optional Airflow DAG factory

Run Prefect workflows after installing:

```bash
uv sync --extra workflow
```

## AlphaFold3 and Docking Integration

`src/integrations/structure_pipeline.py` defines project-level contracts for:

- protein target metadata
- local AlphaFold3-style protein-ligand JSON payloads
- docking job specifications
- command-line docking runners

The AlphaFold3 contract exports payloads containing `sequences`, ligand SMILES,
`modelSeeds`, `dialect`, and `version`. AlphaFold Server has non-commercial and
ligand restrictions, so this project keeps the integration focused on local or
managed AlphaFold3 deployments.

Docking support is intentionally backend-neutral. The current runner can build
command lines for tools such as Vina, Gnina, or site-specific docking wrappers.

## Validation

Run the environment smoke check:

```bash
uv run python -B scripts/smoke_uv_env.py
```

Run the discovery extension tests:

```bash
uv run python -B -m pytest tests/test_discovery_extensions.py tests/test_structure_integration_contracts.py -q
```

Run Ruff checks:

```bash
uv run ruff check . --preview
uv run ruff format . --check
```

The current codebase still contains legacy Ruff issues outside the newly added
discovery modules. Treat a full-project Ruff cleanup as a separate refactoring
task.

## Docker Production Stack

The production compose stack includes:

- application service
- PostgreSQL
- Redis
- Ollama for TxGemma
- Nginx
- Prometheus
- Grafana

Typical production deployment:

```bash
cp .env.example .env
cp config/config.yaml.example config/config.yaml
./scripts/deploy.sh production deploy
```

Service defaults:

- application: `http://localhost:8000`
- Grafana: `http://localhost:3000`
- Prometheus: `http://localhost:9090`

## Research Notes

- Use scaffold splits when evaluating medicinal chemistry generalization.
- Keep pIC50, ADMET, synthesis, docking, and AlphaFold-derived evidence
  separate until you have calibration data for combined ranking.
- Track uncertainty and applicability domain for every prediction.
- Validate reaction routes with a chemist and a dedicated retrosynthesis engine
  before synthesis decisions.
- Treat docking and AlphaFold3 outputs as structural hypotheses, not binding
  truth.

## License

This project is licensed under the MIT License.
