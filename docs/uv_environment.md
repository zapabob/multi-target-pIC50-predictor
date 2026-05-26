# UV Environment

This project is managed with UV. The default environment installs the scientific
stack needed for pIC50 prediction, RDKit-based ADMET, ETKDG 3D conformers,
synthetic feasibility, image features, and tests.

## Create the environment

```bash
uv sync
```

## Useful extras

```bash
# Prefect workflow orchestration
uv sync --extra workflow

# PyTorch Geometric model adapters for GNN, SchNet, and DimeNet++
uv sync --extra gnn

# GUI
uv sync --extra gui

# Protein structure and docking file helpers
uv sync --extra structure

# Production API/runtime dependencies
uv sync --extra prod
```

Airflow is separate because it should be installed with official constraints:

```bash
uv pip install "apache-airflow==2.10.5" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.10.5/constraints-3.12.txt"
```

If a PyG operation needs compiled extensions such as `torch-scatter` or
`torch-sparse`, install them after `torch` is synced, using the PyG wheel index
that matches the local Torch/CUDA build.

## Validate

```bash
uv run python scripts/smoke_uv_env.py
uv run pytest tests/test_discovery_extensions.py tests/test_structure_integration_contracts.py
```

## Run discovery triage

```bash
uv run python cli.py assess --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --output assessment.json
```
