# Pharma MVP CPU Demo

## Purpose

This CPU demo is a portfolio-ready, pharma-facing walkthrough path for
multi-target pIC50 triage. It is designed to show reproducible product behavior:
fixed data, a checked-in model artifact, target-level benchmark metrics,
applicability-domain checks, uncertainty estimates, and FastAPI endpoints.

It is not scientific validation for lead selection. The bundled dataset is a
small demo fixture. Replace it with a governed ChEMBL or sponsor snapshot before
using the scores for project decisions.

## Context of Use

- Intended use: early discovery research triage and software demonstration.
- Decision role: decision support only.
- Not for: clinical decisions, regulatory submissions, manufacturing release,
  patient care, or automated compound progression.
- Endpoint: target-specific pIC50 derived from IC50 nM values.
- Current model: CPU descriptor Ridge baseline using RDKit descriptors and
  scikit-learn.

## Evidence Package

- Fixed benchmark: `data/demo_pic50_benchmark.csv`
- CPU model artifact: `models/demo_cpu_pic50_model.json`
- Benchmark report: `artifacts/demo_cpu_benchmark.json`
- ChEMBL fixed snapshot builder: `uv run python -B cli.py build-chembl-snapshot`
- Supported demo targets: `CHEMBL238` and `CHEMBL224`
- Splits: `train`, `scaffold_test`, and `external`
- Metrics: `R2`, `RMSE`, `MAE`, and sample count per target and split

The model stores descriptor min/max ranges from the training split. A prediction
outside those ranges is returned with `applicability_domain.in_domain = false`
and an inflated uncertainty estimate.

## CPU Commands

Refresh the model and benchmark report:

```bash
uv run python -B scripts/build_demo_cpu_model.py
```

Run CLI prediction:

```bash
uv run python -B cli.py predict \
  --model models/demo_cpu_pic50_model.json \
  --target CHEMBL238 \
  --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" \
  --uncertainty
```

Run the API:

```bash
uv run uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

Run with Docker Compose CPU profile:

```bash
docker compose -f docker-compose.cpu.yml up --build
```

## ChEMBL Fixed Snapshot

For a pharma review, replace the demo fixture with a frozen ChEMBL snapshot. This
prevents silent benchmark drift from live ChEMBL queries and gives reviewers a
manifest they can inspect.

```bash
uv run python -B cli.py build-chembl-snapshot \
  --targets CHEMBL238,CHEMBL224,CHEMBL218,CHEMBL253,CHEMBL233,CHEMBL236,CHEMBL237 \
  --output data/chembl_pic50_snapshot.csv \
  --manifest artifacts/chembl_pic50_snapshot.manifest.json
```

For a quick dry run:

```bash
uv run python -B cli.py build-chembl-snapshot \
  --targets CHEMBL238,CHEMBL224 \
  --max-rows-per-target 200 \
  --output data/chembl_pic50_snapshot_sample.csv \
  --manifest artifacts/chembl_pic50_snapshot_sample.manifest.json
```

The manifest records:

- target IDs and per-target row counts
- assay filters: `IC50`, `nM`, pIC50 range `0..15`
- split policy: stable Murcko-scaffold hash with train, scaffold-test, and
  external splits
- CSV SHA-256 checksum
- research-only context of use

Generated ChEMBL snapshots are local evaluation artifacts and are ignored by Git
unless explicitly force-added after data-license and confidentiality review.

## API Examples

Health:

```bash
curl http://127.0.0.1:8000/health
```

Prediction:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d "{\"smiles\":\"CC(=O)OC1=CC=CC=C1C(=O)O\",\"target\":\"CHEMBL238\"}"
```

Assessment:

```bash
curl -X POST http://127.0.0.1:8000/assess \
  -H "Content-Type: application/json" \
  -d "{\"smiles\":\"CC(=O)OC1=CC=CC=C1C(=O)O\",\"target\":\"CHEMBL238\",\"include_3d\":false,\"include_reactions\":false}"
```

## Regulatory Alignment Notes

The implementation is organized around the pharma AI evidence themes highlighted
by FDA, EMA, and OECD materials:

- clear context of use
- data governance and traceable benchmark artifacts
- risk-based performance assessment
- lifecycle management through rebuildable artifacts
- QSAR-style endpoint, algorithm, applicability-domain, predictivity, and
  interpretability documentation

Primary references:

- FDA, Guiding Principles of Good AI Practice in Drug Development:
  https://www.fda.gov/about-fda/artificial-intelligence-drug-development/guiding-principles-good-ai-practice-drug-development
- FDA, Artificial Intelligence for Drug Development:
  https://www.fda.gov/about-fda/center-drug-evaluation-and-research-cder/artificial-intelligence-drug-development
- OECD, Principles for QSAR validation:
  https://www.oecd.org/content/dam/oecd/en/topics/policy-sub-issues/assessment-of-chemicals/oecd-principles-for-the-validation-for-regulatory-purposes-of-quantitative-structure-activity-relationship-models.pdf

## Upgrade Path to Pharma MVP

1. Replace the demo fixture with a versioned ChEMBL or sponsor data snapshot.
2. Add assay protocol filters and endpoint harmonization rules.
3. Expand scaffold split and external validation by target.
4. Add model cards for each promoted model version.
5. Add drift monitoring for descriptor and prediction distributions.
6. Add authenticated API access, audit logs, and model registry promotion gates.
