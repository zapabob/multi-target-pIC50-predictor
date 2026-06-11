# Research-Only CPU Demo

## Purpose

This CPU demo is a research-only walkthrough path for retrospective
multi-target pIC50 validation. It is designed to show reproducible software
behavior: fixed data, a checked-in model artifact, target-level benchmark
metrics, applicability-domain checks, uncertainty estimates, and FastAPI
endpoints.

It is not scientific validation for lead selection, compound ranking,
optimization, synthesis, dosing, human-use guidance, clinical decisions,
regulatory submissions, or manufacturing decisions. The bundled dataset is a
small demo fixture. Replace it with a governed ChEMBL snapshot before using the
scores for scientific interpretation.

## Context of Use

- Intended use: retrospective research analysis and software demonstration.
- Decision role: non-decisional evidence reporting only.
- Not for: clinical decisions, regulatory submissions, manufacturing release,
  patient care, automated compound progression, synthesis planning, dosing, or
  human-use guidance.
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

For a scientific review, replace the demo fixture with a frozen ChEMBL snapshot.
This prevents silent benchmark drift from live ChEMBL queries and gives
reviewers a manifest they can inspect.

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

## Upgrade Path to Credible QSAR Review

1. Replace the demo fixture with a versioned ChEMBL data snapshot.
2. Add assay protocol filters and endpoint harmonization rules.
3. Expand scaffold split and external validation by target.
4. Add model cards for each promoted model version.
5. Add drift monitoring for descriptor and prediction distributions.
6. Add authenticated API access, audit logs, and model registry promotion gates.

## ELT Deep-Learning Candidate

The methylphenidate sanity check shows a useful but limited CPU baseline:
directionally active, yet about 1.33 pIC50 log units weaker than literature. The
next deep-learning candidate is an elastic-looped Transformer adapted from
`zapabob/elastic-looped-transformer`.

Why it fits here:

- It is the third deep-learning path after the descriptor Transformer and GNN.
- It shares one Transformer block across selectable loop iterations, so the same
  checkpoint can be run at different compute budgets.
- For pIC50 triage, loop count can become an evaluation axis alongside
  uncertainty, applicability domain, scaffold split, and external validation.
- The pIC50 implementation keeps the dependency surface small by adapting the
  looped-Transformer pattern directly instead of importing the causal-LM repo.

Command:

```bash
uv run python -B cli.py train-elt --target CHEMBL238 --loop-count 4 --epochs 20
```

Snapshot smoke run:

```bash
uv run python -B scripts/run_elt_chembl238_smoke.py
```

Current CHEMBL238 smoke result from `artifacts/elt_chembl238_smoke_report.json`:

| Loop count | Methylphenidate pIC50 | Uncertainty | Delta vs literature mean | Fold weaker than literature |
| --- | ---: | ---: | ---: | ---: |
| 1 | 4.7812 | 0.6351 | -2.5907 | 389.6727 |
| 2 | 6.0679 | 0.6477 | -1.3040 | 20.1372 |
| 3 | 6.3114 | 0.6528 | -1.0605 | 11.4948 |
| 4 | 6.3530 | 0.6522 | -1.0189 | 10.4448 |

This is a smoke run, not model selection evidence. The split metrics remain
weak (`external R2 = -0.0213`, `RMSE = 1.1566`), but the loop trajectory shows
the ELT mechanism is doing something useful for methylphenidate: deeper loops
move the prediction closer to the literature mean and improve on the Ridge
baseline by about 0.313 pIC50 at loop 4.

References checked on 2026-05-27:

- GitHub: https://github.com/zapabob/elastic-looped-transformer
- arXiv: https://arxiv.org/abs/2604.09168
- Hugging Face: `zapabobouj` model listings were checked through
  `https://huggingface.co/api/models?author=zapabobouj`; the AEGIS cards are
  used only as contextual transformer/multimodal references, not as imported
  chemistry models.

## GNN vs Multimodal ELT Cross-Validation Smoke

The next step beyond the single external split smoke is a shared-fold comparison
between compact GNN and a ViT-style multimodal ELT. The ELT variant renders the
molecule image, converts the grayscale image grid into non-overlapping ViT-like
patch tokens, concatenates those with descriptor tokens, and applies the same
weight-shared looped Transformer block. This is inspired by the visual-generation
ELT paper's recurrent shared block and Any-Time loop budget, while keeping the
chemistry implementation CPU-runnable.

Command:

```bash
uv run python -B cli.py deep-cv --folds 3 --epochs 2 --max-rows 240
```

Current checked report: `artifacts/deep_cv_chembl238_report.json`.

| Model | CV rows | Folds | R2 mean | RMSE mean | MAE mean | MSE loss |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| multimodal ELT | 240 | 3 | -0.0342 | 1.1735 | 0.9881 | 1.3785 |
| compact GNN | 240 | 3 | -0.1055 | 1.2146 | 1.0161 | 1.4854 |

This is still smoke evidence. It proves that GNN and multimodal ELT can now be
evaluated under the same stable scaffold fold policy on CPU. It does not replace
the governed full-snapshot ChEMBL evaluation, longer training, hyperparameter
tuning, calibration, or external validation.

## Category-Expanded Scaffold CV

The scaffold CV runner now supports multi-label category metrics for:

- psychedelic: `CHEMBL224` / 5-HT2A rows
- cannabinoid: `CHEMBL218` / CB1 and `CHEMBL253` / CB2 rows
- opioid: `CHEMBL233` / mu-opioid and `CHEMBL236` / delta-opioid rows in the
  checked run
- phenethylamine: a structure rule for aromatic ring to aliphatic nitrogen
  within two to three bonds

Snapshot command:

```bash
uv run python -B cli.py build-chembl-snapshot --targets CHEMBL224,CHEMBL218,CHEMBL253,CHEMBL233,CHEMBL236,CHEMBL238 --output data/chembl_category_pic50_snapshot.csv --manifest artifacts/chembl_category_pic50_snapshot.manifest.json --max-rows-per-target 300
```

CV command:

```bash
uv run python -B cli.py deep-cv --snapshot data/chembl_category_pic50_snapshot.csv --output artifacts/deep_cv_category_report.json --target ALL --folds 3 --epochs 2 --max-rows 0
```

Current checked category report:

| Model | Category | n | R2 | RMSE | MAE | MSE loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| multimodal ELT | overall | 1,538 | 0.1413 | 1.2627 | 1.0510 | 1.5952 |
| GNN | overall | 1,538 | 0.0118 | 1.3560 | 1.1249 | 1.8410 |
| multimodal ELT | psychedelic | 272 | -0.5130 | 1.2497 | 0.9878 | 1.5616 |
| GNN | psychedelic | 272 | -0.5412 | 1.2612 | 1.0163 | 1.5907 |
| multimodal ELT | cannabinoid | 488 | 0.1243 | 1.3665 | 1.1450 | 1.8672 |
| GNN | cannabinoid | 488 | 0.0032 | 1.4579 | 1.2177 | 2.1256 |
| multimodal ELT | opioid | 519 | 0.0141 | 1.2454 | 1.0875 | 1.5511 |
| GNN | opioid | 519 | -0.1291 | 1.3328 | 1.1560 | 1.7764 |
| multimodal ELT | phenethylamine | 1,066 | 0.1576 | 1.3040 | 1.1024 | 1.7005 |
| GNN | phenethylamine | 1,066 | 0.0022 | 1.4193 | 1.1932 | 2.0143 |

The snapshot contains 1,800 rows from six targets and holds out 262 external
rows from CV. Kappa opioid target fetches timed out in this local CPU run; the
category code already maps `CHEMBL237` to opioid once that target cache is
available.

## Psychopharmacology Literature Check

The project now includes a fixed, audit-friendly psychopharmacology reference
fixture:

```bash
uv run python -B scripts/run_psychopharm_literature_check.py --reference data/psychopharm_literature_reference.csv --model models/chembl_category_cpu_pic50_model.json --output artifacts/psychopharm_literature_prediction_check.json
```

The run checks Adderall as a d-amphetamine proxy, LSD, delta-9-THC, morphine,
methylphenidate, and bkMDMA/methylone against curated ChEMBL activity rows. It
uses pX-style `Ki`/`IC50`/`EC50` values as a sanity check and labels endpoint or
species caveats.

| Compound | Target | Literature mean pX | Predicted pIC50 | Delta | Fold error | Domain |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Adderall / d-amphetamine | DAT | 7.0367 | 5.9340 | -1.1027 | 12.6678 | out |
| LSD | 5HT2A | 8.3186 | 6.7230 | -1.5956 | 39.4094 | out |
| delta-9-THC | CB1 | 7.9650 | 7.9280 | -0.0370 | 1.0889 | in |
| delta-9-THC | CB2 | 7.6800 | 6.4260 | -1.2540 | 17.9473 | out |
| Morphine | mu-opioid | 8.6300 | 6.3950 | -2.2350 | 171.7908 | in |
| Morphine | delta-opioid | 6.6850 | 6.1670 | -0.5180 | 3.2961 | in |
| Methylphenidate | DAT | 7.3250 | 6.1540 | -1.1710 | 14.8252 | in |
| bkMDMA / methylone | DAT | 6.8800 | 5.6640 | -1.2160 | 16.4437 | in |

Interpretation for MVP review: the evidence is not uniformly favorable, and
that is the point. The CB1 delta-9-THC check is close to literature, while LSD,
mu-opioid morphine, and DAT stimulants remain underpredicted. This gives pharma,
MLOps, LLMOps, and AI reviewers a concrete calibration backlog instead of a
single cherry-picked example.
