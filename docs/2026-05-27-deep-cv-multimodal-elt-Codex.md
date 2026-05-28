# 2026-05-27 Deep CV Multimodal ELT - Codex

## Work Date

2026-05-27

## Implementer

Codex

## Purpose and Scope

Added a CPU-runnable cross-validation path for comparing compact GNN and
multimodal elastic-looped Transformer candidates on CHEMBL238. The multimodal
ELT adapts the ELT idea toward a small ViT-like chemistry model by combining
descriptor tokens with rendered molecule image patch tokens in a shared looped
Transformer.

The work was extended to category-expanded scaffold CV for psychedelic,
cannabinoid, opioid, and phenethylamine-like rows.

This work is still smoke evidence. It does not claim production model quality.

## References Checked

- GitHub repository: https://github.com/zapabob/elastic-looped-transformer
- GitHub description checked on 2026-05-27: ELT with Intra-Loop
  Self-Distillation and GRPO, causal-LM port in PyTorch.
- arXiv: https://arxiv.org/abs/2604.09168
- Hugging Face paper API: https://huggingface.co/api/papers/2604.09168
- Hugging Face author API: https://huggingface.co/api/models?author=zapabobouj
- Hugging Face contextual model example:
  `zapabobouj/AEGIS-Phi3.5-Enhanced`

The Hugging Face `zapabobouj` models were used as contextual references for
transformer, multimodal, and structured reasoning claims. No Hugging Face model
weights were imported into this chemistry repository.

## Instructions and SOP Read

- Repository-local `AGENTS.md`: missing
- Repository-local `AGENT.md`: missing
- Repository-local `SOP/README.md`: missing
- Repository-local `SOP/ENCODING.md`: missing
- Common `AGENTS.md`
- Common `SOP/README.md`
- Common `SOP/ENCODING.md`
- Common `SOP/MLOPS.md`
- Common `SOP/LLMOPS.md`
- Common `SOP/PYTHON.md`
- `github:github`
- `hugging-face:huggingface-papers`
- `deepresearch-defense-standard`
- `superpowers:test-driven-development`
- `superpowers:verification-before-completion`

## Changed Files

- `.gitignore`
- `README.md`
- `cli.py`
- `docs/pharma_mvp_cpu_demo.md`
- `docs/2026-05-27-deep-cv-multimodal-elt-Codex.md`
- `artifacts/deep_cv_chembl238_report.json`
- `artifacts/deep_cv_category_report.json`
- `artifacts/chembl_category_pic50_snapshot.manifest.json`
- `data/chembl_category_pic50_snapshot.csv`
- `scripts/run_deep_cv_chembl238.py`
- `src/models/elastic_looped_transformer.py`
- `tests/test_cli_elt.py`
- `tests/test_deep_cv_chembl238.py`
- `tests/test_multimodal_elt.py`

## Design Decision

The original ELT paper targets visual generation. The local pIC50 adaptation
keeps that visual emphasis by adding molecule render patch tokens:

- RDKit descriptors become descriptor tokens.
- Rendered molecule grayscale grids become non-overlapping ViT-style patch
  tokens.
- Graph summary vectors are appended as a graph modality token.
- A single shared `TransformerEncoderLayer` is iterated across the selected
  loop schedule.
- Fold-local pIC50 z-score standardization is used during short CPU CV and
  predictions are inverse-transformed before metrics are computed.
- Category metrics are multi-label. Rows may be counted under a target-derived
  family such as cannabinoid and also under a structure-derived family such as
  phenethylamine.

The GNN side uses the existing molecular graph featurizer and compact GCN path
so both candidates can run in a CPU smoke setting.

## Verification Performed

- Wrote failing tests first for `MultimodalElasticLoopedPIC50Model`.
- Verified the multimodal ELT test failed because the class did not exist.
- Wrote failing tests first for `scripts.run_deep_cv_chembl238`.
- Verified the deep-CV test failed because the script module did not exist.
- Wrote a failing CLI help test for `deep-cv`.
- Verified the CLI test failed because the command was not listed.
- Added fold-local target standardization and verified the report contract test
  failed before the report included the new field.
- Ran:
  `uv run python -B -m pytest tests\test_deep_cv_chembl238.py tests\test_multimodal_elt.py tests\test_cli_elt.py -q`
- Ran:
  `uv run ruff check scripts\run_deep_cv_chembl238.py src\models\elastic_looped_transformer.py cli.py tests\test_deep_cv_chembl238.py tests\test_multimodal_elt.py tests\test_cli_elt.py --preview`
- Ran:
  `uv run python -B scripts\run_deep_cv_chembl238.py --snapshot data\chembl238_pic50_snapshot.csv --report artifacts\deep_cv_chembl238_report.json --models multimodal_elt,gnn --folds 3 --epochs 2 --hidden-dim 32 --descriptor-token-count 4 --image-grid-size 16 --image-patch-size 4 --loop-count 4 --batch-size 32 --learning-rate 0.0005 --random-seed 42 --max-rows 240`
- Extended the tests so `target=ALL`, `--max-rows 0`, category metrics, fold
  losses, and graph-summary evidence must be present.
- The first category snapshot attempt with delta and kappa opioid targets
  timed out after 15 minutes while fetching `CHEMBL236`/`CHEMBL237`. The
  lingering snapshot process was stopped by PID.
- Ran:
  `uv run python -B cli.py build-chembl-snapshot --targets CHEMBL224,CHEMBL218,CHEMBL253,CHEMBL233,CHEMBL236,CHEMBL238 --output data\chembl_category_pic50_snapshot.csv --manifest artifacts\chembl_category_pic50_snapshot.manifest.json --max-rows-per-target 300 --random-seed 42 --scaffold-test-fraction 0.15 --external-fraction 0.15`
- Ran:
  `uv run python -B scripts\run_deep_cv_chembl238.py --snapshot data\chembl_category_pic50_snapshot.csv --report artifacts\deep_cv_category_report.json --target ALL --models multimodal_elt,gnn --folds 3 --epochs 2 --hidden-dim 32 --descriptor-token-count 4 --image-grid-size 16 --image-patch-size 4 --loop-count 4 --batch-size 32 --learning-rate 0.0005 --random-seed 42 --max-rows 0`

## CHEMBL238 Deep CV Smoke Result

The run used the frozen CHEMBL238 snapshot:

- target rows: 2,382
- external holdout rows excluded from CV: 261
- CV rows sampled for CPU smoke: 240
- folds: 3 stable scaffold-hash folds
- epochs: 2

Mean CV metrics:

| Model | R2 | RMSE | MAE | MSE loss | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| multimodal ELT | -0.0342 | 1.1735 | 0.9881 | 1.3785 | 240 |
| compact GNN | -0.1055 | 1.2146 | 1.0161 | 1.4854 | 240 |

## Category-Expanded Scaffold CV Result

The category snapshot contains 1,800 rows:

- CHEMBL224 / 5HT2A: psychedelic
- CHEMBL218 / CB1: cannabinoid
- CHEMBL253 / CB2: cannabinoid
- CHEMBL233 / mu-opioid: opioid
- CHEMBL236 / delta-opioid: opioid
- CHEMBL238 / DAT: contributes structural phenethylamine-like rows

External rows were excluded from CV:

- target rows: 1,800
- CV rows: 1,538
- external holdout rows: 262
- folds: 3 stable scaffold-hash folds
- epochs: 2

Overall metrics:

| Model | R2 | RMSE | MAE | MSE loss | n |
| --- | ---: | ---: | ---: | ---: | ---: |
| multimodal ELT | 0.1413 | 1.2627 | 1.0510 | 1.5952 | 1,538 |
| compact GNN | 0.0118 | 1.3560 | 1.1249 | 1.8410 | 1,538 |

Category metrics:

| Model | Category | n | R2 | RMSE | MAE | MSE loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| multimodal ELT | psychedelic | 272 | -0.5130 | 1.2497 | 0.9878 | 1.5616 |
| compact GNN | psychedelic | 272 | -0.5412 | 1.2612 | 1.0163 | 1.5907 |
| multimodal ELT | cannabinoid | 488 | 0.1243 | 1.3665 | 1.1450 | 1.8672 |
| compact GNN | cannabinoid | 488 | 0.0032 | 1.4579 | 1.2177 | 2.1256 |
| multimodal ELT | opioid | 519 | 0.0141 | 1.2454 | 1.0875 | 1.5511 |
| compact GNN | opioid | 519 | -0.1291 | 1.3328 | 1.1560 | 1.7764 |
| multimodal ELT | phenethylamine | 1,066 | 0.1576 | 1.3040 | 1.1024 | 1.7005 |
| compact GNN | phenethylamine | 1,066 | 0.0022 | 1.4193 | 1.1932 | 2.0143 |

## Residual Risks and Next Actions

- This is not full-snapshot model selection. It is a CPU smoke comparison.
- Full evaluation should remove or raise the row cap, train for longer, and
  compare against Ridge, descriptor Transformer, GNN, multimodal ELT, and
  ensemble baselines on identical frozen splits.
- Kappa opioid rows are mapped by code but were not included in the checked
  category snapshot because the live `CHEMBL237` fetch timed out locally.
- The current multimodal path uses rendered 2D molecule images. It should be
  compared with graph, 3D geometry, and fingerprint modalities before promotion.
- Hugging Face model cards were checked as references only. A chemistry model
  release would need its own model card, governed data lineage, license review,
  and lifecycle gates.
