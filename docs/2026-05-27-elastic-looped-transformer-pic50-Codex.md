# 2026-05-27 Elastic Looped Transformer pIC50 - Codex

## Work Date

2026-05-27

## Implementer

Codex

## Purpose and Scope

Added an ELT-inspired third deep-learning model path for pIC50 regression after
the existing descriptor Transformer and GNN paths. The goal is to test whether
the elastic-looped Transformer idea can address the CPU Ridge baseline's
methylphenidate underprediction by adding a learnable iterative-refinement model
with selectable loop count.

## References Checked

- GitHub repository: `zapabob/elastic-looped-transformer`
- Repository URL: https://github.com/zapabob/elastic-looped-transformer
- Default branch: `main`
- Description from GitHub: ELT with Intra-Loop Self-Distillation and GRPO,
  causal-LM port in PyTorch.
- Core files inspected: `README.md`, `src/elt_lm/model.py`,
  `src/elt_lm/composite.py`, `src/elt_lm/ilsd.py`, and `AGENTS.md`.
- arXiv reference: https://arxiv.org/abs/2604.09168
- Hugging Face note: local `hf` CLI was not installed. API checks on
  `https://huggingface.co/api/models?search=zapabob%2Felt` and
  `https://huggingface.co/api/models?author=zapabob` returned empty model
  lists, while `https://huggingface.co/api/models/zapabob/elt-lm-base-275m`
  did not return a public model record. Hub-side publishing and model lookup
  were therefore left as follow-up rather than claimed as complete.

## Instructions and SOP Read

- Repository-local `AGENTS.md`: missing
- Repository-local `AGENT.md`: missing
- Repository-local `SOP/README.md`: missing
- Repository-local `SOP/ENCODING.md`: missing
- Common `AGENTS.md`
- Common `SOP/README.md`
- Common `SOP/ENCODING.md`
- Common `SOP/PYTHON.md`
- Common `SOP/MLOPS.md`
- Common `SOP/LLMOPS.md`
- `deepresearch-defense-standard`
- `superpowers:test-driven-development`

## Changed Files

- `README.md`
- `cli.py`
- `docs/pharma_mvp_cpu_demo.md`
- `docs/2026-05-27-elastic-looped-transformer-pic50-Codex.md`
- `artifacts/elt_chembl238_smoke_report.json`
- `scripts/run_elt_chembl238_smoke.py`
- `src/models/elastic_looped_transformer.py`
- `tests/test_elastic_looped_transformer.py`
- `tests/test_elt_chembl238_smoke.py`
- `tests/test_cli_elt.py`

## Design Decision

The external ELT repository is a causal language-model implementation. This repo
does not import it directly. Instead, it adapts the core architectural idea:
project molecular descriptors into tokens, run a shared Transformer block for a
selectable loop schedule, and read out pIC50 plus a positive uncertainty proxy.
This keeps the pIC50 project on its current Python and PyTorch dependency
surface while making ELT a real model path that can be trained and benchmarked.

## Verification Performed

- Wrote failing tests first for the new model module and CLI entry.
- Verified the model test failed before implementation because
  `src.models.elastic_looped_transformer` did not exist.
- Verified the CLI test failed before implementation because `train-elt` was not
  listed.
- Implemented `ElasticLoopedPIC50Model`, `LitElasticLoopedPIC50`, and the
  `train-elt` CLI command.
- Ran `uv run python -B -m pytest tests\test_elastic_looped_transformer.py tests\test_cli_elt.py -q`.
- Added `scripts/run_elt_chembl238_smoke.py` to train a small CPU ELT model from
  the frozen CHEMBL238 snapshot and write methylphenidate loop predictions.
- Ran `uv run python -B scripts\run_elt_chembl238_smoke.py --snapshot data\chembl238_pic50_snapshot.csv --report artifacts\elt_chembl238_smoke_report.json --analysis artifacts\methylphenidate_chembl238_activity_analysis.json --model models\elt_chembl238_smoke.ckpt --epochs 5 --hidden-dim 64 --token-count 4 --num-heads 4 --loop-count 4 --batch-size 64 --learning-rate 0.0003 --random-seed 42`.

## CHEMBL238 Smoke Result

The 5-epoch CPU smoke run used 2,382 CHEMBL238 rows from the frozen snapshot.
Split metrics were weak but executable:

- train: R2 = 0.0023, RMSE = 1.2041, MAE = 1.0153
- scaffold_test: R2 = -0.0227, RMSE = 1.0718, MAE = 0.8835
- external: R2 = -0.0213, RMSE = 1.1566, MAE = 0.9751

Methylphenidate loop predictions:

- L=1: pIC50 4.7812
- L=2: pIC50 6.0679
- L=3: pIC50 6.3114
- L=4: pIC50 6.3530

Compared with the literature mean pIC50 7.3719, loop 4 remains 1.0189 log units
weak, or about 10.44x weaker than literature. It is nevertheless 0.3130 pIC50
closer than the CPU Ridge baseline and reduces the methylphenidate fold error
from roughly 21x to roughly 10x.

## Residual Risks and Next Actions

- The CHEMBL238 run is only a short CPU smoke run. It proves the integration and
  loop trajectory, not production model quality.
- The next evidence step is hyperparameter tuning and comparison against Ridge,
  descriptor Transformer, GNN, and ensemble baselines on the same frozen splits.
- Hub publishing should be revisited after installing or otherwise enabling the
  Hugging Face CLI or connector and confirming the target model repository.
