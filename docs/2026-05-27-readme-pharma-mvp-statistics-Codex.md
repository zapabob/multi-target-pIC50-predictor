# 2026-05-27 README Pharma MVP Statistics - Codex

## Work Date

2026-05-27

## Implementer

Codex

## Purpose and Scope

Repositioned the README for pharma, MLOps, LLMOps, and AI engineering reviewers.
The change adds the CHEMBL238 CPU benchmark metrics, methylphenidate
literature-vs-model statistics, an error-bar plot, p-value, effect size, and
observed power. It also adds a reproducible script for regenerating the README
statistics and graph from local JSON evidence.

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
- Common `SOP/CONTENT_DESIGN.md`
- `superpowers:test-driven-development`

## Changed Files

- `.gitignore`
- `README.md`
- `docs/assets/methylphenidate_chembl238_errorbar.png`
- `docs/assets/methylphenidate_chembl238_readme_stats.json`
- `scripts/build_pharma_mvp_readme_assets.py`
- `src/reporting/__init__.py`
- `src/reporting/pharma_mvp_assets.py`
- `tests/test_pharma_mvp_readme_assets.py`
- `docs/2026-05-27-readme-pharma-mvp-statistics-Codex.md`

## Verification Performed

- Wrote a failing test first for README stats and plot generation.
- Verified the test failed because `src.reporting.pharma_mvp_assets` did not
  exist.
- Implemented the reporting module and script.
- Ran `uv run python -B -m pytest tests\test_pharma_mvp_readme_assets.py -q`.
- Ran `uv run python -B scripts\build_pharma_mvp_readme_assets.py`.
- Visually checked the generated PNG for readable title, error bars, p-value,
  effect size, and power annotation.

## Safety, Security, and Operations Decisions

- README claims remain scoped to research-use decision support and explicitly
  avoid regulatory, clinical, manufacturing, or patient-care claims.
- The observed power is labeled as post-hoc and based on only four literature
  IC50 values.
- The graph is regenerated from JSON artifacts instead of being hand-edited.
- UTF-8 was used for all new text files.

## Residual Risks and Next Actions

- The CPU Ridge model remains a demo baseline and underpredicts
  methylphenidate potency by about 1.33 log units.
- The methylphenidate check is a sanity check, not a confirmatory validation
  study.
- Before serious pharma diligence, replace the local snapshot with a governed
  ChEMBL or sponsor snapshot and add model registry, calibration, drift
  monitoring, audit logging, and lifecycle management.
