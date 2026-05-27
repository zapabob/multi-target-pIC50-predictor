# 2026-05-27 Bandit Low Hardening

## Goal

Clear the remaining focused Bandit LOW findings while preserving existing downloader,
docking-runner, and reaction-planning behavior.

## Changes

- Wrapped local Ollama CLI execution in `download_txgemma.py` with executable lookup,
  non-empty argv validation, and explicit `shell=False` semantics.
- Wrapped docking backend execution in `src/integrations/structure_pipeline.py` with
  argv normalization and empty-command rejection while preserving dry-run output.
- Replaced the RDKit invalid-product `continue` branch in `src/reactions/planner.py`
  with a helper that returns `None` for unsanitizable products.
- Added a regression test for the non-dry-run docking command path using a local
  Python argv command.

## Verification

- Focused Bandit scan: `HIGH=0`, `MEDIUM=0`, `LOW=0`.
- Python compile check: passed for the touched modules.
- Pytest: `tests/test_structure_integration_contracts.py` passed.
