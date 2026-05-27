# 2026-05-27 security and Dependabot remediation - Codex

## Goal

Close open Dependabot alerts and remove unsafe local deserialization paths while preserving the pIC50 predictor workflows.

## Scope

- Replaced local pickle caches/config snapshots with JSON-compatible formats.
- Restricted PyTorch checkpoint loading with `weights_only=True`.
- Replaced cache-key MD5 hashes with SHA-256.
- Added an explicit timeout to the TxGemma downloader request.
- Changed the default app bind host from all interfaces to localhost; production can still opt in with `HOST`.
- Replaced `python-jose[cryptography]` with `PyJWT[crypto]` to remove the unpatched `ecdsa` transitive dependency.
- Updated the separate Airflow install guidance and pin to `apache-airflow==3.2.1`.
- Initialized CodeGraph for local structural analysis; only `.codegraph/.gitignore` is tracked.

## Verification

- `uv lock --check`: passed.
- `uv lock --dry-run`: no lockfile changes detected.
- Python AST/compile checks for touched Python files: passed.
- Bandit focused scan for `dat_predictor.py`, `src`, and `download_txgemma.py`: 0 high, 0 medium, 6 low residual legacy findings.
- `pickle.load`, `pickle.dump`, `hashlib.md5`, and old Airflow pins are absent from tracked Python/dependency/docs files.
- A focused pytest run for `LitPIC50.load_from_checkpoint` exceeded the local timeout during test startup, so runtime pytest is recorded as inconclusive in this session.
