# 2026-05-27 Pharma MVP CPU API - Codex

## Work Date

2026-05-27

## Implementer

Codex

## Purpose and Scope

Added a CPU-only pharma MVP demo path for the multi-target pIC50 predictor. The
scope covered a checked-in demo benchmark, reproducible CPU model artifact,
target-level metrics, applicability-domain and uncertainty output, FastAPI
prediction endpoints, and documentation of research-only context of use.

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
- Common `SOP/CICD.md`
- `superpowers:test-driven-development`
- `superpowers:verification-before-completion`

## Changed Files

- `.gitignore`
- `Dockerfile`
- `README.md`
- `cli.py`
- `src/data/chembl_snapshot.py`
- `src/data/loader.py`
- `src/utils/cache.py`
- `data/demo_pic50_benchmark.csv`
- `docker-compose.cpu.yml`
- `entrypoint.sh`
- `models/demo_cpu_pic50_model.json`
- `artifacts/demo_cpu_benchmark.json`
- `scripts/build_demo_cpu_model.py`
- `src/api/__init__.py`
- `src/api/app.py`
- `src/models/demo_cpu.py`
- `tests/test_chembl_snapshot.py`
- `tests/test_cpu_demo_mvp.py`
- `docs/pharma_mvp_cpu_demo.md`

## Verification Performed

- Wrote failing tests first for CPU artifact building, CPU prediction,
  invalid-SMILES handling, FastAPI prediction and assessment, and CLI JSON-model
  prediction.
- Ran `uv run python -B -m pytest tests\test_cpu_demo_mvp.py -q`.
- Ran `uv run python -B -m pytest tests -q` after adding the ChEMBL snapshot tests.
- Ran `uv run ruff check src\data\chembl_snapshot.py tests\test_chembl_snapshot.py src\models\demo_cpu.py src\api\app.py scripts\build_demo_cpu_model.py tests\test_cpu_demo_mvp.py cli.py --preview`.
- Rebuilt artifacts with `uv run python -B scripts\build_demo_cpu_model.py`.
- Verified CLI CPU prediction with `uv run python -B cli.py predict --model models\demo_cpu_pic50_model.json --target CHEMBL238 --smiles "CC(=O)OC1=CC=CC=C1C(=O)O" --uncertainty`.
- Verified FastAPI health through `TestClient`, returning status `200` and device `cpu`.
- Ran `uv run python -B scripts\smoke_uv_env.py`.
- Added tests for fixed ChEMBL snapshot CSV and manifest generation.
- Verified `uv run python -B cli.py build-chembl-snapshot --help`.

## Safety, Security, and Operations Decisions

- The CPU demo uses JSON model artifacts, not pickle, to avoid unsafe external
  model deserialization.
- The bundled data is clearly marked as `demo_fixture` and documented as
  unsuitable for scientific or regulatory claims.
- ChEMBL snapshot generation writes a CSV and manifest with checksum, split
  policy, filters, row counts, and research-only context of use.
- ChEMBL loader raw-cache storage now converts client result objects to plain
  records before pickling, avoiding cached session serialization failures.
- Data cache falls back to pickle when parquet engines are not installed, keeping
  CPU-only environments free from optional pyarrow or fastparquet requirements.
- API output includes context of use, model version, device, uncertainty, and
  applicability-domain status.
- Docker default command now starts FastAPI, and `docker-compose.cpu.yml`
  provides a CPU-only demo service without GPU reservation.
- UTF-8 was used for all new text files.

## Residual Risks and Next Actions

- The demo fixture is intentionally small and can produce poor external metrics;
  replace it with a governed ChEMBL or sponsor snapshot for scientific use.
- A CHEMBL238 CPU run was generated locally:
  `data/chembl238_pic50_snapshot.csv`,
  `artifacts/chembl238_pic50_snapshot.manifest.json`,
  `models/chembl238_cpu_pic50_model.json`, and
  `artifacts/chembl238_cpu_benchmark.json`.
- Full-project Ruff still has legacy issues outside this change scope.
- Authentication, audit logging, model registry promotion, and drift monitoring
  are not yet implemented.
