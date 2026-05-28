# 2026-05-28 Psychopharmacology Literature Check - Codex

## Objective

Add an audit-friendly comparison of local pIC50 predictions against curated
psychopharmacology activity values for Adderall, LSD, delta-9-THC, morphine,
methylphenidate, and bkMDMA.

## Scope

- Added `data/psychopharm_literature_reference.csv`.
- Added `scripts/run_psychopharm_literature_check.py`.
- Added `tests/test_psychopharm_literature_check.py`.
- Generated `models/chembl_category_cpu_pic50_model.json`.
- Generated `artifacts/chembl_category_cpu_benchmark.json`.
- Generated `artifacts/psychopharm_literature_prediction_check.json`.
- Updated README and the pharma MVP document with a compact results table.

## Source Basis

The fixture uses ChEMBL activity rows and ChEMBL document metadata. The values
are pX-style `Ki`, `IC50`, or `EC50` records, not a single harmonized endpoint.
The report labels proxy choices:

- Adderall is represented by d-amphetamine.
- bkMDMA is represented by methylone.
- bkMDMA DAT literature uses a rat DAT proxy compared with the local human DAT
  model target.
- D9THC is checked against CB1 and CB2.
- Morphine is checked against mu-opioid and delta-opioid targets.

## Result Summary

| Compound | Target | Literature mean pX | Predicted pIC50 | Delta | Fold error |
| --- | --- | ---: | ---: | ---: | ---: |
| Adderall / d-amphetamine | DAT | 7.0367 | 5.9340 | -1.1027 | 12.6678 |
| LSD | 5HT2A | 8.3186 | 6.7230 | -1.5956 | 39.4094 |
| delta-9-THC | CB1 | 7.9650 | 7.9280 | -0.0370 | 1.0889 |
| delta-9-THC | CB2 | 7.6800 | 6.4260 | -1.2540 | 17.9473 |
| Morphine | mu-opioid | 8.6300 | 6.3950 | -2.2350 | 171.7908 |
| Morphine | delta-opioid | 6.6850 | 6.1670 | -0.5180 | 3.2961 |
| Methylphenidate | DAT | 7.3250 | 6.1540 | -1.1710 | 14.8252 |
| bkMDMA / methylone | DAT | 6.8800 | 5.6640 | -1.2160 | 16.4437 |

## Verification

- RED: `tests/test_psychopharm_literature_check.py` failed before the script
  existed with `ModuleNotFoundError`.
- GREEN: `.venv\Scripts\python.exe -B -m pytest tests\test_psychopharm_literature_check.py -q`
  passed.
- Generated the comparison report with:
  `.venv\Scripts\python.exe -B scripts\run_psychopharm_literature_check.py --reference data\psychopharm_literature_reference.csv --model models\chembl_category_cpu_pic50_model.json --output artifacts\psychopharm_literature_prediction_check.json`

## Residual Risks

- This is a sanity check, not confirmatory validation.
- Endpoint classes are mixed across `Ki`, `IC50`, and `EC50`.
- Some classic-compound rows come from older assay literature even though the
  comparison is framed for current psychopharmacology review.
- Kappa-opioid prediction remains absent until the CHEMBL237 target cache is
  available in the checked category snapshot.
