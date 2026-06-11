# Research-Only QSAR Validation Plan

## Scope

This repository is a research-only QSAR validation prototype for retrospective
analysis of public DAT bioactivity data. It must not recommend, rank,
synthesize, optimize, dose, or support human use of psychoactive substances or
therapeutic candidates.

The current scientific value is not that a single model gives a final answer.
The value is that the pipeline can expose ChEMBL provenance, assay context,
split policy, model disagreement, and uncertainty in a reproducible form.

## Current Evidence

| Evidence item | Current value |
| --- | --- |
| Target snapshot | CHEMBL238 DAT |
| Aggregated rows | 4,374 rows from 4,769 measurements |
| Endpoints | pIC50 and pKi, kept separate |
| Duplicate policy | median aggregation by molecule, endpoint, and assay context |
| Inactive rule | standard values at or above 1000 uM may be labeled inactive |
| Outlier rule | dIQR-style endpoint outlier tracking in the manifest |
| CPU Ridge external RMSE | pIC50 0.9289, pKi 1.0937 |
| 4B-MAR CUDA deep50 consensus median | pIC50 6.1702, pKi 5.7715 |
| 4B-MAR pKi disagreement | ELT 4.2293 versus GNN 7.3700 |
| Best 4B-MAR pKi refit metric | GNN RMSE 0.7943 under the recorded split |

The pIC50 result is usable as an engineering stress test. The pKi result should
be reported as high-disagreement and should withhold decisive numeric
interpretation until local chemical-series validation and uncertainty
calibration improve.

## MVP Judgment

The current accuracy is acceptable for a research-only MVP demonstration, but it
is still weak for strong scientific interpretation. A scaffold or external RMSE
near 0.6 to 0.8 should be treated as a provisional target for a context-specific
DAT model. The current CPU baseline does not meet that bar. One deep pKi member
is near the bar, but its disagreement with the other models prevents strong
interpretation.

The near-term objective is therefore not to promote a best model. The objective
is to make the evidence harder to misunderstand: separate binding from uptake,
separate human from non-human species, report local-neighborhood support, and
return a withhold decision when uncertainty or assay context is unsupported.

## Required Gates

| Gate | Required artifact | Minimum requirement |
| --- | --- | --- |
| A. Dataset governance | `chembl238_dataset_card.md`, manifest JSON | ChEMBL release, target IDs, organisms, standard types, assay types, units, confidence score, duplicate policy, salt/parent handling, stereochemistry policy, excluded row counts |
| B. Endpoint taxonomy | endpoint taxonomy table | Separate `binding_Ki`, `binding_IC50`, and `uptake_IC50`; keep human, rat, and mixed-species datasets distinct |
| C. Validation suite | benchmark JSON and report table | Random, scaffold, fingerprint-cluster, document, assay, temporal if available, local-series, and species-holdout splits |
| D. Uncertainty and domain | prediction payload fields | `prediction_interval_90`, `ensemble_sd`, nearest-neighbor similarity, assay context support, species support, and withhold decision |
| E. Baseline-first modeling | model report | Mean/median, Tanimoto kNN, Ridge, random forest or ExtraTrees, boosted trees if installed, and local-neighborhood baselines before deep claims |

## Pipeline Stages

1. `chembl_fetch`
2. `chembl_freeze`
3. `standardize_molecules`
4. `build_endpoint_table`
5. `make_splits`
6. `train_baselines`
7. `train_deep`
8. `evaluate`
9. `calibrate_uncertainty`
10. `generate_report`

Each prediction should be traceable to model artifact hash, dataset snapshot
hash, split identifier, preprocessing configuration, RDKit version, ChEMBL
version, source activity rows, `assay_chembl_id`, `document_chembl_id`, and
`target_chembl_id`.

## Assay Context Policy

| Axis | Required handling |
| --- | --- |
| Endpoint | Do not pool IC50, Ki, EC50, and uptake readouts into one target variable |
| Modality | Separate binding from uptake before training context-specific models |
| Species | Prefer human-only primary models; report rat and mixed-species models as secondary |
| Cell system | Preserve assay cell type, tissue, and BAO format for filtering and reporting |
| Confidence score | Prefer high-confidence ChEMBL target mappings for promoted datasets |
| Duplicates | Aggregate duplicate measurements by robust median or robust mean within assay context |
| Outliers | Track dIQR outliers and excluded rows in the manifest |

## Candidate Panel Reporting

Candidate panels may include methylphenidate, amphetamine-like reference
structures, cocaine-like DAT references, phenethylamine scaffolds,
Betanamin/pemoline, aminorex, 4-MAR, 4,4-DMAR, and 4B-MAR only as retrospective
research references or validation stress tests.

The report should include:

- canonical SMILES and Murcko scaffold
- RDKit descriptor summary
- SMILES token sequence length
- molecular graph node and edge counts
- nearest training-neighbor similarity
- CPU baseline prediction
- Transformer, GNN, and ELT predictions when available
- consensus median, mean, standard deviation, and range
- explicit withhold status when disagreement is high

## LLM Reporting Guardrails

LLM-assisted reports must consume structured JSON artifacts only. They should
not accept free-form potency claims from the model runtime. The report generator
must refuse synthesis routes, dosing advice, human-use recommendations,
controlled-substance optimization guidance, or unsupported compound ranking.

Recommended safe repository sentence:

```text
This repository is a research-only QSAR validation prototype for retrospective
analysis of public DAT bioactivity data. It is not intended to recommend, rank,
synthesize, optimize, dose, or support human use of psychoactive substances or
therapeutic candidates.
```

## GitHub Release Checklist

- README uses research-only wording and avoids therapeutic or optimization
  claims.
- Generated large ChEMBL snapshots are not committed unless license and
  reproducibility review is complete.
- JSON artifacts include dataset hash, split policy, model family, seed,
  dependency versions, and command line.
- Tests cover snapshot aggregation, assay context preservation, CUDA request
  validation, and candidate-panel consensus fields.
- Issues track remaining gates rather than promoting exploratory predictions.

## Hugging Face Card Checklist

- Model card states research-only retrospective use.
- Dataset card lists ChEMBL release, target IDs, endpoint taxonomy, excluded
  rows, aggregation policy, and assay-context fields.
- Tags avoid therapeutic recommendation, lead optimization, and human-use
  framing.
- Evaluation tables separate CPU baselines, deep models, assay contexts,
  endpoint types, and external/scaffold/local-series splits.
- Limitations section states when numeric interpretation should be withheld.

## BioRender Figure Plan

Panel A: research-only boundary and prohibited uses.

Panel B: ChEMBL238 data-governance flow, including measurement count,
aggregation, exclusions, and split creation.

Panel C: assay stratification matrix for endpoint, binding/uptake modality,
species, cell system, and confidence score.

Panel D: model families: CPU baseline, descriptor Transformer, molecular graph
GNN, elastic-looped Transformer, and consensus layer.

Panel E: validation barriers: random, scaffold, cluster, document, assay,
species, and local-series splits.

Panel F: uncertainty gate with three outcomes: report exploratory result,
withhold numeric interpretation, or require expert review.

Suggested tables:

- dataset curation report
- endpoint taxonomy
- model validation matrix
- model card summary
- candidate-panel consensus and disagreement table
