# CHEMBL238 Dataset Card

## Intended Use

This card describes a research-only CHEMBL238 DAT endpoint snapshot for
retrospective QSAR validation. It is not a dataset for compound recommendation,
lead optimization, synthesis planning, dosing, human-use guidance, clinical
decision-making, regulatory submission, or manufacturing decisions.

## Source

| Field | Value |
| --- | --- |
| Primary source | ChEMBL public API/data |
| Target | CHEMBL238, dopamine transporter |
| Current local snapshot | `data/chembl238_endpoint_activity_snapshot.csv` |
| Current local manifest | `artifacts/chembl238_endpoint_activity_snapshot.manifest.json` |
| Endpoint values | pIC50 from IC50 and pKi from Ki, kept separate |
| Aggregation | Median by molecule, endpoint, and assay context |
| Current row count | 4,374 aggregated rows |
| Current measurement count | 4,769 source measurements |

Generated snapshots should be treated as local evaluation artifacts unless a
license, attribution, and reproducibility review approves redistribution.

## Curation Policy

| Rule | Current policy |
| --- | --- |
| Units | Normalize standard values in nM before p-value conversion |
| Endpoints | Keep IC50 and Ki separate as pIC50 and pKi |
| Inactive threshold | Values at or above 1000 uM may be treated as inactive for research triage |
| Outliers | Track dIQR-style outliers in the dataset manifest |
| Duplicate measurements | Aggregate with median or robust mean within assay context |
| Assay context | Preserve assay ID, assay type, organism, cell type, tissue, BAO format, and binding/uptake modality |
| Splits | Use train, scaffold_test, and external partitions; add document, assay, species, and local-series splits before promoted claims |

## Current Snapshot Summary

| Endpoint | Aggregated rows | Measurement count | External rows | Scaffold-test rows | Train rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| pIC50 | 2,299 | 2,540 | 304 | 225 | 1,770 |
| pKi | 2,075 | 2,229 | 245 | 189 | 1,641 |

## Known Limitations

- Binding and uptake assays must be trained and reported as separate contexts
  before model claims are promoted.
- Human, rat, and mixed-species evidence must be separated for primary
  interpretation.
- Local aminorex and phenethylamine neighborhoods remain sparse.
- The current CPU baseline external RMSE is above the preferred provisional
  target of 0.6 to 0.8 log units.
- Candidate predictions with high ensemble disagreement should return a
  withhold decision rather than a strong numeric interpretation.

## Recommended Hugging Face Summary

```text
Research-only CHEMBL238 DAT endpoint snapshot for retrospective QSAR validation.
The dataset keeps pIC50 and pKi separate, preserves assay-context metadata, and
records aggregation, split, outlier, and inactive-threshold policies. It is not
intended for compound recommendation, optimization, synthesis, dosing, or
human-use guidance.
```
