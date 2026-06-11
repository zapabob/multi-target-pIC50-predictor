# BioRender Figure Plan: CHEMBL238 Research-Only QSAR Workflow

## Figure Title

Research-only assay-context-aware QSAR validation workflow for public DAT
bioactivity data.

## Panel Layout

| Panel | Content | Visual elements |
| --- | --- | --- |
| A | Research-only scope | Boundary box with allowed retrospective analysis and prohibited recommendation, synthesis, optimization, dosing, and human-use guidance |
| B | Data governance | ChEMBL238 source measurements, standardization, duplicate aggregation, excluded rows, and frozen snapshot manifest |
| C | Assay stratification | Matrix for pIC50, pKi, binding, uptake, human, rat, mixed species, cell system, tissue, BAO format, and confidence score |
| D | Model families | CPU Ridge, descriptor Transformer, molecular graph GNN, elastic-looped Transformer, and consensus layer |
| E | Validation barriers | Random, scaffold, cluster, document, assay, species, temporal if available, and local-series splits |
| F | Uncertainty gate | Three outcomes: exploratory report, withhold numeric interpretation, expert review required |

## Callout Text

- 4,374 aggregated rows from 4,769 CHEMBL238 measurements.
- Endpoint-aware modeling keeps pIC50 and pKi separate.
- Binding and uptake evidence must not be silently pooled.
- 4B-MAR pKi shows high member-model disagreement.
- Numeric interpretation is withheld when assay context, domain, or uncertainty
  is unsupported.

## Tables To Pair With The Figure

1. Dataset curation report.
2. Endpoint taxonomy and assay-context matrix.
3. Model validation matrix by split and endpoint.
4. Candidate-panel consensus and disagreement table.
5. Model-card summary with limitations and intended use.
