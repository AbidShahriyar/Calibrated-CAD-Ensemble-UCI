# Supplementary materials

## Study

**Prediction of Coronary Artery Disease from Non-Invasive Test Reports and
Demographic Characteristics Using Calibrated Ensemble Learning**

This archive contains the figures, executable analysis code, exact input and
analytic data, row-level predictions, complete cross-validation results,
statistical comparisons, explanation values, and run metadata supporting the
manuscript. It contains only analysis artifacts; authoring correspondence,
working provenance, and temporary build files are excluded.

## Contents

- `figures/`: publication-ready PNG figures used by the manuscript and its
  supporting analyses.
- `analysis_code/`: self-contained Python workflow, pinned requirements, four
  UCI input files, UCI documentation, and the seven ordered analysis stages.
- `data/`: the 920-record analytic dataset, development/evaluation assignment,
  data dictionary, missingness and invalid-value audits, 740- and 661-record
  record-exclusion sensitivity datasets, cohort flow, row-level predictions,
  and raw-input checksums.
- `tables/`: development rankings, evaluation metrics, confidence intervals,
  calibration selection, paired McNemar and AUROC tests, feature-set and
  deletion sensitivities, cohort results, partial-dependence values,
  permutation importance, and SHAP outputs.
- `grid_search/`: every tested parameter combination and cross-validation score
  for the main model families and feature sets.
- `extended/`: the focused comparison of Extra Trees, gradient boosting,
  histogram gradient boosting, equal averaging, constrained probability
  blending, and probability stacking, including full grids, development
  selection, evaluation probabilities, statistical tests, and metadata.
- `logs/`: machine-readable seeds, selected protocols, software versions, and
  data, SHAP, and cohort-transport metadata.
- `FILE_MANIFEST_SHA256.txt`: SHA-256 digest for every file in this archive.

## Reproducing the workflow

From the unzipped archive root:

```bash
python3 -m pip install -r analysis_code/requirements.txt
MPLCONFIGDIR=/tmp/cad-mpl python3 analysis_code/run_all.py
```

The complete workflow is computationally intensive because it reruns model
grids, nested calibration, bootstrap confidence intervals, leave-one-cohort-out
analysis, permutation importance, partial dependence, and Monte Carlo SHAP.
A fresh run writes to `analysis_code/outputs/`; the frozen manuscript results in
the other folders remain unchanged.

## Analysis conventions

- Primary population: all 920 records from Cleveland, Hungarian, Switzerland,
  and VA Long Beach.
- Primary feature set: 12 non-invasive variables (`noninvasive12`).
- Primary evaluation: stratified 70/30 split with random seed 10, comprising
  644 development and 276 evaluation records.
- Learned imputation, encoding, scaling, hyperparameter tuning, ensemble
  selection, and calibration are confined to development/training data.
- Cohort identity and stable row identifiers are retained for provenance and
  transport analysis but are not supplied as predictors.
- Leave-one-cohort-out results are transportability sensitivity analyses.

## Data acknowledgment

The source files were downloaded from the UCI Machine Learning Repository Heart
Disease dataset. In accordance with the accompanying UCI documentation,
publications using these data should acknowledge the principal investigators:
Andras Janosi, William Steinbrunn, Matthias Pfisterer, and Robert Detrano.

The files contain no patient names or original identifiers. The `row_id` field
in derived CSV files is a study-generated stable audit identifier.
