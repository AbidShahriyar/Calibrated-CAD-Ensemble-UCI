# Reproducible CAD analysis workflow

This folder contains the reproducible analysis of the four UCI heart-disease
cohorts. The original manuscript, notebook, and downloaded UCI files were
treated as read-only inputs and were not modified.

This folder is the executable code bundle copied into the Overleaf/repository
package. Frozen outputs used by the manuscript are under the project-level
`supplementary/` and `figures/` folders. A fresh run writes independently to
`analysis_code/outputs/`, allowing the archived results to remain unchanged.

## Scope

The workflow implements the following methodological checks:

- retain all four UCI cohorts and document the flow of records by source;
- treat invalid zero values as missing instead of silently deleting records;
- retain and impute `slope`, `thal`, and (in a sensitivity analysis) `ca`;
- fit imputation, encoding, and scaling only inside training folds;
- use stratified splitting with an explicit seed;
- report the full hyperparameter grids and their cross-validation results;
- compare classifiers using deterministic and probabilistic measures;
- select ensemble members using development-set cross-validation only;
- state explicitly that soft voting averages class probabilities;
- compare uncalibrated, sigmoid-calibrated, and isotonic-calibrated voting;
- report confidence intervals, paired McNemar tests, and paired AUROC bootstrap
  comparisons;
- create ROC, calibration, confusion-matrix, correlation, OOB, and
  partial-dependence figures;
- estimate model-agnostic Monte Carlo SHAP values for the final ensemble and
  export both a summary plot and numerical values for a reproducibly selected
  random evaluation case;
- quantify cohort heterogeneity with within-evaluation-set subgroup reporting
  and a leave-one-cohort-out transportability sensitivity analysis.
- compare Extra Trees, gradient boosting, and histogram gradient boosting with
  equal-probability averaging, constrained blending, and probability stacking.

The focused comparison preserves the non-invasive feature set. Hyperparameters,
ensemble membership, blend weights, stacking, and any operating-threshold
choice use development data only.

## Feature sets

1. `common10`: a reduced ten-predictor set with broad availability.
2. `noninvasive12`: `common10` plus `slope` and `thal`. This is the primary
   feature set.
3. `full13`: `noninvasive12` plus `ca`. This is a sensitivity
   analysis, not the proposed pre-angiography deployment model, because `ca`
   records vessels colored by fluoroscopy.

## Run

From the unzipped package root:

```bash
python3 -m pip install -r analysis_code/requirements.txt
MPLCONFIGDIR=/tmp/cad-mpl python3 analysis_code/run_all.py
```

Individual stages can also be run independently:

```bash
python3 analysis_code/scripts/01_prepare_data.py
python3 analysis_code/scripts/02_model_analysis.py
python3 analysis_code/scripts/03_generate_figures.py
python3 analysis_code/scripts/04_shap_analysis.py
python3 analysis_code/scripts/05_zero_deletion_sensitivity.py
python3 analysis_code/scripts/06_cohort_transport_sensitivity.py
python3 analysis_code/scripts/07_extended_models_and_ensembles.py
```

The complete workflow can take substantial time because it refits all grids,
bootstrap intervals, calibration protocols, LOCO folds, and Monte Carlo SHAP
calculations. The exact versions used for the reported analysis are pinned in
`requirements.txt`; the saved run metadata records the realized environment.

## Inputs

`raw_data/` contains the four downloaded UCI files actually read by the
workflow, along with the UCI documentation and warning file. Stage 1 verifies
the expected cohort row counts and writes SHA-256 checksums to the new run's
output folder before any modeling begins.

## Outputs

- `outputs/data/`: regenerated analytic dataset, cohort flow, missingness summaries,
  target distributions, source checksums, data dictionary, and reproducible
copies of the 740- and 661-record record-exclusion sensitivity subsets.
- `outputs/tables/`: full grid-search results, model-selection tables, hold-out
  metrics with 95% confidence intervals, statistical tests, feature
  importance, partial-dependence values, and Monte Carlo SHAP values.
- `outputs/figures/`: publication-ready figures.
- `outputs/models/`: fitted final models and selection metadata.
- `outputs/logs/`: execution logs and machine-readable run metadata.
- `outputs/extended/`: complete results, predictions, grids, statistical tests,
  fitted models, and metadata for the focused additional-model analysis.

The cohort-transport stage is a sensitivity analysis only. It reports performance
within each cohort in the original pooled evaluation set and then withholds each
cohort in turn. Cohort identity is never added to the clinical predictor set.
The stage writes its protocol, key results, and integrity checks to
`outputs/logs/cohort_transport_metadata.json`.

## Reproducibility note

The downloaded UCI documentation requests that publications acknowledge the
principal investigators responsible for data collection: Andras Janosi,
William Steinbrunn, Matthias Pfisterer, and Robert Detrano.
