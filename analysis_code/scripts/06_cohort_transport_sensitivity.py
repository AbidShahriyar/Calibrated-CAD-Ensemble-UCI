"""Evaluate cohort-specific performance and leave-one-cohort-out transport.

This sensitivity analysis complements the primary pooled study by asking two
narrower questions:

1. How does the locked primary ensemble perform within each source cohort in
   the existing untouched evaluation partition?
2. How well does the same three-family ensemble protocol transport when every
   record from one cohort is withheld from model fitting and used only for
   testing?

Cohort identity is provenance, not a clinical predictor. It is deliberately
excluded from the feature matrix because it can encode referral prevalence,
measurement practice, and other site-specific shortcuts. In each leave-one-
cohort-out fold, hyperparameters and probability calibration are selected using
only the other three cohorts and the same grids used in the primary workflow.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cad-analysis-matplotlib")

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from pipeline_utils import (
    FEATURE_SETS,
    FittedProbabilityEnsemble,
    OUTPUT_ROOT,
    RANDOM_SEED,
    ROW_ID,
    SOURCE,
    TARGET,
    add_confidence_intervals,
    classification_metrics,
    make_model_pipeline,
    positive_probability,
    stratified_bootstrap_intervals,
)


PRIMARY_FEATURE_SET = "noninvasive12"
FEATURES = FEATURE_SETS[PRIMARY_FEATURE_SET]
N_BOOTSTRAP = 2_000
COHORT_ORDER = ["cleveland", "hungarian", "switzerland", "va_long_beach"]
COHORT_LABELS = {
    "cleveland": "Cleveland",
    "hungarian": "Hungarian",
    "switzerland": "Switzerland",
    "va_long_beach": "VA Long Beach",
}
COMPONENT_ORDER = ["knn", "logistic_regression", "svc"]


warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def component_specifications() -> dict[str, tuple[object, dict[str, list[object]]]]:
    """Return only the three model families in the primary ensemble."""

    return {
        "knn": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [5, 10, 20, 30, 50],
                "model__p": [1, 2],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "logistic_regression": (
            LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
            {
                "model__C": [0.001, 0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["liblinear"],
            },
        ),
        "svc": (
            SVC(probability=True, random_state=RANDOM_SEED),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["linear", "rbf"],
            },
        ),
    }


def calibration_oof_probability(
    estimator: object,
    method: str,
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: StratifiedKFold,
    seed_offset: int,
) -> np.ndarray:
    """Create training-only OOF probabilities for calibration selection."""

    if method == "none":
        model = clone(estimator)
    else:
        inner_cv = StratifiedKFold(
            n_splits=3,
            shuffle=True,
            random_state=RANDOM_SEED + 100 + seed_offset,
        )
        model = CalibratedClassifierCV(
            estimator=clone(estimator), method=method, cv=inner_cv
        )
    return cross_val_predict(
        model,
        X,
        y,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def calibration_selection_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Return the development measures needed to select calibration."""

    prediction = (probability >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "log_loss": float(
            log_loss(y, np.column_stack([1.0 - probability, probability]), labels=[0, 1])
        ),
    }


def metric_row(
    y: np.ndarray,
    probability: np.ndarray,
    random_state: int,
) -> dict[str, float]:
    """Calculate estimates and stratified-bootstrap intervals."""

    estimates = classification_metrics(y, probability)
    intervals = stratified_bootstrap_intervals(
        y,
        probability,
        n_bootstrap=N_BOOTSTRAP,
        random_state=random_state,
    )
    return add_confidence_intervals(estimates, intervals)


def existing_evaluation_by_cohort() -> pd.DataFrame:
    """Evaluate the already locked primary model inside each cohort subgroup."""

    predictions = pd.read_csv(
        OUTPUT_ROOT / "data" / f"evaluation_predictions__{PRIMARY_FEATURE_SET}.csv"
    )
    model_columns = {
        "knn": "knn__probability",
        "logistic_regression": "logistic_regression__probability",
        "svc": "svc__probability",
        "soft_voting_sigmoid": "soft_voting_sigmoid__probability",
    }
    rows: list[dict[str, object]] = []
    for cohort_offset, cohort in enumerate(COHORT_ORDER):
        subset = predictions.loc[predictions[SOURCE] == cohort].copy()
        y = subset[TARGET].to_numpy(dtype=int)
        for model_offset, (model_name, probability_column) in enumerate(model_columns.items()):
            row = metric_row(
                y,
                subset[probability_column].to_numpy(dtype=float),
                RANDOM_SEED + 1_000 + 10 * cohort_offset + model_offset,
            )
            row.update(
                {
                    "validation_scheme": "pooled_split_cohort_subgroup",
                    "cohort": cohort,
                    "model": model_name,
                    "target_0": int((y == 0).sum()),
                    "target_1": int((y == 1).sum()),
                }
            )
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(
        OUTPUT_ROOT / "tables" / "primary_evaluation_metrics_by_cohort.csv",
        index=False,
    )
    return result


def leave_one_cohort_out(analytic: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the primary ensemble protocol on three cohorts and test on the fourth."""

    table_root = OUTPUT_ROOT / "tables"
    grid_root = table_root / "grid_search" / "loco"
    model_root = OUTPUT_ROOT / "models" / "loco"
    grid_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    calibration_rows: list[dict[str, object]] = []
    selection_rows: list[dict[str, object]] = []
    specifications = component_specifications()

    for fold_offset, held_out in enumerate(COHORT_ORDER):
        print(f"\nLOCO held-out cohort: {held_out}", flush=True)
        train = analytic.loc[analytic[SOURCE] != held_out].copy()
        test = analytic.loc[analytic[SOURCE] == held_out].copy()
        X_train = train[FEATURES]
        y_train = train[TARGET].astype(int)
        X_test = test[FEATURES]
        y_test = test[TARGET].astype(int).to_numpy()
        tuning_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=RANDOM_SEED + fold_offset,
        )

        best_estimators: dict[str, object] = {}
        for component in COMPONENT_ORDER:
            estimator, grid = specifications[component]
            search = GridSearchCV(
                estimator=make_model_pipeline(FEATURES, estimator),
                param_grid=grid,
                scoring="roc_auc",
                refit=True,
                cv=tuning_cv,
                n_jobs=1,
                return_train_score=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            best_estimators[component] = clone(search.best_estimator_)
            search_results = pd.DataFrame(search.cv_results_)
            search_results.insert(0, "component", component)
            search_results.insert(0, "held_out_cohort", held_out)
            search_results.to_csv(
                grid_root / f"heldout_{held_out}__{component}.csv", index=False
            )
            selection_rows.append(
                {
                    "held_out_cohort": held_out,
                    "component": component,
                    "training_n": len(train),
                    "test_n": len(test),
                    "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                    "best_mean_training_cv_roc_auc": float(search.best_score_),
                }
            )

        calibration_probabilities: dict[str, dict[str, np.ndarray]] = {}
        for method_offset, method in enumerate(["none", "sigmoid", "isotonic"]):
            component_oof = {}
            for component_offset, component in enumerate(COMPONENT_ORDER):
                component_oof[component] = calibration_oof_probability(
                    best_estimators[component],
                    method,
                    X_train,
                    y_train,
                    tuning_cv,
                    seed_offset=100 * fold_offset + 10 * method_offset + component_offset,
                )
            calibration_probabilities[method] = component_oof
            ensemble_probability = np.column_stack(
                [component_oof[component] for component in COMPONENT_ORDER]
            ).mean(axis=1)
            calibration_rows.append(
                {
                    "held_out_cohort": held_out,
                    "method": method,
                    "components": ";".join(COMPONENT_ORDER),
                    **calibration_selection_metrics(
                        y_train.to_numpy(), ensemble_probability
                    ),
                }
            )

        cohort_calibration = pd.DataFrame(
            [row for row in calibration_rows if row["held_out_cohort"] == held_out]
        ).sort_values(["brier", "log_loss", "roc_auc"], ascending=[True, True, False])
        selected_method = str(cohort_calibration.iloc[0]["method"])
        print(f"Selected training-only calibration: {selected_method}", flush=True)

        fitted_components = []
        component_probabilities: dict[str, np.ndarray] = {}
        for component_offset, component in enumerate(COMPONENT_ORDER):
            base = clone(best_estimators[component])
            if selected_method == "none":
                final_component = base.fit(X_train, y_train)
            else:
                calibration_cv = StratifiedKFold(
                    n_splits=5,
                    shuffle=True,
                    random_state=RANDOM_SEED + 500 + 10 * fold_offset + component_offset,
                )
                final_component = CalibratedClassifierCV(
                    estimator=base,
                    method=selected_method,
                    cv=calibration_cv,
                ).fit(X_train, y_train)
            fitted_components.append(final_component)
            component_probabilities[component] = positive_probability(
                final_component, X_test
            )

        ensemble = FittedProbabilityEnsemble(tuple(fitted_components))
        ensemble_probability = ensemble.predict_proba(X_test)[:, 1]
        probability_by_model = {
            **component_probabilities,
            "soft_voting": ensemble_probability,
        }
        joblib.dump(ensemble, model_root / f"heldout_{held_out}__soft_voting.joblib")
        (model_root / f"heldout_{held_out}__metadata.json").write_text(
            json.dumps(
                {
                    "held_out_cohort": held_out,
                    "training_cohorts": [
                        cohort for cohort in COHORT_ORDER if cohort != held_out
                    ],
                    "features": FEATURES,
                    "cohort_used_as_predictor": False,
                    "components": COMPONENT_ORDER,
                    "selected_calibration": selected_method,
                    "selection_scope": "training cohorts only",
                },
                indent=2,
            )
            + "\n"
        )

        fold_predictions = test[[ROW_ID, SOURCE, TARGET]].copy()
        fold_predictions["held_out_cohort"] = held_out
        fold_predictions["selected_calibration"] = selected_method
        for model_name, probability in probability_by_model.items():
            fold_predictions[f"{model_name}__probability"] = probability
            fold_predictions[f"{model_name}__prediction"] = (
                probability >= 0.5
            ).astype(int)
            row = metric_row(
                y_test,
                probability,
                RANDOM_SEED + 2_000 + 10 * fold_offset + list(probability_by_model).index(model_name),
            )
            row.update(
                {
                    "validation_scheme": "leave_one_cohort_out",
                    "cohort": held_out,
                    "model": model_name,
                    "training_n": len(train),
                    "training_target_0": int((y_train == 0).sum()),
                    "training_target_1": int((y_train == 1).sum()),
                    "target_0": int((y_test == 0).sum()),
                    "target_1": int((y_test == 1).sum()),
                    "selected_calibration": selected_method,
                }
            )
            metric_rows.append(row)
        prediction_frames.append(fold_predictions)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(
        OUTPUT_ROOT / "data" / "leave_one_cohort_out_predictions.csv", index=False
    )

    for model_offset, model_name in enumerate([*COMPONENT_ORDER, "soft_voting"]):
        probability = predictions[f"{model_name}__probability"].to_numpy(dtype=float)
        y = predictions[TARGET].to_numpy(dtype=int)
        row = metric_row(y, probability, RANDOM_SEED + 3_000 + model_offset)
        row.update(
            {
                "validation_scheme": "leave_one_cohort_out_pooled_predictions",
                "cohort": "all",
                "model": model_name,
                "training_n": np.nan,
                "training_target_0": np.nan,
                "training_target_1": np.nan,
                "target_0": int((y == 0).sum()),
                "target_1": int((y == 1).sum()),
                "selected_calibration": "fold_specific_training_only",
            }
        )
        metric_rows.append(row)

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(table_root / "leave_one_cohort_out_metrics.csv", index=False)
    pd.DataFrame(selection_rows).to_csv(
        table_root / "leave_one_cohort_out_model_selection.csv", index=False
    )
    pd.DataFrame(calibration_rows).to_csv(
        table_root / "leave_one_cohort_out_calibration_selection.csv", index=False
    )
    return predictions, metrics


def cohort_comparison_figure(
    existing: pd.DataFrame,
    loco_metrics: pd.DataFrame,
) -> None:
    """Compare within-mixture subgroup and transportability performance."""

    existing_ensemble = existing.loc[existing["model"] == "soft_voting_sigmoid"].copy()
    loco_ensemble = loco_metrics.loc[
        (loco_metrics["model"] == "soft_voting")
        & (loco_metrics["cohort"] != "all")
    ].copy()
    existing_ensemble["scheme_label"] = "Pooled split: cohort subgroup"
    loco_ensemble["scheme_label"] = "Leave-one-cohort-out"
    combined = pd.concat([existing_ensemble, loco_ensemble], ignore_index=True)
    combined.to_csv(
        OUTPUT_ROOT / "tables" / "cohort_sensitivity_ensemble_summary.csv",
        index=False,
    )

    colors = {
        "Pooled split: cohort subgroup": "#4C78A8",
        "Leave-one-cohort-out": "#E17C47",
    }
    offsets = {
        "Pooled split: cohort subgroup": -0.10,
        "Leave-one-cohort-out": 0.10,
    }
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.6), sharex=True)
    for ax, metric, title in [
        (axes[0], "balanced_accuracy", "Balanced accuracy"),
        (axes[1], "roc_auc", "AUROC"),
    ]:
        for scheme in colors:
            subset = combined.loc[combined["scheme_label"] == scheme].set_index("cohort")
            subset = subset.loc[COHORT_ORDER]
            estimate = subset[metric].to_numpy(dtype=float)
            lower = subset[f"{metric}_ci_low"].to_numpy(dtype=float)
            upper = subset[f"{metric}_ci_high"].to_numpy(dtype=float)
            x = np.arange(len(COHORT_ORDER)) + offsets[scheme]
            ax.errorbar(
                x,
                estimate,
                yerr=np.vstack([estimate - lower, upper - estimate]),
                fmt="o",
                color=colors[scheme],
                capsize=3,
                label=scheme,
            )
        ax.set_xticks(
            np.arange(len(COHORT_ORDER)),
            [COHORT_LABELS[cohort] for cohort in COHORT_ORDER],
            rotation=20,
            ha="right",
        )
        ax.set_ylim(0.0, 1.02)
        ax.set_ylabel(f"{title} (95% bootstrap CI)")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Cohort sensitivity of the primary calibrated soft-voting protocol")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_ROOT / "figures" / "figure_cohort_transportability.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)


def validate_and_record_outputs(
    analytic: pd.DataFrame,
    existing: pd.DataFrame,
    predictions: pd.DataFrame,
    loco_metrics: pd.DataFrame,
) -> None:
    """Validate saved cohort outputs and write reproducibility metadata.

    These checks make the separation between provenance and predictors explicit,
    verify that every source record is tested exactly once in LOCO validation,
    and guard against malformed probability or confusion-matrix outputs.
    """

    probability_columns = [
        f"{model}__probability" for model in [*COMPONENT_ORDER, "soft_voting"]
    ]
    cohort_summary = pd.read_csv(
        OUTPUT_ROOT / "tables" / "cohort_sensitivity_ensemble_summary.csv"
    )
    checks = {
        "cohort_excluded_from_predictors": SOURCE not in FEATURES,
        "all_920_records_have_loco_predictions": len(predictions) == len(analytic) == 920,
        "row_ids_are_unique": predictions[ROW_ID].nunique() == len(predictions),
        "held_out_cohort_matches_source": (
            predictions["held_out_cohort"] == predictions[SOURCE]
        ).all(),
        "four_source_cohorts_present": predictions[SOURCE].nunique() == 4,
        "all_probabilities_are_finite": all(
            np.isfinite(predictions[column]).all() for column in probability_columns
        ),
        "all_probabilities_are_in_unit_interval": all(
            predictions[column].between(0.0, 1.0).all()
            for column in probability_columns
        ),
        "loco_confusion_matrices_sum_to_n": loco_metrics[
            ["tn", "fp", "fn", "tp"]
        ]
        .sum(axis=1)
        .eq(loco_metrics["n"])
        .all(),
        "subgroup_confusion_matrices_sum_to_n": existing[
            ["tn", "fp", "fn", "tp"]
        ]
        .sum(axis=1)
        .eq(existing["n"])
        .all(),
        "four_models_evaluated_per_held_out_cohort": loco_metrics.loc[
            loco_metrics["cohort"] != "all"
        ]
        .groupby("cohort")["model"]
        .nunique()
        .eq(4)
        .all(),
        "both_designs_cover_all_four_cohorts": cohort_summary.groupby(
            "scheme_label"
        )["cohort"]
        .nunique()
        .eq(4)
        .all(),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"Cohort-output validation failed: {failed}")

    ensemble_rows = loco_metrics.loc[
        loco_metrics["model"].eq("soft_voting"),
        [
            "cohort",
            "n",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "roc_auc",
            "brier",
            "selected_calibration",
        ],
    ]
    ensemble_results = []
    for row in ensemble_rows.to_dict(orient="records"):
        ensemble_results.append(
            {
                "cohort": str(row["cohort"]),
                "n": int(row["n"]),
                "balanced_accuracy": float(row["balanced_accuracy"]),
                "sensitivity": float(row["sensitivity"]),
                "specificity": float(row["specificity"]),
                "roc_auc": float(row["roc_auc"]),
                "brier": float(row["brier"]),
                "selected_calibration": str(row["selected_calibration"]),
            }
        )

    log_root = OUTPUT_ROOT / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "cohort_transport_metadata.json").write_text(
        json.dumps(
            {
                "analysis_role": "sensitivity analysis; primary pooled study unchanged",
                "feature_set": PRIMARY_FEATURE_SET,
                "features": FEATURES,
                "cohort_used_as_predictor": False,
                "component_families": COMPONENT_ORDER,
                "bootstrap_replicates": N_BOOTSTRAP,
                "selection_scope": "training cohorts only within each LOCO fold",
                "validation_checks": checks,
                "loco_ensemble_results": ensemble_results,
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    analytic = pd.read_csv(OUTPUT_ROOT / "data" / "analytic_920.csv")
    if set(analytic[SOURCE]) != set(COHORT_ORDER):
        raise ValueError("Unexpected source-cohort labels in analytic data")
    existing = existing_evaluation_by_cohort()
    predictions, loco_metrics = leave_one_cohort_out(analytic)
    cohort_comparison_figure(existing, loco_metrics)
    validate_and_record_outputs(analytic, existing, predictions, loco_metrics)

    ensemble = loco_metrics.loc[
        (loco_metrics["model"] == "soft_voting")
        & (loco_metrics["cohort"] != "all"),
        [
            "cohort",
            "n",
            "target_0",
            "target_1",
            "balanced_accuracy",
            "roc_auc",
            "brier",
            "selected_calibration",
        ],
    ]
    print("\nLeave-one-cohort-out ensemble results")
    print(ensemble.to_string(index=False))


if __name__ == "__main__":
    main()
