"""Train, select, calibrate, ensemble, and evaluate the candidate models.

Model and ensemble choices are made only from the development partition. The
stratified evaluation partition is opened once after the feature set, component
models, hyperparameters, and ensemble calibration method have been selected.
"""

from __future__ import annotations

import itertools
import json
import platform
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

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
    ensure_output_directories,
    holm_adjust,
    make_model_pipeline,
    positive_probability,
    stratified_bootstrap_intervals,
)


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

N_BOOTSTRAP_CI = 2000
N_BOOTSTRAP_AUC_DIFFERENCE = 5000


@dataclass(frozen=True)
class ModelSpec:
    estimator: BaseEstimator
    grid: dict[str, list[object]]
    ensemble_candidate: bool = True


def model_specs() -> dict[str, ModelSpec]:
    """Return the paper's original model families and requested ANN ablations."""

    return {
        "logistic_regression": ModelSpec(
            LogisticRegression(max_iter=5000, random_state=RANDOM_SEED),
            {
                "model__C": [0.001, 0.01, 0.1, 1.0, 10.0],
                "model__penalty": ["l2"],
                "model__solver": ["liblinear"],
            },
        ),
        "knn": ModelSpec(
            KNeighborsClassifier(),
            {
                "model__n_neighbors": [5, 10, 20, 30, 50],
                "model__p": [1, 2],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "gaussian_nb": ModelSpec(
            GaussianNB(),
            {
                "model__var_smoothing": [
                    1e-11,
                    1e-10,
                    1e-9,
                    1e-8,
                    1e-7,
                    1e-6,
                ]
            },
        ),
        "svc": ModelSpec(
            SVC(probability=True, random_state=RANDOM_SEED),
            {
                "model__C": [0.01, 0.1, 1.0, 10.0],
                "model__gamma": ["scale", "auto"],
                "model__kernel": ["linear", "rbf"],
            },
        ),
        "random_forest": ModelSpec(
            RandomForestClassifier(random_state=RANDOM_SEED, n_jobs=-1),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [4, 8, None],
                "model__max_features": ["sqrt", "log2"],
                "model__n_estimators": [100, 300, 700],
            },
        ),
        "decision_tree": ModelSpec(
            DecisionTreeClassifier(random_state=RANDOM_SEED),
            {
                "model__criterion": ["gini", "entropy"],
                "model__max_depth": [3, 5, 7, None],
                "model__max_features": [None, "sqrt", "log2"],
                "model__splitter": ["best", "random"],
            },
        ),
        "ann_shallow": ModelSpec(
            MLPClassifier(
                hidden_layer_sizes=(12,),
                activation="relu",
                solver="adam",
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=25,
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
            {
                "model__alpha": [0.0001, 0.001],
                "model__learning_rate_init": [0.001],
            },
            ensemble_candidate=False,
        ),
        "ann_original_12_6_3": ModelSpec(
            MLPClassifier(
                hidden_layer_sizes=(12, 6, 3),
                activation="relu",
                solver="adam",
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=25,
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
            {
                "model__alpha": [0.0001, 0.001],
                "model__learning_rate_init": [0.001],
            },
        ),
        "ann_deep": ModelSpec(
            MLPClassifier(
                hidden_layer_sizes=(24, 12, 6, 3),
                activation="relu",
                solver="adam",
                early_stopping=True,
                validation_fraction=0.2,
                n_iter_no_change=25,
                max_iter=1000,
                random_state=RANDOM_SEED,
            ),
            {
                "model__alpha": [0.0001, 0.001],
                "model__learning_rate_init": [0.001],
            },
            ensemble_candidate=False,
        ),
    }


def development_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    metrics = classification_metrics(y_true, probability)
    return {
        key: metrics[key]
        for key in [
            "accuracy",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "precision",
            "npv",
            "f1",
            "mcc",
            "roc_auc",
            "average_precision",
            "brier",
            "log_loss",
        ]
    }


def rank_models(metrics: pd.DataFrame, candidates: set[str]) -> pd.DataFrame:
    """Rank candidates equally across deterministic and probabilistic measures."""

    ranked = metrics.copy()
    higher_is_better = [
        "roc_auc",
        "average_precision",
        "balanced_accuracy",
        "f1",
        "sensitivity",
    ]
    lower_is_better = ["brier", "log_loss"]
    rank_columns = []
    eligible = ranked["model"].isin(candidates)
    for metric in higher_is_better:
        column = f"rank_{metric}"
        ranked.loc[eligible, column] = ranked.loc[eligible, metric].rank(
            ascending=False, method="average"
        )
        rank_columns.append(column)
    for metric in lower_is_better:
        column = f"rank_{metric}"
        ranked.loc[eligible, column] = ranked.loc[eligible, metric].rank(
            ascending=True, method="average"
        )
        rank_columns.append(column)
    ranked.loc[eligible, "composite_rank"] = ranked.loc[eligible, rank_columns].mean(axis=1)
    ranked.loc[~eligible, "composite_rank"] = np.nan
    return ranked.sort_values(["composite_rank", "roc_auc"], ascending=[True, False])


def paired_mcnemar_table(
    y_true: np.ndarray, predictions: dict[str, np.ndarray]
) -> pd.DataFrame:
    rows = []
    for model_a, model_b in itertools.combinations(sorted(predictions), 2):
        correct_a = predictions[model_a] == y_true
        correct_b = predictions[model_b] == y_true
        a_only = int(np.sum(correct_a & ~correct_b))
        b_only = int(np.sum(~correct_a & correct_b))
        discordant = a_only + b_only
        p_value = 1.0 if discordant == 0 else binomtest(min(a_only, b_only), discordant, 0.5).pvalue
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "a_correct_b_wrong": a_only,
                "a_wrong_b_correct": b_only,
                "discordant_pairs": discordant,
                "exact_mcnemar_p": p_value,
            }
        )
    table = pd.DataFrame(rows)
    table["holm_adjusted_p"] = holm_adjust(table["exact_mcnemar_p"])
    return table


def paired_auc_bootstrap(
    y_true: np.ndarray,
    probability_a: np.ndarray,
    probability_b: np.ndarray,
    model_a: str,
    model_b: str,
) -> dict[str, float | str]:
    """Paired stratified bootstrap for AUROC(A)-AUROC(B)."""

    y_true = np.asarray(y_true, dtype=int)
    zero = np.flatnonzero(y_true == 0)
    one = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(RANDOM_SEED)
    differences = []
    for _ in range(N_BOOTSTRAP_AUC_DIFFERENCE):
        index = np.concatenate(
            [
                rng.choice(zero, size=len(zero), replace=True),
                rng.choice(one, size=len(one), replace=True),
            ]
        )
        differences.append(
            roc_auc_score(y_true[index], probability_a[index])
            - roc_auc_score(y_true[index], probability_b[index])
        )
    differences = np.asarray(differences)
    observed = roc_auc_score(y_true, probability_a) - roc_auc_score(y_true, probability_b)
    p_value = 2.0 * min(np.mean(differences <= 0), np.mean(differences >= 0))
    return {
        "model_a": model_a,
        "model_b": model_b,
        "auc_difference_a_minus_b": observed,
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
        "two_sided_bootstrap_p": min(1.0, float(p_value)),
        "bootstrap_replicates": N_BOOTSTRAP_AUC_DIFFERENCE,
    }


def calibration_oof_probability(
    estimator: BaseEstimator,
    method: str,
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: StratifiedKFold,
) -> np.ndarray:
    if method == "none":
        model = clone(estimator)
    else:
        inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED + 1)
        model = CalibratedClassifierCV(estimator=clone(estimator), method=method, cv=inner_cv)
    return cross_val_predict(
        model,
        X,
        y,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def analyze_feature_set(
    feature_set_name: str,
    features: list[str],
    analytic: pd.DataFrame,
    development_index: np.ndarray,
    evaluation_index: np.ndarray,
) -> dict[str, object]:
    print(f"\n--- Feature set: {feature_set_name} ({len(features)} predictors) ---", flush=True)
    table_root = OUTPUT_ROOT / "tables"
    grid_root = table_root / "grid_search"
    model_root = OUTPUT_ROOT / "models"
    data_root = OUTPUT_ROOT / "data"

    X = analytic[features]
    y = analytic[TARGET].astype(int)
    X_development = X.iloc[development_index].copy()
    y_development = y.iloc[development_index].copy()
    X_evaluation = X.iloc[evaluation_index].copy()
    y_evaluation = y.iloc[evaluation_index].copy()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "f1": "f1",
        "sensitivity": "recall",
        "neg_brier": "neg_brier_score",
        "neg_log_loss": "neg_log_loss",
    }

    fitted_models: dict[str, BaseEstimator] = {}
    best_estimators: dict[str, BaseEstimator] = {}
    oof_probabilities: dict[str, np.ndarray] = {}
    selection_rows = []
    specifications = model_specs()

    for model_name, spec in specifications.items():
        print(f"Tuning {model_name}...", flush=True)
        pipeline = make_model_pipeline(features, spec.estimator)
        search = GridSearchCV(
            estimator=pipeline,
            param_grid=spec.grid,
            scoring=scoring,
            refit="roc_auc",
            cv=cv,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_development, y_development)
        results = pd.DataFrame(search.cv_results_)
        results.insert(0, "model", model_name)
        results.insert(0, "feature_set", feature_set_name)
        results.to_csv(grid_root / f"{feature_set_name}__{model_name}.csv", index=False)

        best = search.best_estimator_
        best_estimators[model_name] = clone(best)
        fitted_models[model_name] = best
        probability = cross_val_predict(
            clone(best),
            X_development,
            y_development,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        oof_probabilities[model_name] = probability
        row = {
            "feature_set": feature_set_name,
            "model": model_name,
            "best_parameters": json.dumps(search.best_params_, sort_keys=True),
            "grid_refit_metric": "roc_auc",
            "best_mean_cv_roc_auc": float(search.best_score_),
            **development_metrics(y_development.to_numpy(), probability),
        }
        selection_rows.append(row)
        joblib.dump(best, model_root / f"{feature_set_name}__{model_name}.joblib")

    selection = pd.DataFrame(selection_rows)
    ensemble_candidates = {
        name for name, spec in specifications.items() if spec.ensemble_candidate
    }
    ranked = rank_models(selection, ensemble_candidates)
    ranked.to_csv(table_root / f"development_model_ranking__{feature_set_name}.csv", index=False)
    selected_components = ranked.loc[ranked["model"].isin(ensemble_candidates), "model"].head(3).tolist()
    print(f"Selected ensemble components: {selected_components}", flush=True)

    calibration_rows = []
    calibration_component_oof: dict[str, dict[str, np.ndarray]] = {}
    for method in ["none", "sigmoid", "isotonic"]:
        component_probabilities = {}
        for component in selected_components:
            component_probabilities[component] = calibration_oof_probability(
                best_estimators[component],
                method,
                X_development,
                y_development,
                cv,
            )
        calibration_component_oof[method] = component_probabilities
        ensemble_probability = np.column_stack(list(component_probabilities.values())).mean(axis=1)
        metrics = development_metrics(y_development.to_numpy(), ensemble_probability)
        calibration_rows.append(
            {
                "feature_set": feature_set_name,
                "calibration_method": method,
                "components": ";".join(selected_components),
                **metrics,
            }
        )
    calibration_table = pd.DataFrame(calibration_rows).sort_values(
        ["brier", "log_loss", "roc_auc"], ascending=[True, True, False]
    )
    calibration_table.to_csv(
        table_root / f"ensemble_calibration_selection__{feature_set_name}.csv", index=False
    )
    selected_calibration = str(calibration_table.iloc[0]["calibration_method"])
    print(f"Selected calibration: {selected_calibration}", flush=True)

    # The held-out evaluation partition is used only from this point onward.
    evaluation_probabilities: dict[str, np.ndarray] = {}
    evaluation_predictions: dict[str, np.ndarray] = {}
    for model_name, model in fitted_models.items():
        probability = positive_probability(model, X_evaluation)
        evaluation_probabilities[model_name] = probability
        evaluation_predictions[model_name] = (probability >= 0.5).astype(int)

    fitted_components = []
    for component in selected_components:
        base = clone(best_estimators[component])
        if selected_calibration == "none":
            final_component = base.fit(X_development, y_development)
        else:
            calibration_cv = StratifiedKFold(
                n_splits=5, shuffle=True, random_state=RANDOM_SEED + 2
            )
            final_component = CalibratedClassifierCV(
                estimator=base,
                method=selected_calibration,
                cv=calibration_cv,
            ).fit(X_development, y_development)
        fitted_components.append(final_component)

    soft_name = f"soft_voting_{selected_calibration}"
    soft_ensemble = FittedProbabilityEnsemble(tuple(fitted_components))
    soft_probability = soft_ensemble.predict_proba(X_evaluation)[:, 1]
    evaluation_probabilities[soft_name] = soft_probability
    evaluation_predictions[soft_name] = (soft_probability >= 0.5).astype(int)

    component_predictions = np.column_stack(
        [evaluation_predictions[component] for component in selected_components]
    )
    hard_vote_fraction = component_predictions.mean(axis=1)
    hard_name = "hard_voting_majority"
    evaluation_probabilities[hard_name] = hard_vote_fraction
    evaluation_predictions[hard_name] = (hard_vote_fraction >= 0.5).astype(int)

    joblib.dump(soft_ensemble, model_root / f"{feature_set_name}__{soft_name}.joblib")
    (model_root / f"{feature_set_name}__ensemble_metadata.json").write_text(
        json.dumps(
            {
                "feature_set": feature_set_name,
                "features": features,
                "components": selected_components,
                "calibration_method": selected_calibration,
                "soft_voting_definition": "arithmetic mean of positive-class probabilities",
                "classification_threshold": 0.5,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    prediction_table = analytic.iloc[evaluation_index][[ROW_ID, SOURCE]].copy()
    prediction_table[TARGET] = y_evaluation.to_numpy()
    for model_name in evaluation_probabilities:
        prediction_table[f"{model_name}__probability"] = evaluation_probabilities[model_name]
        prediction_table[f"{model_name}__prediction"] = evaluation_predictions[model_name]
    prediction_table.to_csv(
        data_root / f"evaluation_predictions__{feature_set_name}.csv", index=False
    )

    holdout_rows = []
    for offset, (model_name, probability) in enumerate(evaluation_probabilities.items()):
        metrics = classification_metrics(y_evaluation.to_numpy(), probability)
        intervals = stratified_bootstrap_intervals(
            y_evaluation.to_numpy(),
            probability,
            n_bootstrap=N_BOOTSTRAP_CI,
            random_state=RANDOM_SEED + offset,
        )
        row = {
            "feature_set": feature_set_name,
            "model": model_name,
            "probability_source": (
                "mean calibrated predict_proba"
                if model_name == soft_name
                else "vote fraction"
                if model_name == hard_name
                else "model predict_proba"
            ),
            **add_confidence_intervals(metrics, intervals),
        }
        if model_name == hard_name:
            # Hard voting is a deterministic comparator; do not present its vote
            # fraction as a calibrated clinical probability.
            for metric in [
                "brier",
                "log_loss",
                "calibration_intercept",
                "calibration_slope",
            ]:
                row[metric] = np.nan
            for metric in ["brier", "log_loss"]:
                row[f"{metric}_ci_low"] = np.nan
                row[f"{metric}_ci_high"] = np.nan
        holdout_rows.append(row)
    holdout = pd.DataFrame(holdout_rows).sort_values("roc_auc", ascending=False)
    holdout.to_csv(table_root / f"evaluation_metrics__{feature_set_name}.csv", index=False)

    mcnemar = paired_mcnemar_table(y_evaluation.to_numpy(), evaluation_predictions)
    mcnemar.insert(0, "feature_set", feature_set_name)
    mcnemar.to_csv(table_root / f"mcnemar_pairwise__{feature_set_name}.csv", index=False)

    best_single = ranked.loc[ranked["model"].isin(ensemble_candidates), "model"].iloc[0]
    auc_comparison = paired_auc_bootstrap(
        y_evaluation.to_numpy(),
        evaluation_probabilities[soft_name],
        evaluation_probabilities[best_single],
        soft_name,
        best_single,
    )
    pd.DataFrame([{**{"feature_set": feature_set_name}, **auc_comparison}]).to_csv(
        table_root / f"paired_auc_bootstrap__{feature_set_name}.csv", index=False
    )

    # Random-forest OOB trajectory, using preprocessing learned only
    # from the development partition.
    rf_model = fitted_models["random_forest"]
    transformed = rf_model.named_steps["preprocess"].transform(X_development)
    best_rf = rf_model.named_steps["model"]
    oob_rows = []
    for trees in [10, 25, 50, 100, 200, 300, 500, 700, 1000]:
        forest = RandomForestClassifier(
            n_estimators=trees,
            criterion=best_rf.criterion,
            max_depth=best_rf.max_depth,
            max_features=best_rf.max_features,
            bootstrap=True,
            oob_score=True,
            n_jobs=1,
            random_state=RANDOM_SEED,
        ).fit(transformed, y_development)
        oob_rows.append(
            {
                "feature_set": feature_set_name,
                "n_estimators": trees,
                "oob_accuracy": forest.oob_score_,
                "oob_error": 1.0 - forest.oob_score_,
            }
        )
    pd.DataFrame(oob_rows).to_csv(
        table_root / f"random_forest_oob__{feature_set_name}.csv", index=False
    )

    ann_ablation = ranked.loc[
        ranked["model"].isin(["ann_shallow", "ann_original_12_6_3", "ann_deep"]),
        [
            "feature_set",
            "model",
            "best_parameters",
            "roc_auc",
            "average_precision",
            "balanced_accuracy",
            "f1",
            "brier",
            "log_loss",
        ],
    ]
    ann_ablation.to_csv(table_root / f"ann_ablation__{feature_set_name}.csv", index=False)

    return {
        "feature_set": feature_set_name,
        "n_features": len(features),
        "selected_components": selected_components,
        "selected_calibration": selected_calibration,
        "soft_ensemble_name": soft_name,
        "best_single_by_development_rank": best_single,
    }


def main() -> None:
    ensure_output_directories()
    analytic_path = OUTPUT_ROOT / "data" / "analytic_920.csv"
    if not analytic_path.exists():
        raise FileNotFoundError("Run 01_prepare_data.py before model analysis")
    analytic = pd.read_csv(analytic_path)
    indices = np.arange(len(analytic))
    development_index, evaluation_index = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=analytic[TARGET],
    )
    pd.DataFrame(
        {
            "row_id": analytic[ROW_ID],
            "source_cohort": analytic[SOURCE],
            "target": analytic[TARGET],
            "partition": np.where(
                np.isin(indices, development_index), "development", "evaluation"
            ),
        }
    ).to_csv(OUTPUT_ROOT / "data" / "split_assignment.csv", index=False)

    summaries = []
    for feature_set_name, features in FEATURE_SETS.items():
        summaries.append(
            analyze_feature_set(
                feature_set_name,
                features,
                analytic,
                development_index,
                evaluation_index,
            )
        )

    split_table = pd.DataFrame(
        {
            "partition": ["development", "evaluation"],
            "n": [len(development_index), len(evaluation_index)],
            "target_0": [
                int((analytic.iloc[development_index][TARGET] == 0).sum()),
                int((analytic.iloc[evaluation_index][TARGET] == 0).sum()),
            ],
            "target_1": [
                int((analytic.iloc[development_index][TARGET] == 1).sum()),
                int((analytic.iloc[evaluation_index][TARGET] == 1).sum()),
            ],
        }
    )
    split_table.to_csv(OUTPUT_ROOT / "tables" / "split_summary.csv", index=False)

    metadata = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
        "random_seed": RANDOM_SEED,
        "split": {
            "method": "stratified 70/30 hold-out",
            "development_n": int(len(development_index)),
            "evaluation_n": int(len(evaluation_index)),
        },
        "bootstrap_ci_replicates": N_BOOTSTRAP_CI,
        "bootstrap_auc_difference_replicates": N_BOOTSTRAP_AUC_DIFFERENCE,
        "feature_set_results": summaries,
    }
    (OUTPUT_ROOT / "logs" / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
