"""Focused extension of the model and ensemble analysis.

This stage adds three established tree-ensemble families (Extra Trees,
gradient boosting, and histogram gradient boosting) and compares three ensemble
strategies: equal-probability averaging, a non-negative log-loss-optimized
blend, and logistic probability stacking. Hyperparameters, ensemble membership,
weights, meta-model behavior, and an optional operating threshold are selected
using the 644-record development partition only. The existing 276-record
evaluation partition is used once after the protocol is locked.

The comparison is deliberately small, uses only scikit-learn models, preserves
the paper's non-invasive feature set, and exports every tried grid combination
and prediction.
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import binomtest
from sklearn.base import BaseEstimator, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from pipeline_utils import (
    FEATURE_SETS,
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


FEATURE_SET_NAME = "noninvasive12"
FEATURES = FEATURE_SETS[FEATURE_SET_NAME]
N_BOOTSTRAP_CI = 2000
N_BOOTSTRAP_AUC_DIFFERENCE = 5000

# IterativeImputer/BayesianRidge can emit harmless BLAS RuntimeWarnings while
# converging on small fold-specific matrices. Failed fits still raise because
# GridSearchCV uses error_score="raise".
warnings.filterwarnings("ignore", category=RuntimeWarning)


def fixed_existing_models() -> dict[str, BaseEstimator]:
    """Return the four strongest established models with locked parameters."""

    return {
        "logistic_regression": make_model_pipeline(
            FEATURES,
            LogisticRegression(
                C=0.1,
                penalty="l2",
                solver="liblinear",
                max_iter=5000,
                random_state=RANDOM_SEED,
            ),
        ),
        "knn": make_model_pipeline(
            FEATURES,
            KNeighborsClassifier(n_neighbors=50, p=1, weights="distance"),
        ),
        "svc": make_model_pipeline(
            FEATURES,
            SVC(
                C=1.0,
                gamma="auto",
                kernel="rbf",
                probability=True,
                random_state=RANDOM_SEED,
            ),
        ),
        "random_forest": make_model_pipeline(
            FEATURES,
            RandomForestClassifier(
                n_estimators=300,
                criterion="gini",
                max_depth=4,
                max_features="sqrt",
                n_jobs=-1,
                random_state=RANDOM_SEED,
            ),
        ),
    }


def extension_specs() -> dict[str, tuple[BaseEstimator, dict[str, list[object]]]]:
    """Return a deliberately limited set of additional tree ensembles."""

    return {
        "extra_trees": (
            ExtraTreesClassifier(n_jobs=-1, random_state=RANDOM_SEED),
            {
                "model__n_estimators": [300, 700],
                "model__max_depth": [None, 8, 12],
                "model__min_samples_leaf": [1, 2],
                "model__max_features": ["sqrt", 0.7],
            },
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=RANDOM_SEED),
            {
                "model__n_estimators": [100, 200, 400],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__max_depth": [1, 2],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(
                max_iter=300,
                min_samples_leaf=20,
                random_state=RANDOM_SEED,
            ),
            {
                "model__learning_rate": [0.05, 0.1],
                "model__max_leaf_nodes": [7, 15],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
    }


def calibrated_oof_probability(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    outer_cv: StratifiedKFold,
) -> np.ndarray:
    """Generate outer-fold predictions with inner-fold sigmoid calibration."""

    inner_cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_SEED + 101,
    )
    calibrated = CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=inner_cv,
    )
    return cross_val_predict(
        calibrated,
        X,
        y,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]


def fit_final_calibrated(
    estimator: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
) -> BaseEstimator:
    """Fit one sigmoid-calibrated base model on the complete development set."""

    calibration_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED + 102,
    )
    return CalibratedClassifierCV(
        estimator=clone(estimator),
        method="sigmoid",
        cv=calibration_cv,
    ).fit(X, y)


def compact_metrics(
    name: str,
    kind: str,
    y: np.ndarray,
    probability: np.ndarray,
) -> dict[str, object]:
    """Return metrics used to compare development candidates."""

    values = classification_metrics(y, probability)
    return {
        "candidate": name,
        "candidate_type": kind,
        **values,
    }


def add_composite_rank(table: pd.DataFrame) -> pd.DataFrame:
    """Rank deterministic and probabilistic performance with equal influence."""

    ranked = table.copy()
    columns = []
    for metric in [
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "average_precision",
    ]:
        name = f"rank_{metric}"
        ranked[name] = ranked[metric].rank(ascending=False, method="average")
        columns.append(name)
    for metric in ["brier", "log_loss"]:
        name = f"rank_{metric}"
        ranked[name] = ranked[metric].rank(ascending=True, method="average")
        columns.append(name)
    ranked["composite_rank"] = ranked[columns].mean(axis=1)
    return ranked.sort_values(["composite_rank", "roc_auc"], ascending=[True, False])


def fit_simplex_weights(probabilities: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit non-negative weights summing to one by regularized log loss."""

    n_models = probabilities.shape[1]
    initial = np.full(n_models, 1.0 / n_models)

    def objective(weights: np.ndarray) -> float:
        blended = np.clip(probabilities @ weights, 1e-6, 1.0 - 1e-6)
        return float(log_loss(y, blended, labels=[0, 1]) + 1e-4 * np.sum(weights**2))

    fitted = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * n_models,
        constraints={"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12},
    )
    if not fitted.success:
        raise RuntimeError(f"Weight optimization failed: {fitted.message}")
    weights = np.clip(fitted.x, 0.0, 1.0)
    return weights / weights.sum()


def cross_fitted_weighted_blend(
    probabilities: np.ndarray,
    y: np.ndarray,
    cv: StratifiedKFold,
) -> np.ndarray:
    """Evaluate learned blend weights without scoring their training rows."""

    result = np.empty(len(y), dtype=float)
    for training, validation in cv.split(probabilities, y):
        weights = fit_simplex_weights(probabilities[training], y[training])
        result[validation] = probabilities[validation] @ weights
    return result


def select_threshold(y: np.ndarray, probability: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose a balanced operating threshold using development data only."""

    rows = []
    for threshold in np.round(np.arange(0.30, 0.701, 0.01), 2):
        prediction = (probability >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "balanced_accuracy": balanced_accuracy_score(y, prediction),
                "accuracy": accuracy_score(y, prediction),
                "f1": f1_score(y, prediction),
                "distance_from_0_5": abs(threshold - 0.5),
            }
        )
    table = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "accuracy", "f1", "distance_from_0_5"],
        ascending=[False, False, False, True],
    )
    return float(table.iloc[0]["threshold"]), table


def threshold_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    """Calculate deterministic metrics at a locked non-default threshold."""

    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y, prediction),
        "balanced_accuracy": balanced_accuracy_score(y, prediction),
        "sensitivity": recall_score(y, prediction),
        "specificity": tn / (tn + fp),
        "precision": precision_score(y, prediction),
        "f1": f1_score(y, prediction),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def paired_mcnemar(
    y: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Run exact paired McNemar tests and Holm-adjust all pairs."""

    rows = []
    for first, second in itertools.combinations(sorted(predictions), 2):
        first_correct = predictions[first] == y
        second_correct = predictions[second] == y
        first_only = int(np.sum(first_correct & ~second_correct))
        second_only = int(np.sum(~first_correct & second_correct))
        discordant = first_only + second_only
        p_value = (
            1.0
            if discordant == 0
            else float(binomtest(min(first_only, second_only), discordant, 0.5).pvalue)
        )
        rows.append(
            {
                "model_a": first,
                "model_b": second,
                "a_correct_b_wrong": first_only,
                "a_wrong_b_correct": second_only,
                "discordant_pairs": discordant,
                "exact_mcnemar_p": p_value,
            }
        )
    table = pd.DataFrame(rows)
    table["holm_adjusted_p"] = holm_adjust(table["exact_mcnemar_p"])
    return table


def paired_auc_bootstrap(
    y: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
) -> dict[str, float | int | str]:
    """Paired stratified bootstrap for the difference in AUROC."""

    zero = np.flatnonzero(y == 0)
    one = np.flatnonzero(y == 1)
    rng = np.random.default_rng(RANDOM_SEED + 500)
    differences = []
    for _ in range(N_BOOTSTRAP_AUC_DIFFERENCE):
        index = np.concatenate(
            [
                rng.choice(zero, len(zero), replace=True),
                rng.choice(one, len(one), replace=True),
            ]
        )
        differences.append(
            roc_auc_score(y[index], first[index])
            - roc_auc_score(y[index], second[index])
        )
    differences = np.asarray(differences)
    observed = roc_auc_score(y, first) - roc_auc_score(y, second)
    p_value = 2.0 * min(np.mean(differences <= 0), np.mean(differences >= 0))
    return {
        "model_a": first_name,
        "model_b": second_name,
        "auc_difference_a_minus_b": observed,
        "ci_low": float(np.percentile(differences, 2.5)),
        "ci_high": float(np.percentile(differences, 97.5)),
        "two_sided_bootstrap_p": min(1.0, float(p_value)),
        "bootstrap_replicates": N_BOOTSTRAP_AUC_DIFFERENCE,
    }


def main() -> None:
    ensure_output_directories()
    extension_root = OUTPUT_ROOT / "extended"
    grid_root = extension_root / "grid_search"
    model_root = extension_root / "models"
    for path in [extension_root, grid_root, model_root]:
        path.mkdir(parents=True, exist_ok=True)

    analytic = pd.read_csv(OUTPUT_ROOT / "data" / "analytic_920.csv")
    indices = np.arange(len(analytic))
    development_index, evaluation_index = train_test_split(
        indices,
        test_size=0.30,
        random_state=RANDOM_SEED,
        stratify=analytic[TARGET],
    )
    X = analytic[FEATURES]
    y = analytic[TARGET].astype(int)
    X_development = X.iloc[development_index].copy()
    y_development = y.iloc[development_index].copy()
    X_evaluation = X.iloc[evaluation_index].copy()
    y_evaluation = y.iloc[evaluation_index].copy()

    selection_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED,
    )
    scoring = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
        "accuracy": "accuracy",
        "f1": "f1",
        "neg_brier": "neg_brier_score",
        "neg_log_loss": "neg_log_loss",
    }

    base_estimators = fixed_existing_models()
    extension_selection = []
    for name, (estimator, grid) in extension_specs().items():
        print(f"Tuning {name}...", flush=True)
        search = GridSearchCV(
            make_model_pipeline(FEATURES, estimator),
            param_grid=grid,
            scoring=scoring,
            refit="roc_auc",
            cv=selection_cv,
            n_jobs=1,
            return_train_score=True,
            error_score="raise",
        )
        search.fit(X_development, y_development)
        grid_table = pd.DataFrame(search.cv_results_)
        grid_table.insert(0, "model", name)
        grid_table.to_csv(grid_root / f"{name}.csv", index=False)
        base_estimators[name] = clone(search.best_estimator_)
        extension_selection.append(
            {
                "model": name,
                "best_parameters": json.dumps(search.best_params_, sort_keys=True),
                "best_mean_cv_roc_auc": float(search.best_score_),
            }
        )
    pd.DataFrame(extension_selection).to_csv(
        extension_root / "new_model_selection.csv",
        index=False,
    )

    outer_cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_SEED + 100,
    )
    base_names = list(base_estimators)
    base_oof: dict[str, np.ndarray] = {}
    development_rows = []
    for name in base_names:
        print(f"Generating calibrated OOF predictions for {name}...", flush=True)
        probability = calibrated_oof_probability(
            base_estimators[name],
            X_development,
            y_development,
            outer_cv,
        )
        base_oof[name] = probability
        development_rows.append(
            compact_metrics(
                name,
                "single_model_sigmoid_calibrated",
                y_development.to_numpy(),
                probability,
            )
        )

    development_base = add_composite_rank(pd.DataFrame(development_rows))
    development_base.to_csv(extension_root / "development_extended_models.csv", index=False)

    probability_matrix = np.column_stack([base_oof[name] for name in base_names])
    ensemble_rows = []
    ensemble_development_probabilities: dict[str, np.ndarray] = {}

    # Search equal averages over small subsets using development predictions only.
    equal_subset_rows = []
    for size in range(2, min(6, len(base_names)) + 1):
        for members in itertools.combinations(base_names, size):
            columns = [base_names.index(member) for member in members]
            probability = probability_matrix[:, columns].mean(axis=1)
            row = compact_metrics(
                "+".join(members),
                "equal_probability_average",
                y_development.to_numpy(),
                probability,
            )
            row["members"] = ";".join(members)
            equal_subset_rows.append(row)
    equal_subsets = add_composite_rank(pd.DataFrame(equal_subset_rows))
    equal_subsets.to_csv(extension_root / "development_equal_subset_search.csv", index=False)
    best_equal_members = str(equal_subsets.iloc[0]["members"]).split(";")
    best_equal_columns = [base_names.index(member) for member in best_equal_members]
    best_equal_probability = probability_matrix[:, best_equal_columns].mean(axis=1)
    ensemble_development_probabilities["equal_average_selected"] = best_equal_probability
    ensemble_rows.append(
        compact_metrics(
            "equal_average_selected",
            "ensemble",
            y_development.to_numpy(),
            best_equal_probability,
        )
        | {"members": ";".join(best_equal_members)}
    )

    weighted_probability = cross_fitted_weighted_blend(
        probability_matrix,
        y_development.to_numpy(),
        outer_cv,
    )
    final_weights = fit_simplex_weights(probability_matrix, y_development.to_numpy())
    ensemble_development_probabilities["constrained_weighted_blend"] = weighted_probability
    ensemble_rows.append(
        compact_metrics(
            "constrained_weighted_blend",
            "ensemble",
            y_development.to_numpy(),
            weighted_probability,
        )
        | {"members": ";".join(base_names)}
    )

    stacker = LogisticRegression(C=1.0, solver="liblinear", random_state=RANDOM_SEED)
    stacked_probability = cross_val_predict(
        clone(stacker),
        probability_matrix,
        y_development,
        cv=outer_cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    ensemble_development_probabilities["logistic_probability_stack"] = stacked_probability
    ensemble_rows.append(
        compact_metrics(
            "logistic_probability_stack",
            "ensemble",
            y_development.to_numpy(),
            stacked_probability,
        )
        | {"members": ";".join(base_names)}
    )

    existing_probability = np.column_stack(
        [base_oof[name] for name in ["knn", "logistic_regression", "svc"]]
    ).mean(axis=1)
    ensemble_development_probabilities["original_equal_soft_vote"] = existing_probability
    ensemble_rows.append(
        compact_metrics(
            "original_equal_soft_vote",
            "ensemble",
            y_development.to_numpy(),
            existing_probability,
        )
        | {"members": "knn;logistic_regression;svc"}
    )

    ensemble_development = add_composite_rank(pd.DataFrame(ensemble_rows))
    ensemble_development.to_csv(
        extension_root / "development_ensemble_strategies.csv",
        index=False,
    )
    selected_ensemble = str(ensemble_development.iloc[0]["candidate"])
    selected_development_probability = ensemble_development_probabilities[selected_ensemble]
    selected_threshold, threshold_table = select_threshold(
        y_development.to_numpy(),
        selected_development_probability,
    )
    threshold_table.to_csv(extension_root / "development_threshold_selection.csv", index=False)

    # Evaluation is used only after all development selections above are locked.
    fitted_base: dict[str, BaseEstimator] = {}
    base_evaluation: dict[str, np.ndarray] = {}
    for name in base_names:
        print(f"Fitting final calibrated {name}...", flush=True)
        fitted = fit_final_calibrated(
            base_estimators[name],
            X_development,
            y_development,
        )
        fitted_base[name] = fitted
        base_evaluation[name] = positive_probability(fitted, X_evaluation)
        joblib.dump(fitted, model_root / f"{name}_sigmoid_calibrated.joblib")

    evaluation_matrix = np.column_stack([base_evaluation[name] for name in base_names])
    ensemble_evaluation: dict[str, np.ndarray] = {
        "equal_average_selected": evaluation_matrix[:, best_equal_columns].mean(axis=1),
        "constrained_weighted_blend": evaluation_matrix @ final_weights,
        "original_equal_soft_vote": np.column_stack(
            [base_evaluation[name] for name in ["knn", "logistic_regression", "svc"]]
        ).mean(axis=1),
    }
    final_stacker = clone(stacker).fit(probability_matrix, y_development)
    ensemble_evaluation["logistic_probability_stack"] = final_stacker.predict_proba(
        evaluation_matrix
    )[:, 1]
    joblib.dump(final_stacker, model_root / "logistic_probability_stacker.joblib")

    all_evaluation = {**base_evaluation, **ensemble_evaluation}
    evaluation_rows = []
    for offset, (name, probability) in enumerate(all_evaluation.items()):
        metrics = classification_metrics(y_evaluation.to_numpy(), probability)
        intervals = stratified_bootstrap_intervals(
            y_evaluation.to_numpy(),
            probability,
            n_bootstrap=N_BOOTSTRAP_CI,
            random_state=RANDOM_SEED + 700 + offset,
        )
        evaluation_rows.append(
            {
                "candidate": name,
                "candidate_type": "ensemble" if name in ensemble_evaluation else "single_model",
                **add_confidence_intervals(metrics, intervals),
            }
        )
    evaluation_table = pd.DataFrame(evaluation_rows).sort_values(
        ["accuracy", "roc_auc"],
        ascending=[False, False],
    )
    evaluation_table.to_csv(
        extension_root / "evaluation_extended_models_and_ensembles.csv",
        index=False,
    )

    prediction_table = analytic.iloc[evaluation_index][[ROW_ID, SOURCE]].copy()
    prediction_table[TARGET] = y_evaluation.to_numpy()
    for name, probability in all_evaluation.items():
        prediction_table[f"{name}__probability"] = probability
        prediction_table[f"{name}__prediction_0_5"] = (probability >= 0.5).astype(int)
    prediction_table[f"{selected_ensemble}__prediction_selected_threshold"] = (
        ensemble_evaluation[selected_ensemble] >= selected_threshold
    ).astype(int)
    prediction_table.to_csv(extension_root / "evaluation_predictions.csv", index=False)

    prediction_dict = {
        name: (probability >= 0.5).astype(int)
        for name, probability in all_evaluation.items()
    }
    paired_mcnemar(y_evaluation.to_numpy(), prediction_dict).to_csv(
        extension_root / "mcnemar_all_extended_candidates.csv",
        index=False,
    )

    best_single = str(development_base.iloc[0]["candidate"])
    auc_comparison = paired_auc_bootstrap(
        y_evaluation.to_numpy(),
        ensemble_evaluation[selected_ensemble],
        base_evaluation[best_single],
        selected_ensemble,
        best_single,
    )
    pd.DataFrame([auc_comparison]).to_csv(
        extension_root / "paired_auc_selected_ensemble_vs_best_single.csv",
        index=False,
    )

    threshold_result = threshold_metrics(
        y_evaluation.to_numpy(),
        ensemble_evaluation[selected_ensemble],
        selected_threshold,
    )
    pd.DataFrame([threshold_result]).to_csv(
        extension_root / "evaluation_selected_threshold_metrics.csv",
        index=False,
    )

    metadata = {
        "feature_set": FEATURE_SET_NAME,
        "development_n": int(len(development_index)),
        "evaluation_n": int(len(evaluation_index)),
        "additional_model_families": list(extension_specs()),
        "base_models": base_names,
        "selected_equal_average_members": best_equal_members,
        "constrained_blend_weights": {
            name: float(weight) for name, weight in zip(base_names, final_weights)
        },
        "selected_ensemble_by_development_composite_rank": selected_ensemble,
        "best_single_by_development_composite_rank": best_single,
        "development_selected_threshold": selected_threshold,
        "threshold_evaluation_metrics": threshold_result,
        "auc_comparison": auc_comparison,
        "evaluation_not_used_for_selection": True,
    }
    (extension_root / "extended_analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
