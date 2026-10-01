"""Shared utilities for the reproducible CAD analysis.

The key design constraint is that every learned preprocessing operation lives
inside a scikit-learn pipeline. This prevents imputation, encoding, or scaling
from learning from evaluation observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


ANALYSIS_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ANALYSIS_ROOT.parent
# The Overleaf/repository package carries only the four UCI files used by this
# analysis (plus the UCI documentation) so the workflow is self-contained.
RAW_DATA_ROOT = ANALYSIS_ROOT / "raw_data"
OUTPUT_ROOT = ANALYSIS_ROOT / "outputs"

RANDOM_SEED = 10
TARGET = "target"
SOURCE = "source_cohort"
ROW_ID = "row_id"

UCI_COLUMNS = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
    "num",
]

CONTINUOUS_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL_LEVELS = {
    "sex": [0.0, 1.0],
    "cp": [1.0, 2.0, 3.0, 4.0],
    "fbs": [0.0, 1.0],
    "restecg": [0.0, 1.0, 2.0],
    "exang": [0.0, 1.0],
    "slope": [1.0, 2.0, 3.0],
    "ca": [0.0, 1.0, 2.0, 3.0],
    "thal": [3.0, 6.0, 7.0],
}

FEATURE_SETS = {
    "common10": [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
    ],
    "noninvasive12": [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "thal",
    ],
    "full13": [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
    ],
}


class CategorySnapper(TransformerMixin, BaseEstimator):
    """Map continuous KNN-imputed category codes to valid UCI levels.

    KNN imputation can return a weighted mean such as 2.4. This transformer maps
    each imputed value to the closest valid category before one-hot encoding.
    Observed valid values are unchanged.
    """

    def __init__(self, allowed_levels: tuple[tuple[float, ...], ...]):
        self.allowed_levels = allowed_levels

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "CategorySnapper":
        array = np.asarray(X)
        if array.ndim != 2 or array.shape[1] != len(self.allowed_levels):
            raise ValueError("CategorySnapper received an unexpected number of columns")
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        array = np.asarray(X, dtype=float).copy()
        for column_index, levels in enumerate(self.allowed_levels):
            valid = np.asarray(levels, dtype=float)
            distances = np.abs(array[:, column_index, None] - valid[None, :])
            array[:, column_index] = valid[np.argmin(distances, axis=1)]
        return array


def build_preprocessor(features: list[str], random_state: int = RANDOM_SEED) -> ColumnTransformer:
    """Build fold-fitted preprocessing for one feature set."""

    continuous = [feature for feature in features if feature in CONTINUOUS_FEATURES]
    categorical = [feature for feature in features if feature in CATEGORICAL_LEVELS]
    allowed = tuple(tuple(CATEGORICAL_LEVELS[feature]) for feature in categorical)

    continuous_pipe = Pipeline(
        steps=[
            # StandardScaler natively preserves NaNs. Scaling before Bayesian
            # chained equations stabilizes its matrix operations; because this
            # pipeline is fitted within each training fold, it cannot leak
            # evaluation-set means or variances.
            ("scale", StandardScaler()),
            (
                "mice",
                IterativeImputer(
                    estimator=BayesianRidge(),
                    initial_strategy="median",
                    max_iter=10,
                    random_state=random_state,
                    skip_complete=True,
                    tol=1e-3,
                ),
            ),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("knn_impute", KNNImputer(n_neighbors=5, weights="distance")),
            ("snap", CategorySnapper(allowed)),
            (
                "one_hot",
                OneHotEncoder(
                    categories=[list(levels) for levels in allowed],
                    drop="if_binary",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("continuous", continuous_pipe, continuous),
            ("categorical", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_model_pipeline(features: list[str], estimator: BaseEstimator) -> Pipeline:
    """Combine leakage-free preprocessing and a classifier."""

    return Pipeline(
        steps=[
            ("preprocess", build_preprocessor(features)),
            ("model", estimator),
        ]
    )


def positive_probability(estimator: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """Return the probability assigned to class 1."""

    probabilities = estimator.predict_proba(X)
    classes = np.asarray(estimator.classes_)
    positive_index = int(np.flatnonzero(classes == 1)[0])
    return np.asarray(probabilities[:, positive_index], dtype=float)


class FittedProbabilityEnsemble(ClassifierMixin, BaseEstimator):
    """Read-only mean-probability ensemble over already fitted estimators."""

    def __init__(self, estimators: tuple[BaseEstimator, ...], threshold: float = 0.5):
        self.estimators = estimators
        self.threshold = threshold
        self.estimators_ = estimators
        self.classes_ = np.asarray([0, 1])
        self.n_features_in_ = getattr(estimators[0], "n_features_in_", None)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "FittedProbabilityEnsemble":
        raise RuntimeError("FittedProbabilityEnsemble is constructed from fitted estimators")

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.column_stack(
            [positive_probability(estimator, X) for estimator in self.estimators_]
        ).mean(axis=1)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


def calibration_intercept_slope(y_true: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    """Estimate calibration intercept and slope from log-odds predictions."""

    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)
    predictor = logit(probability).reshape(-1, 1)
    calibration = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    calibration.fit(predictor, y_true)
    return float(calibration.intercept_[0]), float(calibration.coef_[0, 0])


def classification_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Calculate deterministic and probabilistic binary-classification metrics."""

    y_true = np.asarray(y_true, dtype=int)
    probability = np.clip(np.asarray(probability, dtype=float), 1e-12, 1.0 - 1e-12)
    prediction = (probability >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if tn + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    intercept, slope = calibration_intercept_slope(y_true, probability)
    return {
        "accuracy": accuracy_score(y_true, prediction),
        "balanced_accuracy": balanced_accuracy_score(y_true, prediction),
        "sensitivity": recall_score(y_true, prediction, zero_division=0),
        "specificity": specificity,
        "precision": precision_score(y_true, prediction, zero_division=0),
        "npv": npv,
        "f1": f1_score(y_true, prediction, zero_division=0),
        "mcc": matthews_corrcoef(y_true, prediction),
        "roc_auc": roc_auc_score(y_true, probability),
        "average_precision": average_precision_score(y_true, probability),
        "brier": brier_score_loss(y_true, probability),
        "log_loss": log_loss(y_true, np.column_stack([1 - probability, probability]), labels=[0, 1]),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
    }


BOOTSTRAP_METRICS = [
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


def stratified_bootstrap_intervals(
    y_true: np.ndarray,
    probability: np.ndarray,
    n_bootstrap: int = 2000,
    random_state: int = RANDOM_SEED,
) -> dict[str, tuple[float, float]]:
    """Return percentile 95% CIs from a class-stratified paired bootstrap."""

    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    class_zero = np.flatnonzero(y_true == 0)
    class_one = np.flatnonzero(y_true == 1)
    rng = np.random.default_rng(random_state)
    samples: dict[str, list[float]] = {metric: [] for metric in BOOTSTRAP_METRICS}
    for _ in range(n_bootstrap):
        indices = np.concatenate(
            [
                rng.choice(class_zero, size=len(class_zero), replace=True),
                rng.choice(class_one, size=len(class_one), replace=True),
            ]
        )
        bootstrap_y = y_true[indices]
        bootstrap_probability = np.clip(probability[indices], 1e-12, 1.0 - 1e-12)
        bootstrap_prediction = (bootstrap_probability >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            bootstrap_y, bootstrap_prediction, labels=[0, 1]
        ).ravel()
        metrics = {
            "accuracy": accuracy_score(bootstrap_y, bootstrap_prediction),
            "balanced_accuracy": balanced_accuracy_score(
                bootstrap_y, bootstrap_prediction
            ),
            "sensitivity": recall_score(
                bootstrap_y, bootstrap_prediction, zero_division=0
            ),
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
            "precision": precision_score(
                bootstrap_y, bootstrap_prediction, zero_division=0
            ),
            "npv": tn / (tn + fn) if tn + fn else np.nan,
            "f1": f1_score(bootstrap_y, bootstrap_prediction, zero_division=0),
            "mcc": matthews_corrcoef(bootstrap_y, bootstrap_prediction),
            "roc_auc": roc_auc_score(bootstrap_y, bootstrap_probability),
            "average_precision": average_precision_score(
                bootstrap_y, bootstrap_probability
            ),
            "brier": brier_score_loss(bootstrap_y, bootstrap_probability),
            "log_loss": log_loss(
                bootstrap_y,
                np.column_stack(
                    [1 - bootstrap_probability, bootstrap_probability]
                ),
                labels=[0, 1],
            ),
        }
        for metric in BOOTSTRAP_METRICS:
            samples[metric].append(metrics[metric])
    return {
        metric: (
            float(np.nanpercentile(values, 2.5)),
            float(np.nanpercentile(values, 97.5)),
        )
        for metric, values in samples.items()
    }


def add_confidence_intervals(
    metrics: dict[str, float], intervals: dict[str, tuple[float, float]]
) -> dict[str, float]:
    """Flatten estimates and confidence limits for CSV output."""

    row = dict(metrics)
    for metric, (lower, upper) in intervals.items():
        row[f"{metric}_ci_low"] = lower
        row[f"{metric}_ci_high"] = upper
    return row


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    """Holm family-wise-error adjustment without an external dependency."""

    p_values = np.asarray(list(p_values), dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running_max = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * p_values[index])
        running_max = max(running_max, candidate)
        adjusted[index] = running_max
    return adjusted


def ensure_output_directories() -> None:
    """Create the clean output structure used by all stages."""

    for relative in [
        "data",
        "tables",
        "tables/grid_search",
        "figures",
        "models",
        "logs",
    ]:
        (OUTPUT_ROOT / relative).mkdir(parents=True, exist_ok=True)
