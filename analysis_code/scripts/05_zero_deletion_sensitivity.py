"""Quantify the modeling impact of zero-value row deletion.

This bounded sensitivity analysis compares model metrics before and after the
removal of zero-coded cholesterol records. It constructs the complete-case
population (n=740) and its zero-deleted population (n=661), then fits the same
fixed common-feature component models and their unweighted probability
ensemble to each population.

It is descriptive rather than causal: deleting rows changes cohort composition
and the evaluation population. The 920-record imputed analysis remains primary.
"""

from __future__ import annotations

import json
import os
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/tmp/cad-analysis-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from pipeline_utils import (
    FEATURE_SETS,
    OUTPUT_ROOT,
    RANDOM_SEED,
    ROW_ID,
    TARGET,
    add_confidence_intervals,
    classification_metrics,
    make_model_pipeline,
    positive_probability,
    stratified_bootstrap_intervals,
)


N_BOOTSTRAP = 2_000
MODEL_ORDER = ["logistic_regression", "knn", "svc"]
MODEL_LABELS = {
    "logistic_regression": "Logistic regression",
    "knn": "KNN",
    "svc": "SVC",
    "soft_voting_fixed": "Soft voting",
}


def base_estimators() -> dict[str, object]:
    """Return the three common10 components selected on development data."""

    return {
        "logistic_regression": LogisticRegression(
            max_iter=5000, random_state=RANDOM_SEED
        ),
        "knn": KNeighborsClassifier(),
        "svc": SVC(probability=True, random_state=RANDOM_SEED),
    }


def main() -> None:
    analytic = pd.read_csv(OUTPUT_ROOT / "data" / "analytic_920.csv")
    membership = pd.read_csv(OUTPUT_ROOT / "data" / "record_exclusion_membership.csv")
    analytic = analytic.merge(membership[[ROW_ID, "in_complete_case_740", "in_after_zero_deletion_661"]], on=ROW_ID)
    ranking = pd.read_csv(OUTPUT_ROOT / "tables" / "development_model_ranking__common10.csv")
    parameter_lookup = {
        row["model"]: json.loads(row["best_parameters"])
        for _, row in ranking.loc[ranking["model"].isin(MODEL_ORDER)].iterrows()
    }
    features = FEATURE_SETS["common10"]
    stages = {
        "before_zero_row_deletion_n740": "in_complete_case_740",
        "after_zero_row_deletion_n661": "in_after_zero_deletion_661",
    }

    rows: list[dict[str, object]] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        for stage, membership_column in stages.items():
            data = analytic.loc[analytic[membership_column]].copy()
            train, evaluation = train_test_split(
                data,
                test_size=0.30,
                random_state=RANDOM_SEED,
                stratify=data[TARGET],
            )
            fitted = {}
            component_probabilities = []
            for model_name, estimator in base_estimators().items():
                pipeline = make_model_pipeline(features, estimator)
                pipeline.set_params(**parameter_lookup[model_name])
                pipeline.fit(train[features], train[TARGET])
                fitted[model_name] = pipeline
                component_probabilities.append(
                    positive_probability(pipeline, evaluation[features])
                )

            probability_by_model = {
                model_name: component_probabilities[position]
                for position, model_name in enumerate(MODEL_ORDER)
            }
            probability_by_model["soft_voting_fixed"] = np.column_stack(
                component_probabilities
            ).mean(axis=1)

            for model_offset, (model_name, probability) in enumerate(
                probability_by_model.items()
            ):
                estimates = classification_metrics(evaluation[TARGET], probability)
                intervals = stratified_bootstrap_intervals(
                    evaluation[TARGET].to_numpy(),
                    probability,
                    n_bootstrap=N_BOOTSTRAP,
                    random_state=RANDOM_SEED + model_offset,
                )
                row = add_confidence_intervals(estimates, intervals)
                row.update(
                    {
                        "stage": stage,
                        "model": model_name,
                        "population_n": len(data),
                        "development_n": len(train),
                        "evaluation_n": len(evaluation),
                        "development_target_0": int((train[TARGET] == 0).sum()),
                        "development_target_1": int((train[TARGET] == 1).sum()),
                        "evaluation_target_0": int((evaluation[TARGET] == 0).sum()),
                        "evaluation_target_1": int((evaluation[TARGET] == 1).sum()),
                    }
                )
                rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(
        OUTPUT_ROOT / "tables" / "zero_deletion_metric_sensitivity.csv", index=False
    )

    plot = results.loc[results["model"] == "soft_voting_fixed"].copy()
    plot["display_stage"] = plot["stage"].map(
        {
            "before_zero_row_deletion_n740": "Before deletion\n(n=740)",
            "after_zero_row_deletion_n661": "After deletion\n(n=661)",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.2))
    for ax, metric, label in [
        (axes[0], "accuracy", "Accuracy"),
        (axes[1], "roc_auc", "AUROC"),
    ]:
        x = np.arange(len(plot))
        estimate = plot[metric].to_numpy()
        error = np.vstack(
            [
                estimate - plot[f"{metric}_ci_low"].to_numpy(),
                plot[f"{metric}_ci_high"].to_numpy() - estimate,
            ]
        )
        ax.errorbar(x, estimate, yerr=error, fmt="o", capsize=4, color="#4C78A8")
        ax.set_xticks(x, plot["display_stage"])
        ax.set_ylabel(f"{label} (95% bootstrap CI)")
        ax.set_ylim(0.65, 1.0)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Sensitivity of the fixed soft-voting model to zero-row deletion")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_ROOT / "figures" / "figure_zero_deletion_metric_sensitivity.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(fig)
    print(results.loc[results["model"] == "soft_voting_fixed", [
        "stage", "evaluation_n", "accuracy", "roc_auc", "brier"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
