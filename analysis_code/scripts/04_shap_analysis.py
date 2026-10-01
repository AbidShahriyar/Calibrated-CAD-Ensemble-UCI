"""Estimate and plot SHAP values for the final ensemble.

This implementation uses the permutation definition of interventional Shapley
values and therefore does not depend on the optional ``shap`` package. For each
explained evaluation record, a random development-set background record is
chosen for every feature permutation. Features are then replaced in permutation
order, and each change in the ensemble's positive-class probability is credited
to the feature just introduced.

The estimates explain the saved, sigmoid-calibrated soft-voting ensemble itself
rather than a surrogate classifier. They are in probability units, and the
random seed, background source, sample size, and additivity error are exported.
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

from pipeline_utils import FEATURE_SETS, OUTPUT_ROOT, RANDOM_SEED, TARGET


PRIMARY_FEATURE_SET = "noninvasive12"
N_SUMMARY_RECORDS = 60
N_PERMUTATIONS = 64


def save_figure(fig: plt.Figure, filename: str) -> None:
    """Save a publication-resolution PNG and close its Matplotlib figure."""

    destination = OUTPUT_ROOT / "figures" / filename
    fig.savefig(destination, dpi=300, bbox_inches="tight")
    plt.close(fig)


def explain_record(
    model: object,
    target_row: pd.Series,
    background: pd.DataFrame,
    features: list[str],
    rng: np.random.Generator,
    n_permutations: int,
) -> tuple[np.ndarray, float, float, float]:
    """Return permutation-SHAP values and an additivity diagnostic.

    Each Monte Carlo path begins at one random development record and ends at
    the evaluation record. Predictions for all path states are evaluated in a
    single batch to keep the model-agnostic calculation tractable.
    """

    n_features = len(features)
    background_positions = rng.integers(0, len(background), size=n_permutations)
    states: list[np.ndarray] = []
    permutations: list[np.ndarray] = []
    target_values = target_row[features].to_numpy(dtype=float)

    for background_position in background_positions:
        current = background.iloc[background_position][features].to_numpy(dtype=float).copy()
        permutation = rng.permutation(n_features)
        permutations.append(permutation)
        states.append(current.copy())
        for feature_position in permutation:
            current[feature_position] = target_values[feature_position]
            states.append(current.copy())

    state_frame = pd.DataFrame(states, columns=features)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        probabilities = model.predict_proba(state_frame)[:, 1]
    paths = probabilities.reshape(n_permutations, n_features + 1)

    contributions = np.zeros((n_permutations, n_features), dtype=float)
    for path_index, permutation in enumerate(permutations):
        marginal_changes = np.diff(paths[path_index])
        contributions[path_index, permutation] = marginal_changes

    shap_values = contributions.mean(axis=0)
    baseline_probability = float(paths[:, 0].mean())
    model_probability = float(paths[:, -1].mean())
    reconstructed_probability = float(baseline_probability + shap_values.sum())
    additivity_error = float(reconstructed_probability - model_probability)
    return shap_values, baseline_probability, model_probability, additivity_error


def summary_plot(values: pd.DataFrame, features: list[str]) -> None:
    """Create a SHAP-style beeswarm summary plot in probability units."""

    mean_absolute = values.groupby("feature")["shap_value"].apply(lambda x: x.abs().mean())
    feature_order = mean_absolute.sort_values(ascending=True).index.tolist()
    jitter_rng = np.random.default_rng(RANDOM_SEED)

    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    scatter = None
    for y_position, feature in enumerate(feature_order):
        subset = values.loc[values["feature"] == feature].copy()
        numeric = pd.to_numeric(subset["feature_value"], errors="coerce")
        valid = numeric.dropna()
        if valid.empty or float(valid.max()) == float(valid.min()):
            normalized = np.full(len(subset), 0.5)
        else:
            low, high = valid.quantile([0.05, 0.95])
            if high == low:
                low, high = valid.min(), valid.max()
            normalized = ((numeric - low) / (high - low)).clip(0, 1).fillna(0.5).to_numpy()
        jitter = jitter_rng.uniform(-0.22, 0.22, size=len(subset))
        scatter = ax.scatter(
            subset["shap_value"],
            np.full(len(subset), y_position) + jitter,
            c=normalized,
            cmap="coolwarm",
            vmin=0,
            vmax=1,
            s=25,
            alpha=0.82,
            edgecolors="none",
        )

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(len(feature_order)), feature_order)
    ax.set_xlabel("Monte Carlo SHAP contribution to CAD probability")
    ax.set_title("SHAP summary for the calibrated soft-voting ensemble")
    if scatter is not None:
        colorbar = fig.colorbar(scatter, ax=ax, pad=0.02)
        colorbar.set_label("Feature value (within-feature scale)")
        colorbar.set_ticks([0, 1], labels=["Low", "High"])
    fig.tight_layout()
    save_figure(fig, "figure_shap_summary_primary_ensemble.png")


def random_case_plot(case_values: pd.DataFrame) -> None:
    """Plot numerical SHAP contributions for the reproducibly sampled case."""

    plot = case_values.sort_values("shap_value")
    colors = np.where(plot["shap_value"] >= 0, "#B55D60", "#4C78A8")
    fig, ax = plt.subplots(figsize=(7.2, 5.3))
    ax.barh(plot["feature"], plot["shap_value"], color=colors, alpha=0.9)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Contribution to predicted CAD probability")
    row = case_values.iloc[0]
    ax.set_title(
        "Random evaluation case "
        f"{row['row_id']}: P(CAD)={row['model_probability']:.3f}, "
        f"baseline={row['baseline_probability']:.3f}"
    )
    fig.tight_layout()
    save_figure(fig, "figure_shap_random_case_primary_ensemble.png")


def main() -> None:
    features = FEATURE_SETS[PRIMARY_FEATURE_SET]
    analytic = pd.read_csv(OUTPUT_ROOT / "data" / "analytic_920.csv")
    assignments = pd.read_csv(OUTPUT_ROOT / "data" / "split_assignment.csv")
    metadata = json.loads((OUTPUT_ROOT / "logs" / "run_metadata.json").read_text())
    feature_metadata = next(
        item for item in metadata["feature_set_results"]
        if item["feature_set"] == PRIMARY_FEATURE_SET
    )
    ensemble_name = str(feature_metadata["soft_ensemble_name"])
    model = joblib.load(
        OUTPUT_ROOT / "models" / f"{PRIMARY_FEATURE_SET}__{ensemble_name}.joblib"
    )

    indexed = analytic.set_index("row_id", drop=False)
    development_ids = assignments.loc[
        assignments["partition"] == "development", "row_id"
    ]
    evaluation_ids = assignments.loc[
        assignments["partition"] == "evaluation", "row_id"
    ]
    background = indexed.loc[development_ids, features].reset_index(drop=True)
    evaluation = indexed.loc[evaluation_ids].reset_index(drop=True)

    selection_rng = np.random.default_rng(RANDOM_SEED)
    summary_positions = np.sort(
        selection_rng.choice(
            len(evaluation),
            size=min(N_SUMMARY_RECORDS, len(evaluation)),
            replace=False,
        )
    )
    random_case_position = int(selection_rng.integers(0, len(evaluation)))
    positions_to_explain = sorted(set(summary_positions.tolist() + [random_case_position]))

    explanation_rng = np.random.default_rng(RANDOM_SEED)
    records: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for position in positions_to_explain:
        row = evaluation.iloc[position]
        shap_values, baseline, probability, additivity_error = explain_record(
            model=model,
            target_row=row,
            background=background,
            features=features,
            rng=explanation_rng,
            n_permutations=N_PERMUTATIONS,
        )
        diagnostics.append(
            {
                "row_id": row["row_id"],
                "baseline_probability": baseline,
                "model_probability": probability,
                "reconstructed_probability": baseline + float(shap_values.sum()),
                "additivity_error": additivity_error,
            }
        )
        for feature, shap_value in zip(features, shap_values):
            records.append(
                {
                    "row_id": row["row_id"],
                    "source_cohort": row["source_cohort"],
                    "target": int(row[TARGET]),
                    "feature": feature,
                    "feature_value": row[feature],
                    "shap_value": shap_value,
                    "baseline_probability": baseline,
                    "model_probability": probability,
                    "n_permutations": N_PERMUTATIONS,
                }
            )

    values = pd.DataFrame(records)
    summary_ids = set(evaluation.iloc[summary_positions]["row_id"])
    summary_values = values.loc[values["row_id"].isin(summary_ids)].copy()
    random_case_id = evaluation.iloc[random_case_position]["row_id"]
    random_case_values = values.loc[values["row_id"] == random_case_id].copy()

    values.to_csv(OUTPUT_ROOT / "tables" / "shap_values_primary_ensemble.csv", index=False)
    random_case_values.to_csv(
        OUTPUT_ROOT / "tables" / "shap_random_case_values.csv", index=False
    )
    pd.DataFrame(diagnostics).to_csv(
        OUTPUT_ROOT / "tables" / "shap_additivity_diagnostics.csv", index=False
    )
    summary_plot(summary_values, features)
    random_case_plot(random_case_values)

    shap_metadata = {
        "method": "Monte Carlo interventional permutation SHAP",
        "model": f"{PRIMARY_FEATURE_SET}__{ensemble_name}",
        "output_scale": "positive-class probability",
        "background_partition": "development",
        "background_n": int(len(background)),
        "summary_evaluation_records": int(len(summary_ids)),
        "permutations_per_record": N_PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        "random_case_row_id": str(random_case_id),
        "maximum_absolute_additivity_error": float(
            pd.DataFrame(diagnostics)["additivity_error"].abs().max()
        ),
    }
    (OUTPUT_ROOT / "logs" / "shap_metadata.json").write_text(
        json.dumps(shap_metadata, indent=2) + "\n"
    )
    print(
        "SHAP outputs written; random evaluation case: "
        f"{random_case_id}, P(CAD)={random_case_values['model_probability'].iloc[0]:.3f}"
    )


if __name__ == "__main__":
    main()
