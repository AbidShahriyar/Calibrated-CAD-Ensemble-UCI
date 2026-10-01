"""Generate analysis tables and publication-ready figures."""

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
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
from sklearn.metrics import ConfusionMatrixDisplay, auc, confusion_matrix, roc_curve

from pipeline_utils import FEATURE_SETS, OUTPUT_ROOT, RANDOM_SEED, SOURCE, TARGET


PRIMARY_FEATURE_SET = "noninvasive12"
DISPLAY_NAMES = {
    "logistic_regression": "Logistic regression",
    "knn": "KNN",
    "gaussian_nb": "Gaussian NB",
    "svc": "SVC",
    "random_forest": "Random forest",
    "decision_tree": "Decision tree",
    "ann_shallow": "ANN shallow",
    "ann_original_12_6_3": "ANN 12-6-3",
    "ann_deep": "ANN deep",
    "hard_voting_majority": "Hard voting",
    "soft_voting_none": "Soft voting",
    "soft_voting_sigmoid": "Soft voting (sigmoid-calibrated)",
    "soft_voting_isotonic": "Soft voting (isotonic-calibrated)",
}


def save_figure(fig: plt.Figure, filename: str) -> None:
    path = OUTPUT_ROOT / "figures" / filename
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_columns(predictions: pd.DataFrame) -> list[str]:
    return [
        column.removesuffix("__probability")
        for column in predictions.columns
        if column.endswith("__probability")
    ]


def target_figures(analytic: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="paper")

    counts = analytic[TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    bars = ax.bar(["No CAD (0)", "CAD present (1)"], counts.values, color=["#4C78A8", "#E17C47"])
    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count + 8,
            f"{count}\n({100 * count / len(analytic):.1f}%)",
            ha="center",
            va="bottom",
        )
    ax.set_ylabel("Patient count")
    ax.set_title("Binary angiographic CAD outcome (N=920)")
    ax.set_ylim(0, max(counts) * 1.18)
    save_figure(fig, "figure_target_binary_counts.png")

    multiclass = analytic["num"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    bars = ax.bar(multiclass.index.astype(int).astype(str), multiclass.values, color="#5975A4")
    for bar, count in zip(bars, multiclass.values):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 5, str(count), ha="center")
    ax.set_xlabel("Original UCI outcome category (num)")
    ax.set_ylabel("Patient count")
    ax.set_title("Original multiclass outcome distribution")
    ax.set_ylim(0, max(multiclass) * 1.14)
    save_figure(fig, "figure_target_original_multiclass.png")


def invalid_cholesterol_figure() -> None:
    audit = pd.read_csv(OUTPUT_ROOT / "data" / "invalid_value_by_cohort.csv")
    cholesterol = audit.loc[
        (audit["feature"] == "chol") & (audit["rule"] == "zero treated as missing")
    ].copy()
    cohort_order = ["cleveland", "hungarian", "switzerland", "va_long_beach"]
    cholesterol["source_cohort"] = pd.Categorical(
        cholesterol["source_cohort"], categories=cohort_order, ordered=True
    )
    cholesterol = cholesterol.sort_values("source_cohort")
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    bars = ax.bar(
        ["Cleveland", "Hungarian", "Switzerland", "VA Long Beach"],
        cholesterol["n_invalid_cells"],
        color="#B55D60",
    )
    for bar, count in zip(bars, cholesterol["n_invalid_cells"]):
        ax.text(bar.get_x() + bar.get_width() / 2, count + 2, str(int(count)), ha="center")
    ax.set_ylabel("Zero-coded cholesterol records (count)")
    ax.set_title("Anomalous cholesterol values by source cohort")
    ax.text(
        0.02,
        0.96,
        "Zero values are treated as missing and imputed within training folds",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    ax.set_ylim(0, max(cholesterol["n_invalid_cells"]) * 1.22)
    save_figure(fig, "figure_zero_cholesterol_by_cohort.png")


def correlation_outputs(analytic: pd.DataFrame) -> None:
    variables = [*FEATURE_SETS["full13"], TARGET]
    correlation = analytic[variables].corr(method="pearson")
    correlation.to_csv(OUTPUT_ROOT / "tables" / "pearson_correlation_matrix.csv")
    target_correlations = (
        correlation[TARGET]
        .drop(TARGET)
        .rename("pearson_r")
        .to_frame()
        .assign(abs_pearson_r=lambda frame: frame["pearson_r"].abs())
        .sort_values("abs_pearson_r", ascending=False)
    )
    target_correlations.to_csv(
        OUTPUT_ROOT / "tables" / "target_correlations_ranked.csv", index_label="feature"
    )
    requested_pairs = [
        ("age", "chol"),
        ("age", "trestbps"),
        ("trestbps", "chol"),
        ("age", "thalach"),
    ]
    pd.DataFrame(
        [
            {"feature_1": first, "feature_2": second, "pearson_r": correlation.loc[first, second]}
            for first, second in requested_pairs
        ]
    ).to_csv(OUTPUT_ROOT / "tables" / "reported_pair_correlations.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.4, 8.4))
    sns.heatmap(
        correlation,
        cmap="vlag",
        center=0,
        vmin=-1,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation coefficient (r)"},
        ax=ax,
    )
    ax.set_title("Pairwise Pearson correlation matrix (available-case estimates)")
    save_figure(fig, "figure_correlation_heatmap.png")


def sex_distribution_figure(analytic: pd.DataFrame) -> None:
    grouped = (
        analytic.groupby(["sex", TARGET]).size().rename("n").reset_index()
    )
    grouped["sex_label"] = grouped["sex"].map({0.0: "Female", 1.0: "Male"})
    grouped["percent_within_sex"] = grouped.groupby("sex")["n"].transform(
        lambda values: 100 * values / values.sum()
    )
    grouped.to_csv(OUTPUT_ROOT / "tables" / "sex_outcome_counts.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    sns.barplot(
        data=grouped,
        x="sex_label",
        y="percent_within_sex",
        hue=TARGET,
        palette={0: "#4C78A8", 1: "#E17C47"},
        ax=ax,
    )
    for patch, (_, row) in zip(ax.patches, grouped.sort_values([TARGET, "sex"]).iterrows()):
        if np.isfinite(patch.get_height()):
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                patch.get_height() + 1.2,
                f"{row['percent_within_sex']:.1f}%\n(n={int(row['n'])})",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xlabel("Sex")
    ax.set_ylabel("Within-sex percentage")
    ax.set_ylim(0, 100)
    ax.set_title("Angiographic CAD outcome by sex")
    ax.legend(title="CAD outcome", labels=["Absent (0)", "Present (1)"])
    save_figure(fig, "figure_sex_outcome_counts.png")


def performance_tables_and_figure() -> None:
    combined_rows = []
    for feature_set in FEATURE_SETS:
        table = pd.read_csv(OUTPUT_ROOT / "tables" / f"evaluation_metrics__{feature_set}.csv")
        combined_rows.append(table)
    combined = pd.concat(combined_rows, ignore_index=True)
    combined.to_csv(OUTPUT_ROOT / "tables" / "evaluation_metrics_all_feature_sets.csv", index=False)

    primary = combined.loc[combined["feature_set"] == PRIMARY_FEATURE_SET].copy()
    primary["display_name"] = primary["model"].map(DISPLAY_NAMES).fillna(primary["model"])
    primary.to_csv(OUTPUT_ROOT / "tables" / "manuscript_model_metrics_primary.csv", index=False)

    plot_data = primary.loc[
        ~primary["model"].isin(["hard_voting_majority", "ann_shallow", "ann_deep"])
    ].sort_values("roc_auc")
    y_positions = np.arange(len(plot_data))
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharey=True)
    for ax, metric, title in [
        (axes[0], "accuracy", "Accuracy (95% bootstrap CI)"),
        (axes[1], "roc_auc", "AUROC (95% bootstrap CI)"),
    ]:
        values = plot_data[metric].to_numpy()
        lower = values - plot_data[f"{metric}_ci_low"].to_numpy()
        upper = plot_data[f"{metric}_ci_high"].to_numpy() - values
        colors = ["#D65F5F" if name.startswith("soft_voting") else "#4C78A8" for name in plot_data["model"]]
        ax.errorbar(
            values,
            y_positions,
            xerr=np.vstack([lower, upper]),
            fmt="none",
            ecolor="#666666",
            capsize=3,
            alpha=0.8,
        )
        ax.scatter(values, y_positions, c=colors, s=38, zorder=3)
        ax.set_xlabel(title)
        ax.set_xlim(0.65, 1.0)
        ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    axes[0].set_yticks(y_positions, plot_data["display_name"])
    fig.suptitle("Primary non-invasive model comparison on untouched evaluation data")
    fig.tight_layout()
    save_figure(fig, "figure_primary_model_performance.png")


def roc_calibration_confusion_figures(predictions: pd.DataFrame, metadata: dict[str, object]) -> None:
    y_true = predictions[TARGET].to_numpy(dtype=int)
    feature_metadata = next(
        item for item in metadata["feature_set_results"] if item["feature_set"] == PRIMARY_FEATURE_SET
    )
    components = feature_metadata["selected_components"]
    soft_name = feature_metadata["soft_ensemble_name"]
    shown = [*components, soft_name]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for model in shown:
        probability = predictions[f"{model}__probability"].to_numpy()
        fpr, tpr, _ = roc_curve(y_true, probability)
        model_auc = auc(fpr, tpr)
        label = f"{DISPLAY_NAMES.get(model, model)} (AUC={model_auc:.3f})"
        linewidth = 2.5 if model == soft_name else 1.6
        ax.plot(fpr, tpr, lw=linewidth, label=label)
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Chance")
    ax.set_xlabel("False-positive rate (1 - specificity)")
    ax.set_ylabel("True-positive rate (sensitivity)")
    ax.set_title("ROC curves from positive-class probabilities")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, "figure_primary_roc_probability_based.png")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for model in shown:
        probability = predictions[f"{model}__probability"].to_numpy()
        observed, predicted = calibration_curve(y_true, probability, n_bins=8, strategy="quantile")
        brier = np.mean((probability - y_true) ** 2)
        linewidth = 2.5 if model == soft_name else 1.6
        ax.plot(
            predicted,
            observed,
            marker="o",
            lw=linewidth,
            label=f"{DISPLAY_NAMES.get(model, model)} (Brier={brier:.3f})",
        )
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1, label="Ideal")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event frequency")
    ax.set_title("Probability calibration on evaluation data")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, "figure_primary_calibration.png")

    prediction = predictions[f"{soft_name}__prediction"].to_numpy(dtype=int)
    matrix = confusion_matrix(y_true, prediction, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.0, 4.4))
    display = ConfusionMatrixDisplay(matrix, display_labels=["No CAD (0)", "CAD (1)"])
    display.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    display.im_.colorbar.set_label("Patient count")
    ax.set_title("Calibrated soft-voting confusion matrix")
    save_figure(fig, "figure_primary_ensemble_confusion_matrix.png")


def oob_and_ann_figures() -> None:
    oob = pd.read_csv(OUTPUT_ROOT / "tables" / f"random_forest_oob__{PRIMARY_FEATURE_SET}.csv")
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    ax.plot(oob["n_estimators"], oob["oob_error"], marker="o", color="#4C78A8")
    best = oob.loc[oob["oob_error"].idxmin()]
    ax.scatter([best["n_estimators"]], [best["oob_error"]], color="#D65F5F", zorder=3)
    ax.annotate(
        f"minimum={best['oob_error']:.3f}",
        (best["n_estimators"], best["oob_error"]),
        xytext=(8, 8),
        textcoords="offset points",
    )
    ax.set_xlabel("Number of trees")
    ax.set_ylabel("Out-of-bag error")
    ax.set_title("Random-forest OOB error trajectory")
    save_figure(fig, "figure_random_forest_oob.png")

    ann = pd.read_csv(OUTPUT_ROOT / "tables" / f"ann_ablation__{PRIMARY_FEATURE_SET}.csv")
    ann["display_name"] = ann["model"].map(DISPLAY_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1))
    sns.barplot(data=ann, x="display_name", y="roc_auc", color="#4C78A8", ax=axes[0])
    sns.barplot(data=ann, x="display_name", y="log_loss", color="#E17C47", ax=axes[1])
    axes[0].set_ylabel("Development out-of-fold AUROC")
    axes[1].set_ylabel("Development out-of-fold log loss")
    for ax in axes:
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("ANN architecture ablation under the same validation protocol")
    fig.tight_layout()
    save_figure(fig, "figure_ann_ablation.png")


def interpretation_outputs(
    analytic: pd.DataFrame, predictions: pd.DataFrame, metadata: dict[str, object]
) -> None:
    feature_metadata = next(
        item for item in metadata["feature_set_results"] if item["feature_set"] == PRIMARY_FEATURE_SET
    )
    soft_name = feature_metadata["soft_ensemble_name"]
    model = joblib.load(OUTPUT_ROOT / "models" / f"{PRIMARY_FEATURE_SET}__{soft_name}.joblib")
    row_ids = set(predictions["row_id"])
    evaluation = analytic.loc[analytic["row_id"].isin(row_ids)].copy()
    evaluation = evaluation.set_index("row_id").loc[predictions["row_id"]].reset_index()
    X_evaluation = evaluation[FEATURE_SETS[PRIMARY_FEATURE_SET]]
    y_evaluation = evaluation[TARGET].astype(int)

    # Repeated prediction through KNN/iterative imputers can emit benign
    # floating-point warnings from scikit-learn's missing-distance algebra.
    # Suppress those warnings here so a reproducible interpretation run does
    # not create multi-megabyte logs; non-finite predictions would still make
    # the AUROC calculation fail. Twenty permutations balance stability and
    # execution time for this 276-record evaluation set.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        importance = permutation_importance(
            model,
            X_evaluation,
            y_evaluation,
            scoring="roc_auc",
            n_repeats=20,
            random_state=RANDOM_SEED,
            n_jobs=1,
        )
    importance_table = pd.DataFrame(
        {
            "feature": FEATURE_SETS[PRIMARY_FEATURE_SET],
            "mean_auc_decrease": importance.importances_mean,
            "std_auc_decrease": importance.importances_std,
        }
    ).sort_values("mean_auc_decrease", ascending=False)
    importance_table.to_csv(
        OUTPUT_ROOT / "tables" / "permutation_importance_primary_ensemble.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    plot = importance_table.sort_values("mean_auc_decrease")
    colors = ["#B55D60" if value < 0 else "#4C78A8" for value in plot["mean_auc_decrease"]]
    ax.barh(
        plot["feature"],
        plot["mean_auc_decrease"],
        xerr=plot["std_auc_decrease"],
        color=colors,
        alpha=0.9,
        capsize=2,
    )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Decrease in evaluation AUROC after permutation")
    ax.set_title("Permutation importance of the primary soft-voting ensemble")
    save_figure(fig, "figure_permutation_importance.png")

    support_rows = []
    partial_dependence_rows = []
    bootstrap_rng = np.random.default_rng(RANDOM_SEED)
    bootstrap_indices = bootstrap_rng.integers(
        0, len(X_evaluation), size=(2_000, len(X_evaluation))
    )
    for feature in ["age", "oldpeak", "thalach"]:
        observed = X_evaluation[feature].dropna()
        bins = pd.qcut(observed, q=min(10, observed.nunique()), duplicates="drop")
        counts = bins.value_counts(sort=False)
        for interval, count in counts.items():
            support_rows.append(
                {
                    "feature": feature,
                    "interval": str(interval),
                    "n": int(count),
                }
            )

        # Compute one-way partial dependence explicitly. The public sklearn
        # display helper can treat a custom ensemble as a constant-response
        # estimator; direct calls also make every numerical curve value
        # exportable for the supplement. The interval bootstraps evaluation
        # records, so it represents uncertainty in the empirical population
        # average and not uncertainty in fitted model parameters.
        grid = np.linspace(
            float(observed.quantile(0.05)),
            float(observed.quantile(0.95)),
            20,
        )
        prediction_rows = []
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            for grid_value in grid:
                modified = X_evaluation.copy()
                modified[feature] = grid_value
                prediction_rows.append(model.predict_proba(modified)[:, 1])
        prediction_matrix = np.asarray(prediction_rows)
        mean_prediction = prediction_matrix.mean(axis=1)
        bootstrap_means = prediction_matrix[:, bootstrap_indices].mean(axis=2)
        lower = np.quantile(bootstrap_means, 0.025, axis=1)
        upper = np.quantile(bootstrap_means, 0.975, axis=1)
        for grid_value, mean_value, low_value, high_value in zip(
            grid, mean_prediction, lower, upper
        ):
            partial_dependence_rows.append(
                {
                    "feature": feature,
                    "grid_value": grid_value,
                    "mean_predicted_probability": mean_value,
                    "bootstrap_ci_low": low_value,
                    "bootstrap_ci_high": high_value,
                }
            )

        fig, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(grid, mean_prediction, color="#4C78A8", lw=2)
        ax.fill_between(grid, lower, upper, color="#4C78A8", alpha=0.18)
        y_low = min(float(lower.min()), float(mean_prediction.min()))
        y_high = max(float(upper.max()), float(mean_prediction.max()))
        margin = max(0.01, 0.08 * (y_high - y_low))
        ax.set_ylim(y_low - margin, y_high + margin)
        rug_y = y_low - 0.55 * margin
        ax.plot(observed, np.full(len(observed), rug_y), "|", color="black", alpha=0.25, ms=8)
        ax.set_xlabel(feature)
        ax.set_ylabel("Mean predicted CAD probability")
        ax.set_title(f"Partial dependence for {feature} with 95% bootstrap interval")
        save_figure(fig, f"figure_pdp_{feature}.png")
    pd.DataFrame(support_rows).to_csv(
        OUTPUT_ROOT / "tables" / "pdp_support_counts.csv", index=False
    )
    pd.DataFrame(partial_dependence_rows).to_csv(
        OUTPUT_ROOT / "tables" / "partial_dependence_primary_ensemble.csv", index=False
    )


def main() -> None:
    analytic = pd.read_csv(OUTPUT_ROOT / "data" / "analytic_920.csv")
    metadata = json.loads((OUTPUT_ROOT / "logs" / "run_metadata.json").read_text())
    predictions = pd.read_csv(
        OUTPUT_ROOT / "data" / f"evaluation_predictions__{PRIMARY_FEATURE_SET}.csv"
    )

    target_figures(analytic)
    invalid_cholesterol_figure()
    correlation_outputs(analytic)
    sex_distribution_figure(analytic)
    performance_tables_and_figure()
    roc_calibration_confusion_figures(predictions, metadata)
    oob_and_ann_figures()
    interpretation_outputs(analytic, predictions, metadata)
    print(f"Figures written to {OUTPUT_ROOT / 'figures'}")


if __name__ == "__main__":
    main()
