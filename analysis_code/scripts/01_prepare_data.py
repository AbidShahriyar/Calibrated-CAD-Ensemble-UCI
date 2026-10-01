"""Prepare and audit the four downloaded UCI Heart Disease cohorts.

This stage never overwrites raw inputs. It exports a 920-row analytic file in
which explicit missing codes, invalid category codes, zero cholesterol, and zero
resting blood pressure are represented as missing values for fold-specific
imputation by the modeling stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline_utils import (
    CATEGORICAL_LEVELS,
    FEATURE_SETS,
    OUTPUT_ROOT,
    RAW_DATA_ROOT,
    ROW_ID,
    SOURCE,
    TARGET,
    UCI_COLUMNS,
    ensure_output_directories,
)


SOURCE_FILES = {
    "cleveland": ("processed.cleveland.data", ","),
    "hungarian": ("reprocessed.hungarian.data", r"\s+"),
    "switzerland": ("processed.switzerland.data", ","),
    "va_long_beach": ("processed.va.data", ","),
}
EXPECTED_ROWS = {
    "cleveland": 303,
    "hungarian": 294,
    "switzerland": 123,
    "va_long_beach": 200,
}


def read_source(cohort: str, filename: str, separator: str) -> pd.DataFrame:
    path = RAW_DATA_ROOT / filename
    frame = pd.read_csv(
        path,
        sep=separator,
        header=None,
        names=UCI_COLUMNS,
        na_values=["?", -9, -9.0, "-9", "-9.0"],
        engine="python",
    )
    if len(frame) != EXPECTED_ROWS[cohort]:
        raise ValueError(f"{cohort}: expected {EXPECTED_ROWS[cohort]} rows, found {len(frame)}")
    for column in UCI_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.insert(0, SOURCE, cohort)
    frame.insert(0, ROW_ID, [f"{cohort}_{index:03d}" for index in range(len(frame))])
    return frame


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    ensure_output_directories()
    data_output = OUTPUT_ROOT / "data"
    frames = [read_source(cohort, *spec) for cohort, spec in SOURCE_FILES.items()]
    raw = pd.concat(frames, ignore_index=True)
    if len(raw) != 920:
        raise ValueError(f"Expected 920 concatenated records, found {len(raw)}")

    raw[TARGET] = (raw["num"] > 0).astype(int)

    # Construct record-exclusion sensitivity populations for comparison.
    exclusion_base = raw.replace([np.inf, -np.inf], np.nan)
    exclusion_740 = exclusion_base.drop(columns=["slope", "ca", "thal"]).dropna(
        subset=FEATURE_SETS["common10"]
    )
    exclusion_661 = exclusion_740.loc[
        (exclusion_740["chol"] != 0) & (exclusion_740["trestbps"] != 0)
    ]
    sensitivity_export_columns = [
        ROW_ID,
        SOURCE,
        *FEATURE_SETS["common10"],
        "num",
        TARGET,
    ]
    exclusion_740[sensitivity_export_columns].to_csv(
        data_output / "record_exclusion_before_zero_deletion_740.csv", index=False
    )
    exclusion_661[sensitivity_export_columns].to_csv(
        data_output / "record_exclusion_after_zero_deletion_661.csv", index=False
    )
    pd.DataFrame(
        {
            ROW_ID: raw[ROW_ID],
            SOURCE: raw[SOURCE],
            "chol_was_zero": raw["chol"].eq(0),
            "trestbps_was_zero": raw["trestbps"].eq(0),
            "in_complete_case_740": raw[ROW_ID].isin(exclusion_740[ROW_ID]),
            "in_after_zero_deletion_661": raw[ROW_ID].isin(exclusion_661[ROW_ID]),
        }
    ).to_csv(data_output / "record_exclusion_membership.csv", index=False)

    flow_rows = []
    for stage, frame in [
        ("downloaded_and_concatenated", raw),
        ("complete_case_after_column_drop", exclusion_740),
        ("after_zero_value_row_deletion", exclusion_661),
    ]:
        for cohort in [*SOURCE_FILES, "all"]:
            subset = frame if cohort == "all" else frame.loc[frame[SOURCE] == cohort]
            flow_rows.append(
                {
                    "stage": stage,
                    "source_cohort": cohort,
                    "n": len(subset),
                    "target_0": int((subset[TARGET] == 0).sum()),
                    "target_1": int((subset[TARGET] == 1).sum()),
                }
            )
    pd.DataFrame(flow_rows).to_csv(data_output / "cohort_flow.csv", index=False)

    audit = raw.copy()
    invalid_rows: list[dict[str, object]] = []
    invalid_by_cohort_rows: list[dict[str, object]] = []

    for feature in ["chol", "trestbps"]:
        invalid = audit[feature].eq(0)
        invalid_rows.append(
            {
                "feature": feature,
                "rule": "zero treated as missing",
                "n_invalid_cells": int(invalid.sum()),
                "n_affected_rows": int(invalid.sum()),
            }
        )
        for cohort, cohort_index in audit.groupby(SOURCE).groups.items():
            cohort_invalid = invalid.loc[cohort_index]
            invalid_by_cohort_rows.append(
                {
                    "source_cohort": cohort,
                    "feature": feature,
                    "rule": "zero treated as missing",
                    "n_invalid_cells": int(cohort_invalid.sum()),
                }
            )
        audit.loc[invalid, feature] = np.nan

    for feature, levels in CATEGORICAL_LEVELS.items():
        observed = audit[feature].notna()
        invalid = observed & ~audit[feature].isin(levels)
        invalid_rows.append(
            {
                "feature": feature,
                "rule": f"outside allowed levels {levels}",
                "n_invalid_cells": int(invalid.sum()),
                "n_affected_rows": int(invalid.sum()),
            }
        )
        for cohort, cohort_index in audit.groupby(SOURCE).groups.items():
            cohort_invalid = invalid.loc[cohort_index]
            invalid_by_cohort_rows.append(
                {
                    "source_cohort": cohort,
                    "feature": feature,
                    "rule": f"outside allowed levels {levels}",
                    "n_invalid_cells": int(cohort_invalid.sum()),
                }
            )
        audit.loc[invalid, feature] = np.nan

    pd.DataFrame(invalid_rows).to_csv(data_output / "invalid_value_audit.csv", index=False)
    pd.DataFrame(invalid_by_cohort_rows).to_csv(
        data_output / "invalid_value_by_cohort.csv", index=False
    )

    feature_columns = [column for column in UCI_COLUMNS if column != "num"]
    overall_missing = pd.DataFrame(
        {
            "feature": feature_columns,
            "missing_n": [int(audit[column].isna().sum()) for column in feature_columns],
            "missing_percent": [100.0 * audit[column].isna().mean() for column in feature_columns],
        }
    )
    overall_missing.to_csv(data_output / "missingness_overall.csv", index=False)

    cohort_missing_rows = []
    for cohort, subset in audit.groupby(SOURCE, sort=False):
        for feature in feature_columns:
            cohort_missing_rows.append(
                {
                    "source_cohort": cohort,
                    "feature": feature,
                    "n": len(subset),
                    "missing_n": int(subset[feature].isna().sum()),
                    "missing_percent": 100.0 * subset[feature].isna().mean(),
                }
            )
    pd.DataFrame(cohort_missing_rows).to_csv(
        data_output / "missingness_by_cohort.csv", index=False
    )

    raw.groupby([SOURCE, "num"], dropna=False).size().rename("n").reset_index().to_csv(
        data_output / "target_distribution_multiclass.csv", index=False
    )
    audit.groupby([SOURCE, TARGET], dropna=False).size().rename("n").reset_index().to_csv(
        data_output / "target_distribution_binary.csv", index=False
    )

    analytic_columns = [ROW_ID, SOURCE, *UCI_COLUMNS, TARGET]
    audit[analytic_columns].to_csv(data_output / "analytic_920.csv", index=False)

    dictionary = [
        ("age", "Age in years", "continuous"),
        ("sex", "Sex: 0 female, 1 male", "categorical"),
        ("cp", "Chest-pain type", "categorical"),
        ("trestbps", "Resting blood pressure, mm Hg", "continuous"),
        ("chol", "Serum cholesterol, mg/dL", "continuous"),
        ("fbs", "Fasting blood sugar >120 mg/dL", "categorical"),
        ("restecg", "Resting electrocardiographic result", "categorical"),
        ("thalach", "Maximum heart rate achieved", "continuous"),
        ("exang", "Exercise-induced angina", "categorical"),
        ("oldpeak", "ST depression induced by exercise relative to rest", "continuous"),
        ("slope", "Slope of peak exercise ST segment", "categorical"),
        ("ca", "Major vessels colored by fluoroscopy", "categorical"),
        ("thal", "Thallium-test result: normal/fixed/reversible defect", "categorical"),
        ("num", "Original angiographic disease-status category, 0-4", "outcome"),
        (TARGET, "Binary outcome: 0 if num=0; 1 if num>0", "outcome"),
        (SOURCE, "Source UCI cohort", "provenance"),
        (ROW_ID, "Stable source-specific row identifier", "identifier"),
    ]
    pd.DataFrame(dictionary, columns=["variable", "description", "role"]).to_csv(
        data_output / "data_dictionary.csv", index=False
    )

    manifest = []
    for cohort, (filename, _) in SOURCE_FILES.items():
        path = RAW_DATA_ROOT / filename
        manifest.append(
            {
                "source_cohort": cohort,
                "path": str(path.relative_to(RAW_DATA_ROOT.parent.parent)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    pd.DataFrame(manifest).to_csv(data_output / "raw_file_manifest.csv", index=False)

    metadata = {
        "source_rows": EXPECTED_ROWS,
        "concatenated_n": int(len(raw)),
        "record_exclusion_after_complete_case_n": int(len(exclusion_740)),
        "record_exclusion_after_zero_deletion_n": int(len(exclusion_661)),
        "analytic_n": int(len(audit)),
        "rows_dropped_for_predictors": 0,
        "feature_sets": FEATURE_SETS,
    }
    (OUTPUT_ROOT / "logs" / "data_preparation_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
