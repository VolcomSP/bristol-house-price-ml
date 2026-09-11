"""Validate the included processed artefact without requiring large raw downloads."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2022_2024.csv"
DATASET_VALIDATION = DATASET.with_suffix(".validation.json")
CRIME = PROJECT_ROOT / "outputs" / "candidate" / "crime_features_2022_2024_lsoa11_candidate.csv"
CLAIMANT = PROJECT_ROOT / "outputs" / "candidate" / "claimant_features_2022_2024_lsoa11_candidate.csv"
METRICS = PROJECT_ROOT / "outputs" / "models" / "09_lagged_sensitivity_test_metrics.csv"
MODEL = PROJECT_ROOT / "models" / "09_lagged_enriched_random_forest.joblib"
DASHBOARD_VALIDATION = PROJECT_ROOT / "dashboard" / "dashboard_validation.json"

EXPECTED_DOCUMENTS = [
    "README.md",
    "docs/PROJECT_SCOPE.md",
    "docs/DATA_SOURCES.md",
    "docs/DATA_DICTIONARY.md",
    "docs/METHODOLOGY.md",
    "docs/RESULTS_SUMMARY.md",
    "docs/ETHICS_AND_LIMITATIONS.md",
    "docs/REPRODUCIBILITY.md",
    "dashboard/README.md",
    "dashboard/ARTEFACT_EVALUATION.md",
]
EXPECTED_MODEL_FEATURES = [
    "property_type",
    "duration",
    "old_new",
    "month_index",
    "month_sin",
    "month_cos",
    "log_lagged_total_crime",
    "lagged_claimant_rate_percent",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    data = pd.read_csv(DATASET, low_memory=False)
    validation = json.loads(DATASET_VALIDATION.read_text(encoding="utf-8"))
    assert sha256(DATASET) == validation["output_sha256"]
    assert len(data) == 21_968
    assert len(data.columns) == 53
    assert data["transaction_id"].duplicated().sum() == 0
    assert data["sale_month"].nunique() == 36
    assert data["analysis_lsoa_code"].nunique() == 263
    assert data["lad_code"].eq("E06000023").all()
    assert data["crime_data_matched"].all()
    assert data["claimant_data_matched"].all()
    assert data.loc[data["ppd_category"].eq("A")].shape[0] == 18_670

    crime = pd.read_csv(CRIME)
    claimant = pd.read_csv(CLAIMANT)
    assert len(crime) == len(claimant) == 9_468
    assert crime[["lsoa_code", "month"]].duplicated().sum() == 0
    assert claimant[["lsoa11cd", "month"]].duplicated().sum() == 0
    assert crime.isna().sum().sum() == 0
    assert claimant[
        ["claimant_count", "working_age_population_2020", "claimant_rate_percent"]
    ].isna().sum().sum() == 0
    expected_rate = claimant["claimant_count"] / claimant["working_age_population_2020"] * 100
    np.testing.assert_allclose(claimant["claimant_rate_percent"], expected_rate, rtol=0, atol=1e-6)

    metrics = pd.read_csv(METRICS)
    property_row = metrics.loc[
        metrics["model"].eq("Property-only random forest, common cohort")
    ].iloc[0]
    lagged_row = metrics.loc[metrics["model"].eq("Lagged enriched random forest")].iloc[0]
    assert int(lagged_row["rows"]) == 5_885
    assert abs(float(property_row["mae_gbp"]) - 124_633.526413) < 0.01
    assert abs(float(lagged_row["mae_gbp"]) - 99_654.695789) < 0.01
    improvement = (property_row["mae_gbp"] - lagged_row["mae_gbp"]) / property_row["mae_gbp"] * 100
    assert abs(float(improvement) - 20.041823) < 1e-5

    model_bundle = joblib.load(MODEL)
    assert model_bundle["model_features"] == EXPECTED_MODEL_FEATURES
    assert model_bundle["training_period"] == "2022-02 to 2023-12"
    assert model_bundle["neighbourhood_timing"] == "one calendar month before sale month"

    dashboard_validation = json.loads(DASHBOARD_VALIDATION.read_text(encoding="utf-8"))
    for relative, recorded in dashboard_validation["artefact_files"].items():
        path = PROJECT_ROOT / relative
        assert path.exists(), relative
        assert path.stat().st_size == recorded["bytes"], relative
        assert sha256(path) == recorded["sha256"], relative

    for relative in EXPECTED_DOCUMENTS:
        path = PROJECT_ROOT / relative
        assert path.exists() and path.stat().st_size > 0, relative

    dictionary_text = (PROJECT_ROOT / "docs" / "DATA_DICTIONARY.md").read_text(
        encoding="utf-8"
    )
    column_section = dictionary_text.split("## Columns", 1)[1].split("## Model use", 1)[0]
    documented_columns = re.findall(r"^\| `([^`]+)` \|", column_section, flags=re.MULTILINE)
    assert documented_columns == data.columns.tolist()

    print("Submission artefact validation passed")
    print(f"Transactions: {len(data):,}")
    print(f"Category A modelling transactions: {data['ppd_category'].eq('A').sum():,}")
    print(f"Neighbourhood panel rows: {len(crime):,}")
    print(f"Selected-model 2024 MAE: £{lagged_row['mae_gbp']:,.0f}")
    print(f"MAE improvement over property-only random forest: {improvement:.1f}%")


if __name__ == "__main__":
    main()
