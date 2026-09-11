"""Sensitivity checkpoint: one-month-lagged neighbourhood conditions.

The main enriched models use crime and claimant conditions from the property
sale month. This script creates a stricter prospective version in which each
sale receives the previous month's neighbourhood information. Three random
forests are fitted on the same transactions:

1. property and time features only;
2. contemporaneous crime and claimant rate;
3. one-month-lagged crime and claimant rate.

Hyperparameters are frozen from the earlier validated random-forest
checkpoints so that feature timing, rather than additional tuning, drives the
sensitivity comparison.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODULE_PATH = Path(__file__).with_name("06_property_only_random_forest.py")
BASE_SPEC = importlib.util.spec_from_file_location("model_checkpoint_helpers", BASE_MODULE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError(f"Cannot load shared helpers from {BASE_MODULE_PATH}")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

DATASET = PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2022_2024.csv"
CRIME_FEATURES = (
    PROJECT_ROOT / "outputs" / "candidate" / "crime_features_2022_2024_lsoa11_candidate.csv"
)
CLAIMANT_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "candidate"
    / "claimant_features_2022_2024_lsoa11_candidate.csv"
)
PROPERTY_METADATA = PROJECT_ROOT / "outputs" / "models" / "06_property_rf.metadata.json"
ENRICHED_METADATA = PROJECT_ROOT / "outputs" / "models" / "07_enriched_rf.metadata.json"
MODEL_OUTPUTS = PROJECT_ROOT / "outputs" / "models"
MODEL_FOLDER = PROJECT_ROOT / "models"

FEATURE_ALIGNMENT = MODEL_OUTPUTS / "09_lagged_neighbourhood_feature_alignment.csv"
CV_METRICS = MODEL_OUTPUTS / "09_lagged_sensitivity_cross_validation_metrics.csv"
TEST_METRICS = MODEL_OUTPUTS / "09_lagged_sensitivity_test_metrics.csv"
TEST_PREDICTIONS = MODEL_OUTPUTS / "09_lagged_sensitivity_2024_predictions.csv"
ERROR_BY_TYPE = MODEL_OUTPUTS / "09_lagged_sensitivity_error_by_property_type.csv"
MONTHLY_PERFORMANCE = MODEL_OUTPUTS / "09_lagged_sensitivity_monthly_performance.csv"
MODEL_COMPARISON = MODEL_OUTPUTS / "09_lagged_sensitivity_model_comparison.csv"
PERFORMANCE_CHART = MODEL_OUTPUTS / "09_lagged_sensitivity_performance.png"
MONTHLY_CHART = MODEL_OUTPUTS / "09_lagged_sensitivity_monthly_mae.png"
FINDINGS = MODEL_OUTPUTS / "09_lagged_neighbourhood_findings.md"
MODEL_METADATA = MODEL_OUTPUTS / "09_lagged_neighbourhood.metadata.json"
MANIFEST = MODEL_OUTPUTS / "09_lagged_neighbourhood.validation.json"
PROPERTY_MODEL_FILE = MODEL_FOLDER / "09_property_rf_same_cohort.joblib"
CURRENT_MODEL_FILE = MODEL_FOLDER / "09_current_enriched_rf_same_cohort.joblib"
LAGGED_MODEL_FILE = MODEL_FOLDER / "09_lagged_enriched_random_forest.joblib"

CATEGORICAL_FEATURES = ["property_type", "duration", "old_new"]
TIME_FEATURES = ["month_index", "month_sin", "month_cos"]
PROPERTY_FEATURES = CATEGORICAL_FEATURES + TIME_FEATURES
CURRENT_NEIGHBOURHOOD_FEATURES = [
    "log_current_total_crime",
    "current_claimant_rate_percent",
]
LAGGED_NEIGHBOURHOOD_FEATURES = [
    "log_lagged_total_crime",
    "lagged_claimant_rate_percent",
]
CURRENT_FEATURES = PROPERTY_FEATURES + CURRENT_NEIGHBOURHOOD_FEATURES
LAGGED_FEATURES = PROPERTY_FEATURES + LAGGED_NEIGHBOURHOOD_FEATURES

PROPERTY_NAME = "Property-only random forest, common cohort"
CURRENT_NAME = "Contemporaneous enriched random forest, common cohort"
LAGGED_NAME = "Lagged enriched random forest"
PROPERTY_SMEAR_NAME = f"{PROPERTY_NAME} with smearing"
CURRENT_SMEAR_NAME = f"{CURRENT_NAME} with smearing"
LAGGED_SMEAR_NAME = f"{LAGGED_NAME} with smearing"

RANDOM_STATE = 42
N_ESTIMATORS = 240
FOLD_DEFINITIONS = BASE.FOLD_DEFINITIONS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def money(value: float) -> str:
    return f"£{value:,.0f}"


def load_selected_parameters() -> tuple[dict[str, object], dict[str, object]]:
    property_metadata = json.loads(PROPERTY_METADATA.read_text(encoding="utf-8"))
    enriched_metadata = json.loads(ENRICHED_METADATA.read_text(encoding="utf-8"))
    return property_metadata["selected_parameters"], enriched_metadata["selected_parameters"]


def build_pipeline(features: list[str], parameters: dict[str, object]) -> Pipeline:
    numeric_features = [feature for feature in features if feature not in CATEGORICAL_FEATURES]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", numeric_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    forest = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        criterion="squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=parameters["max_depth"],
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        max_features=float(parameters["max_features"]),
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", forest)])


def prepare_feature_alignment() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    sales = pd.read_csv(DATASET, low_memory=False)
    crime = pd.read_csv(CRIME_FEATURES)
    claimant = pd.read_csv(CLAIMANT_FEATURES)

    assert len(sales) == 21_968
    assert len(crime) == 9_468
    assert len(claimant) == 9_468
    assert crime[["lsoa_code", "month"]].duplicated().sum() == 0
    assert claimant[["lsoa11cd", "month"]].duplicated().sum() == 0

    monthly = crime[["lsoa_code", "month", "total_crime"]].merge(
        claimant[["lsoa11cd", "month", "claimant_count", "claimant_rate_percent"]],
        left_on=["lsoa_code", "month"],
        right_on=["lsoa11cd", "month"],
        how="inner",
        validate="one_to_one",
    )
    assert len(monthly) == 9_468
    assert monthly[["lsoa_code", "month"]].duplicated().sum() == 0
    assert monthly[["total_crime", "claimant_count", "claimant_rate_percent"]].isna().sum().sum() == 0

    monthly = monthly.rename(
        columns={
            "lsoa_code": "analysis_lsoa_code",
            "month": "neighbourhood_source_month",
            "total_crime": "lagged_total_crime",
            "claimant_count": "lagged_claimant_count",
            "claimant_rate_percent": "lagged_claimant_rate_percent",
        }
    ).drop(columns="lsoa11cd")
    monthly["sale_month"] = (
        pd.PeriodIndex(monthly["neighbourhood_source_month"], freq="M") + 1
    ).astype(str)

    alignment = sales[
        [
            "transaction_id",
            "sale_month",
            "analysis_lsoa_code",
            "total_crime",
            "claimant_count",
            "claimant_rate_percent",
        ]
    ].rename(
        columns={
            "total_crime": "current_total_crime",
            "claimant_count": "current_claimant_count",
            "claimant_rate_percent": "current_claimant_rate_percent",
        }
    )
    alignment["expected_neighbourhood_source_month"] = (
        pd.PeriodIndex(alignment["sale_month"], freq="M") - 1
    ).astype(str)
    alignment = alignment.merge(
        monthly,
        on=["analysis_lsoa_code", "sale_month"],
        how="left",
        validate="many_to_one",
    )
    alignment["lagged_neighbourhood_matched"] = alignment[
        "neighbourhood_source_month"
    ].notna()

    expected_missing = alignment["sale_month"].eq("2022-01")
    actual_missing = ~alignment["lagged_neighbourhood_matched"]
    assert actual_missing.equals(expected_missing)
    assert (
        alignment.loc[
            alignment["lagged_neighbourhood_matched"], "neighbourhood_source_month"
        ].to_numpy()
        == alignment.loc[
            alignment["lagged_neighbourhood_matched"],
            "expected_neighbourhood_source_month",
        ].to_numpy()
    ).all()

    audit = sales.merge(
        alignment[
            [
                "transaction_id",
                "expected_neighbourhood_source_month",
                "neighbourhood_source_month",
                "lagged_total_crime",
                "lagged_claimant_count",
                "lagged_claimant_rate_percent",
                "lagged_neighbourhood_matched",
            ]
        ],
        on="transaction_id",
        how="left",
        validate="one_to_one",
    )
    audit["date_of_transfer"] = pd.to_datetime(audit["date_of_transfer"])
    month_number = audit["date_of_transfer"].dt.month
    audit["month_index"] = (
        (audit["date_of_transfer"].dt.year - 2022) * 12 + month_number - 1
    ).astype(float)
    audit["month_sin"] = np.sin(2 * np.pi * month_number / 12)
    audit["month_cos"] = np.cos(2 * np.pi * month_number / 12)
    audit["log_price"] = np.log(audit["price"])
    audit["current_total_crime"] = audit["total_crime"]
    audit["current_claimant_count"] = audit["claimant_count"]
    audit["current_claimant_rate_percent"] = audit["claimant_rate_percent"]
    audit["log_current_total_crime"] = np.log1p(audit["current_total_crime"])
    audit["log_lagged_total_crime"] = np.log1p(audit["lagged_total_crime"])

    audit_columns = [
        "transaction_id",
        "date_of_transfer",
        "sale_month",
        "analysis_lsoa_code",
        "expected_neighbourhood_source_month",
        "neighbourhood_source_month",
        "current_total_crime",
        "lagged_total_crime",
        "current_claimant_count",
        "lagged_claimant_count",
        "current_claimant_rate_percent",
        "lagged_claimant_rate_percent",
        "lagged_neighbourhood_matched",
    ]
    audit[audit_columns].to_csv(FEATURE_ALIGNMENT, index=False, float_format="%.8f")

    checks = {
        "source_sales_rows": len(sales),
        "crime_lsoa_month_rows": len(crime),
        "claimant_lsoa_month_rows": len(claimant),
        "monthly_feature_rows": len(monthly),
        "feature_alignment_rows": len(alignment),
        "expected_january_2022_unmatched_rows": int(expected_missing.sum()),
        "category_a_january_2022_unmatched_rows": int(
            (
                audit["ppd_category"].eq("A")
                & ~audit["lagged_neighbourhood_matched"]
            ).sum()
        ),
        "actual_unmatched_rows": int(actual_missing.sum()),
        "matched_rows": int(alignment["lagged_neighbourhood_matched"].sum()),
    }
    return sales, audit, checks


def model_names(variant: str) -> tuple[str, str]:
    return {
        "property_only": (PROPERTY_NAME, PROPERTY_SMEAR_NAME),
        "current_enriched": (CURRENT_NAME, CURRENT_SMEAR_NAME),
        "lagged_enriched": (LAGGED_NAME, LAGGED_SMEAR_NAME),
    }[variant]


def evaluate_cross_validation(
    variant: str,
    features: list[str],
    parameters: dict[str, object],
    fold_data: list[tuple[str, pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    direct_name, smear_name = model_names(variant)
    rows: list[dict[str, object]] = []
    for fold_name, fold_train, fold_validation in fold_data:
        model = build_pipeline(features, parameters)
        model.fit(fold_train[features], fold_train["log_price"])
        train_log_prediction = model.predict(fold_train[features])
        smear = BASE.smearing_factor(
            fold_train["log_price"].to_numpy(), train_log_prediction
        )
        validation_log_prediction = model.predict(fold_validation[features])
        direct_prediction = np.exp(validation_log_prediction)
        smearing_prediction = direct_prediction * smear
        for row in [
            BASE.metric_row(
                direct_name,
                fold_name,
                fold_validation["price"].to_numpy(),
                direct_prediction,
                fold_validation["log_price"].to_numpy(),
                validation_log_prediction,
            ),
            BASE.metric_row(
                smear_name,
                fold_name,
                fold_validation["price"].to_numpy(),
                smearing_prediction,
                fold_validation["log_price"].to_numpy(),
                validation_log_prediction,
            ),
        ]:
            row.update(
                {
                    "variant": variant,
                    "training_start": fold_train["date_of_transfer"].min().date().isoformat(),
                    "training_end": fold_train["date_of_transfer"].max().date().isoformat(),
                    "training_rows": len(fold_train),
                    "validation_start": fold_validation["date_of_transfer"].min().date().isoformat(),
                    "validation_end": fold_validation["date_of_transfer"].max().date().isoformat(),
                    "smearing_factor": smear if row["model"] == smear_name else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fit_and_test(
    variant: str,
    features: list[str],
    parameters: dict[str, object],
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[Pipeline, float, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    direct_name, smear_name = model_names(variant)
    model = build_pipeline(features, parameters)
    model.fit(development[features], development["log_price"])
    development_log_prediction = model.predict(development[features])
    smear = BASE.smearing_factor(
        development["log_price"].to_numpy(), development_log_prediction
    )
    test_log_prediction = model.predict(test[features])
    direct_prediction = np.exp(test_log_prediction)
    smearing_prediction = direct_prediction * smear
    metrics = pd.DataFrame(
        [
            BASE.metric_row(
                direct_name,
                "2024 holdout",
                test["price"].to_numpy(),
                direct_prediction,
                test["log_price"].to_numpy(),
                test_log_prediction,
            ),
            BASE.metric_row(
                smear_name,
                "2024 holdout",
                test["price"].to_numpy(),
                smearing_prediction,
                test["log_price"].to_numpy(),
                test_log_prediction,
            ),
        ]
    )
    metrics.insert(2, "variant", variant)
    return model, smear, metrics, test_log_prediction, direct_prediction, smearing_prediction


def mean_cv_metric(table: pd.DataFrame, model_name: str) -> pd.Series:
    rows = table.loc[table["model"].eq(model_name)]
    return pd.Series(
        {
            "mae_gbp": rows["mae_gbp"].mean(),
            "rmse_gbp": rows["rmse_gbp"].mean(),
            "r2_price": rows["r2_price"].mean(),
            "r2_log_price": rows["r2_log_price"].mean(),
        }
    )


def direct_metric(table: pd.DataFrame, model_name: str) -> pd.Series:
    return table.loc[table["model"].eq(model_name)].iloc[0]


def main() -> None:
    MODEL_OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    property_parameters, enriched_parameters = load_selected_parameters()
    source_sales, audit, alignment_checks = prepare_feature_alignment()

    cohort = audit.loc[
        audit["ppd_category"].eq("A") & audit["lagged_neighbourhood_matched"]
    ].copy()
    cohort = cohort.sort_values(["date_of_transfer", "transaction_id"]).reset_index(drop=True)
    development = cohort.loc[cohort["sale_year"].le(2023)].copy()
    test = cohort.loc[cohort["sale_year"].eq(2024)].copy()

    assert len(cohort) == 18_228
    assert len(development) == 12_343
    assert len(test) == 5_885
    assert development["sale_month"].min() == "2022-02"
    assert development["date_of_transfer"].max() < test["date_of_transfer"].min()
    assert set(development["transaction_id"]).isdisjoint(set(test["transaction_id"]))
    assert development[LAGGED_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert test[LAGGED_FEATURES + ["log_price"]].isna().sum().sum() == 0

    fold_data: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for fold_name, train_end, validation_start, validation_end in FOLD_DEFINITIONS:
        fold_train = development.loc[development["sale_month"].le(train_end)].copy()
        fold_validation = development.loc[
            development["sale_month"].between(validation_start, validation_end)
        ].copy()
        assert fold_train["date_of_transfer"].max() < fold_validation["date_of_transfer"].min()
        fold_data.append((fold_name, fold_train, fold_validation))

    variants = [
        ("property_only", PROPERTY_FEATURES, property_parameters),
        ("current_enriched", CURRENT_FEATURES, enriched_parameters),
        ("lagged_enriched", LAGGED_FEATURES, enriched_parameters),
    ]
    cv_metrics = pd.concat(
        [
            evaluate_cross_validation(variant, features, parameters, fold_data)
            for variant, features, parameters in variants
        ],
        ignore_index=True,
    )
    cv_metrics.to_csv(CV_METRICS, index=False, float_format="%.6f")

    fitted: dict[str, dict[str, object]] = {}
    test_metric_tables: list[pd.DataFrame] = []
    for variant, features, parameters in variants:
        model, smear, metrics, predicted_log, predicted_price, smearing_price = fit_and_test(
            variant, features, parameters, development, test
        )
        fitted[variant] = {
            "model": model,
            "smear": smear,
            "features": features,
            "parameters": parameters,
            "predicted_log": predicted_log,
            "predicted_price": predicted_price,
            "smearing_price": smearing_price,
        }
        test_metric_tables.append(metrics)
    test_metrics = pd.concat(test_metric_tables, ignore_index=True)
    test_metrics.to_csv(TEST_METRICS, index=False, float_format="%.6f")

    predictions = test[
        [
            "transaction_id",
            "date_of_transfer",
            "sale_month",
            "neighbourhood_source_month",
            "price",
            "property_type",
            "property_type_label",
            "duration",
            "old_new",
            "analysis_lsoa_code",
            "current_total_crime",
            "lagged_total_crime",
            "current_claimant_rate_percent",
            "lagged_claimant_rate_percent",
            "log_price",
        ]
    ].copy()
    for variant, prefix in [
        ("property_only", "property_rf"),
        ("current_enriched", "current_enriched_rf"),
        ("lagged_enriched", "lagged_enriched_rf"),
    ]:
        predictions[f"{prefix}_predicted_log_price"] = fitted[variant]["predicted_log"]
        predictions[f"{prefix}_predicted_price"] = fitted[variant]["predicted_price"]
        predictions[f"{prefix}_smearing_predicted_price"] = fitted[variant]["smearing_price"]
        predictions[f"{prefix}_absolute_error"] = (
            predictions[f"{prefix}_predicted_price"] - predictions["price"]
        ).abs()
    predictions["lagged_improvement_vs_property_gbp"] = (
        predictions["property_rf_absolute_error"]
        - predictions["lagged_enriched_rf_absolute_error"]
    )
    predictions["lagged_change_vs_current_gbp"] = (
        predictions["lagged_enriched_rf_absolute_error"]
        - predictions["current_enriched_rf_absolute_error"]
    )
    predictions.to_csv(TEST_PREDICTIONS, index=False, float_format="%.6f")

    error_rows: list[dict[str, object]] = []
    for model_name, prefix in [
        (PROPERTY_NAME, "property_rf"),
        (CURRENT_NAME, "current_enriched_rf"),
        (LAGGED_NAME, "lagged_enriched_rf"),
    ]:
        for property_type, group in predictions.groupby("property_type_label", observed=True):
            row = BASE.metric_row(
                model_name,
                str(property_type),
                group["price"].to_numpy(),
                group[f"{prefix}_predicted_price"].to_numpy(),
                group["log_price"].to_numpy(),
                group[f"{prefix}_predicted_log_price"].to_numpy(),
            )
            row["property_type_label"] = property_type
            error_rows.append(row)
    error_by_type = pd.DataFrame(error_rows)
    error_by_type.to_csv(ERROR_BY_TYPE, index=False, float_format="%.6f")

    monthly_rows: list[dict[str, object]] = []
    for sale_month, group in predictions.groupby("sale_month", observed=True):
        for model_name, prefix in [
            (PROPERTY_NAME, "property_rf"),
            (CURRENT_NAME, "current_enriched_rf"),
            (LAGGED_NAME, "lagged_enriched_rf"),
        ]:
            row = BASE.metric_row(
                model_name,
                str(sale_month),
                group["price"].to_numpy(),
                group[f"{prefix}_predicted_price"].to_numpy(),
                group["log_price"].to_numpy(),
                group[f"{prefix}_predicted_log_price"].to_numpy(),
            )
            row["sale_month"] = sale_month
            row["neighbourhood_source_month_min"] = group[
                "neighbourhood_source_month"
            ].min()
            row["neighbourhood_source_month_max"] = group[
                "neighbourhood_source_month"
            ].max()
            monthly_rows.append(row)
    monthly_performance = pd.DataFrame(monthly_rows).sort_values(["sale_month", "model"])
    monthly_performance.to_csv(MONTHLY_PERFORMANCE, index=False, float_format="%.6f")

    comparison_rows: list[dict[str, object]] = []
    for model_name in [PROPERTY_NAME, CURRENT_NAME, LAGGED_NAME]:
        cv = mean_cv_metric(cv_metrics, model_name)
        test_result = direct_metric(test_metrics, model_name)
        comparison_rows.append(
            {
                "model": model_name,
                "mean_validation_mae_gbp": cv["mae_gbp"],
                "mean_validation_rmse_gbp": cv["rmse_gbp"],
                "mean_validation_r2_price": cv["r2_price"],
                "test_mae_gbp": test_result["mae_gbp"],
                "test_rmse_gbp": test_result["rmse_gbp"],
                "test_r2_price": test_result["r2_price"],
                "test_r2_log_price": test_result["r2_log_price"],
            }
        )
    model_comparison = pd.DataFrame(comparison_rows)
    property_row = model_comparison.loc[model_comparison["model"].eq(PROPERTY_NAME)].iloc[0]
    current_row = model_comparison.loc[model_comparison["model"].eq(CURRENT_NAME)].iloc[0]
    model_comparison["validation_mae_improvement_vs_property_percent"] = (
        (property_row["mean_validation_mae_gbp"] - model_comparison["mean_validation_mae_gbp"])
        / property_row["mean_validation_mae_gbp"]
        * 100
    )
    model_comparison["test_mae_improvement_vs_property_percent"] = (
        (property_row["test_mae_gbp"] - model_comparison["test_mae_gbp"])
        / property_row["test_mae_gbp"]
        * 100
    )
    model_comparison["test_rmse_improvement_vs_property_percent"] = (
        (property_row["test_rmse_gbp"] - model_comparison["test_rmse_gbp"])
        / property_row["test_rmse_gbp"]
        * 100
    )
    model_comparison["test_mae_change_vs_current_enriched_gbp"] = (
        model_comparison["test_mae_gbp"] - current_row["test_mae_gbp"]
    )
    model_comparison.to_csv(MODEL_COMPARISON, index=False, float_format="%.6f")

    property_direct = direct_metric(test_metrics, PROPERTY_NAME)
    current_direct = direct_metric(test_metrics, CURRENT_NAME)
    lagged_direct = direct_metric(test_metrics, LAGGED_NAME)
    property_cv = mean_cv_metric(cv_metrics, PROPERTY_NAME)
    current_cv = mean_cv_metric(cv_metrics, CURRENT_NAME)
    lagged_cv = mean_cv_metric(cv_metrics, LAGGED_NAME)

    BASE.set_chart_style()
    chart_data = model_comparison.set_index("model").loc[
        [PROPERTY_NAME, CURRENT_NAME, LAGGED_NAME]
    ]
    labels = ["Property-only", "Current-month\nenriched", "Previous-month\nenriched"]
    x = np.arange(len(labels))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7))
    mae_values = chart_data["test_mae_gbp"].to_numpy()
    rmse_values = chart_data["test_rmse_gbp"].to_numpy()
    mae_bars = axes[0].bar(x - width / 2, mae_values, width, label="MAE", color=BASE.BLUE)
    rmse_bars = axes[0].bar(x + width / 2, rmse_values, width, label="RMSE", color=BASE.ORANGE)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Error (£)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[0].set_title("2024 prediction error")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    for bars in [mae_bars, rmse_bars]:
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rmse_values) * 0.025,
                f"£{bar.get_height() / 1_000:.0f}k",
                ha="center",
                va="bottom",
                fontsize=9,
                color=BASE.NAVY,
            )
    axes[0].set_ylim(0, max(rmse_values) * 1.20)

    r2_values = chart_data["test_r2_price"].to_numpy()
    r2_bars = axes[1].bar(x, r2_values, width=0.55, color=[BASE.GREY, BASE.TEAL, BASE.PURPLE])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Price R²")
    axes[1].set_title("Explained price variation")
    axes[1].grid(axis="y")
    axes[1].set_ylim(0, max(r2_values) * 1.28)
    for bar in r2_bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(r2_values) * 0.025,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=BASE.NAVY,
        )
    fig.suptitle(
        "Does predictive improvement survive one-month lagging?",
        fontsize=17,
        fontweight="bold",
        color=BASE.NAVY,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(PERFORMANCE_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    monthly_pivot = monthly_performance.pivot(
        index="sale_month", columns="model", values="mae_gbp"
    )
    fig, axis = plt.subplots(figsize=(10.5, 5.5))
    axis.plot(
        monthly_pivot.index,
        monthly_pivot[CURRENT_NAME],
        marker="o",
        linewidth=2.0,
        color=BASE.TEAL,
        label="Current-month conditions",
    )
    axis.plot(
        monthly_pivot.index,
        monthly_pivot[LAGGED_NAME],
        marker="o",
        linewidth=2.0,
        color=BASE.PURPLE,
        label="Previous-month conditions",
    )
    axis.set_ylabel("Monthly MAE (£)")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axis.set_xlabel("2024 sale month")
    axis.set_title("Current-month and lagged model errors remain similar")
    axis.grid(axis="y")
    axis.legend(frameon=False)
    axis.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(MONTHLY_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    model_files = {
        "property_only": PROPERTY_MODEL_FILE,
        "current_enriched": CURRENT_MODEL_FILE,
        "lagged_enriched": LAGGED_MODEL_FILE,
    }
    for variant, path in model_files.items():
        joblib.dump(
            {
                "pipeline": fitted[variant]["model"],
                "smearing_factor": fitted[variant]["smear"],
                "model_features": fitted[variant]["features"],
                "target": "log(price)",
                "selected_parameters": fitted[variant]["parameters"],
                "n_estimators": N_ESTIMATORS,
                "random_state": RANDOM_STATE,
                "training_period": "2022-02 to 2023-12",
                "neighbourhood_timing": {
                    "property_only": "not applicable",
                    "current_enriched": "sale month",
                    "lagged_enriched": "one calendar month before sale month",
                }[variant],
            },
            path,
        )

    validation_improvement = (
        (property_cv["mae_gbp"] - lagged_cv["mae_gbp"])
        / property_cv["mae_gbp"]
        * 100
    )
    test_improvement = (
        (property_direct["mae_gbp"] - lagged_direct["mae_gbp"])
        / property_direct["mae_gbp"]
        * 100
    )
    lag_penalty = (
        (lagged_direct["mae_gbp"] - current_direct["mae_gbp"])
        / current_direct["mae_gbp"]
        * 100
    )
    use_lagged_for_dashboard = bool(
        lagged_cv["mae_gbp"] < property_cv["mae_gbp"]
        and lagged_direct["mae_gbp"] < property_direct["mae_gbp"]
    )
    dashboard_decision = (
        "Lagged enriched random forest"
        if use_lagged_for_dashboard
        else "Reassess neighbourhood timing before dashboard development"
    )

    findings = f"""# Lagged-neighbourhood sensitivity analysis

## Purpose

The main models associate each sale with crime and claimant conditions from the same calendar month. For an early-month sale, some of that monthly information may not yet have been observed. This checkpoint therefore replaces both neighbourhood variables with values from the previous calendar month.

## Common comparison cohort

| Dataset | Period | Transactions |
|---|---|---:|
| Development | February 2022–December 2023 | {len(development):,} |
| Test | January–December 2024 | {len(test):,} |

The full dataset contains {alignment_checks['expected_january_2022_unmatched_rows']:,} January 2022 transactions without December 2021 source data. Of these, {alignment_checks['category_a_january_2022_unmatched_rows']:,} belong to the Category A modelling cohort and are excluded only from this sensitivity exercise. Every 2024 test transaction has correctly dated previous-month crime and claimant information.

## Controlled design

The three models use identical transactions and time folds. Hyperparameters are frozen from the earlier property-only and enriched random-forest checkpoints. The contemporaneous and lagged enriched models use the same enriched parameters, so the only difference between them is neighbourhood timing.

## Time-based validation

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only random forest | {money(property_cv['mae_gbp'])} | {money(property_cv['rmse_gbp'])} | {property_cv['r2_price']:.3f} |
| Current-month enriched random forest | {money(current_cv['mae_gbp'])} | {money(current_cv['rmse_gbp'])} | {current_cv['r2_price']:.3f} |
| Previous-month enriched random forest | {money(lagged_cv['mae_gbp'])} | {money(lagged_cv['rmse_gbp'])} | {lagged_cv['r2_price']:.3f} |

The lagged model reduces mean validation MAE by **{validation_improvement:.1f}%** relative to the property-only model.

## Final 2024 results

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only random forest | {money(property_direct['mae_gbp'])} | {money(property_direct['rmse_gbp'])} | {property_direct['r2_price']:.3f} | {property_direct['r2_log_price']:.3f} |
| Current-month enriched random forest | {money(current_direct['mae_gbp'])} | {money(current_direct['rmse_gbp'])} | {current_direct['r2_price']:.3f} | {current_direct['r2_log_price']:.3f} |
| Previous-month enriched random forest | {money(lagged_direct['mae_gbp'])} | {money(lagged_direct['rmse_gbp'])} | {lagged_direct['r2_price']:.3f} | {lagged_direct['r2_log_price']:.3f} |

The lagged model reduces 2024 MAE by **{test_improvement:.1f}%** relative to the property-only model. Its MAE changes by **{lag_penalty:+.1f}%** relative to the otherwise identical current-month model.

## Interpretation and decision

The predictive value of neighbourhood conditions {"remains after lagging" if use_lagged_for_dashboard else "does not remain clearly after lagging"}. This reduces concern that the main improvement is explained by using information from later in the sale month. The recommended dashboard model is **{dashboard_decision}** because previous-month inputs have a clearer real-world availability interpretation.

This remains an associational prediction exercise. Lagging the variables addresses timing leakage but does not make crime or claimant conditions causal determinants of price.

## Limitations

- A one-month lag is a practical availability rule rather than a guarantee of each source's exact publication date.
- The claimant-rate denominator remains the fixed 2020 working-age population.
- Neighbourhood variables may proxy for unmeasured amenities, housing stock and location quality.
- The data lacks bedrooms, floor area, condition and detailed property amenities.

## Next checkpoint

Create the final dashboard-ready model bundle and prediction input table, then implement the small visual prediction tool promised in the proposal.
"""
    FINDINGS.write_text(findings, encoding="utf-8")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": "sklearn.ensemble.RandomForestRegressor",
        "sklearn_version": sklearn.__version__,
        "purpose": "one-month-lagged neighbourhood feature sensitivity",
        "cohort": "PPD Category A with previous-month crime and claimant features",
        "target": "log(price)",
        "property_features": PROPERTY_FEATURES,
        "current_enriched_features": CURRENT_FEATURES,
        "lagged_enriched_features": LAGGED_FEATURES,
        "property_parameters": property_parameters,
        "enriched_parameters_frozen_for_current_and_lagged": enriched_parameters,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "development_period": "2022-02 to 2023-12",
        "test_period": "2024-01 to 2024-12",
        "development_rows": len(development),
        "test_rows": len(test),
        "lag_rule": "crime and claimant values from one calendar month before sale_month",
        "dashboard_model_decision": dashboard_decision,
        "model_files": {
            variant: str(path.relative_to(PROJECT_ROOT)) for variant, path in model_files.items()
        },
        "model_sha256": {variant: sha256(path) for variant, path in model_files.items()},
    }
    MODEL_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    output_files = [
        FEATURE_ALIGNMENT,
        CV_METRICS,
        TEST_METRICS,
        TEST_PREDICTIONS,
        ERROR_BY_TYPE,
        MONTHLY_PERFORMANCE,
        MODEL_COMPARISON,
        PERFORMANCE_CHART,
        MONTHLY_CHART,
        FINDINGS,
        MODEL_METADATA,
        PROPERTY_MODEL_FILE,
        CURRENT_MODEL_FILE,
        LAGGED_MODEL_FILE,
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            str(DATASET.relative_to(PROJECT_ROOT)): sha256(DATASET),
            str(CRIME_FEATURES.relative_to(PROJECT_ROOT)): sha256(CRIME_FEATURES),
            str(CLAIMANT_FEATURES.relative_to(PROJECT_ROOT)): sha256(CLAIMANT_FEATURES),
            str(PROPERTY_METADATA.relative_to(PROJECT_ROOT)): sha256(PROPERTY_METADATA),
            str(ENRICHED_METADATA.relative_to(PROJECT_ROOT)): sha256(ENRICHED_METADATA),
        },
        "alignment_checks": alignment_checks,
        "model_checks": {
            "category_a_lag_complete_rows": len(cohort),
            "development_rows": len(development),
            "test_rows": len(test),
            "split_overlap_transaction_ids": len(
                set(development["transaction_id"]).intersection(set(test["transaction_id"]))
            ),
            "development_missing_lagged_model_values": int(
                development[LAGGED_FEATURES + ["log_price"]].isna().sum().sum()
            ),
            "test_missing_lagged_model_values": int(
                test[LAGGED_FEATURES + ["log_price"]].isna().sum().sum()
            ),
            "cross_validation_folds": len(FOLD_DEFINITIONS),
            "prediction_rows": len(predictions),
            "prediction_duplicate_transaction_ids": int(
                predictions["transaction_id"].duplicated().sum()
            ),
            "prediction_nonfinite_values": int(
                (
                    ~np.isfinite(
                        predictions[
                            [
                                "property_rf_predicted_price",
                                "current_enriched_rf_predicted_price",
                                "lagged_enriched_rf_predicted_price",
                            ]
                        ]
                    )
                ).sum().sum()
            ),
        },
        "test_metrics": {
            "property_only_mae_gbp": float(property_direct["mae_gbp"]),
            "current_enriched_mae_gbp": float(current_direct["mae_gbp"]),
            "lagged_enriched_mae_gbp": float(lagged_direct["mae_gbp"]),
            "lagged_improvement_vs_property_percent": float(test_improvement),
            "lagged_change_vs_current_percent": float(lag_penalty),
            "lagged_enriched_rmse_gbp": float(lagged_direct["rmse_gbp"]),
            "lagged_enriched_r2_price": float(lagged_direct["r2_price"]),
        },
        "dashboard_model_decision": dashboard_decision,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Lag-complete development rows: {len(development):,}")
    print(f"2024 test rows: {len(test):,}")
    print(f"Property-only MAE: {money(property_direct['mae_gbp'])}")
    print(f"Current-month enriched MAE: {money(current_direct['mae_gbp'])}")
    print(f"Previous-month enriched MAE: {money(lagged_direct['mae_gbp'])}")
    print(f"Lagged improvement versus property-only: {test_improvement:.1f}%")
    print(f"Dashboard model decision: {dashboard_decision}")


if __name__ == "__main__":
    main()
