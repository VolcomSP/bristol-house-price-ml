"""Model checkpoint 2: enrich the property-only linear model with neighbourhood conditions."""

from __future__ import annotations

import hashlib
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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    median_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2022_2024.csv"
BASELINE_CV = PROJECT_ROOT / "outputs" / "models" / "04_baseline_cross_validation_metrics.csv"
BASELINE_TEST = PROJECT_ROOT / "outputs" / "models" / "04_baseline_test_metrics.csv"
BASELINE_PREDICTIONS = PROJECT_ROOT / "outputs" / "models" / "04_baseline_2024_predictions.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "models"
MODEL_FOLDER = PROJECT_ROOT / "models"

CV_METRICS = OUTPUT_FOLDER / "05_enriched_cross_validation_metrics.csv"
TEST_METRICS = OUTPUT_FOLDER / "05_enriched_test_metrics.csv"
COEFFICIENTS = OUTPUT_FOLDER / "05_enriched_coefficients.csv"
TEST_PREDICTIONS = OUTPUT_FOLDER / "05_enriched_2024_predictions.csv"
ERROR_BY_TYPE = OUTPUT_FOLDER / "05_enriched_error_by_property_type.csv"
MONTHLY_PERFORMANCE = OUTPUT_FOLDER / "05_enriched_monthly_performance.csv"
MODEL_COMPARISON = OUTPUT_FOLDER / "05_baseline_vs_enriched_comparison.csv"
NEIGHBOURHOOD_EFFECTS = OUTPUT_FOLDER / "05_neighbourhood_feature_effects.csv"
ACTUAL_PREDICTED_CHART = OUTPUT_FOLDER / "05_enriched_actual_vs_predicted.png"
COMPARISON_CHART = OUTPUT_FOLDER / "05_baseline_vs_enriched_performance.png"
FINDINGS = OUTPUT_FOLDER / "05_enriched_linear_findings.md"
MODEL_METADATA = OUTPUT_FOLDER / "05_enriched_linear.metadata.json"
MANIFEST = OUTPUT_FOLDER / "05_enriched_linear.validation.json"
MODEL_FILE = MODEL_FOLDER / "05_enriched_linear_model.joblib"

CATEGORICAL_FEATURES = ["property_type", "duration", "old_new"]
BASE_NUMERIC_FEATURES = ["month_index", "month_sin", "month_cos"]
NEIGHBOURHOOD_FEATURES = ["log1p_total_crime", "claimant_rate_percent"]
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + NEIGHBOURHOOD_FEATURES
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

NAIVE_NAME = "Training-median benchmark"
BASELINE_NAME = "Property-only linear regression"
BASELINE_SMEAR_NAME = "Property-only linear regression with smearing"
ENRICHED_NAME = "Enriched linear regression"
ENRICHED_SMEAR_NAME = "Enriched linear regression with smearing"

NAVY = "#172554"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#D97706"
PURPLE = "#7C3AED"
GREY = "#64748B"
LIGHT_GREY = "#E2E8F0"
TYPE_COLORS = {
    "Detached": PURPLE,
    "Flat/Maisonette": BLUE,
    "Semi-detached": ORANGE,
    "Terraced": TEAL,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def money(value: float) -> str:
    return f"£{value:,.0f}"


def set_chart_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": LIGHT_GREY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "xtick.color": GREY,
            "ytick.color": GREY,
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.7,
        }
    )


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    engineered = data.copy()
    engineered["date_of_transfer"] = pd.to_datetime(engineered["date_of_transfer"])
    month_number = engineered["date_of_transfer"].dt.month
    engineered["month_index"] = (
        (engineered["date_of_transfer"].dt.year - 2022) * 12 + month_number - 1
    ).astype(float)
    engineered["month_sin"] = np.sin(2 * np.pi * month_number / 12)
    engineered["month_cos"] = np.cos(2 * np.pi * month_number / 12)
    engineered["log1p_total_crime"] = np.log1p(engineered["total_crime"].astype(float))
    engineered["claimant_rate_percent"] = engineered["claimant_rate_percent"].astype(float)
    return engineered


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )


def smearing_factor(y_log: np.ndarray, predicted_log: np.ndarray) -> float:
    return float(np.mean(np.exp(y_log - predicted_log)))


def metric_row(
    model_name: str,
    evaluation_period: str,
    y_price: np.ndarray,
    predicted_price: np.ndarray,
    y_log: np.ndarray,
    predicted_log: np.ndarray,
) -> dict[str, object]:
    return {
        "model": model_name,
        "evaluation_period": evaluation_period,
        "rows": len(y_price),
        "mae_gbp": mean_absolute_error(y_price, predicted_price),
        "rmse_gbp": root_mean_squared_error(y_price, predicted_price),
        "median_absolute_error_gbp": median_absolute_error(y_price, predicted_price),
        "median_absolute_percentage_error_percent": (
            np.median(np.abs(y_price - predicted_price) / np.abs(y_price)) * 100
        ),
        "r2_price": r2_score(y_price, predicted_price),
        "mae_log_price": mean_absolute_error(y_log, predicted_log),
        "rmse_log_price": root_mean_squared_error(y_log, predicted_log),
        "r2_log_price": r2_score(y_log, predicted_log),
    }


def grouped_price_metrics(
    data: pd.DataFrame, group_column: str, model_name: str
) -> pd.DataFrame:
    prediction_column = {
        ENRICHED_NAME: "predicted_price",
        ENRICHED_SMEAR_NAME: "smearing_predicted_price",
    }[model_name]
    rows: list[dict[str, object]] = []
    for group_name, group in data.groupby(group_column, observed=True):
        row = metric_row(
            model_name,
            str(group_name),
            group["price"].to_numpy(),
            group[prediction_column].to_numpy(),
            group["log_price"].to_numpy(),
            group["predicted_log_price"].to_numpy(),
        )
        row[group_column] = group_name
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(DATASET, low_memory=False)
    cohort = engineer_features(source.loc[source["ppd_category"].eq("A")].copy())
    cohort["log_price"] = np.log(cohort["price"])
    cohort = cohort.sort_values(["date_of_transfer", "transaction_id"]).reset_index(drop=True)
    development = cohort.loc[cohort["sale_year"].le(2023)].copy()
    test = cohort.loc[cohort["sale_year"].eq(2024)].copy()

    baseline_cv = pd.read_csv(BASELINE_CV)
    baseline_test = pd.read_csv(BASELINE_TEST)
    baseline_predictions = pd.read_csv(BASELINE_PREDICTIONS)

    assert len(source) == 21_968
    assert len(cohort) == 18_670
    assert len(development) == 12_785
    assert len(test) == 5_885
    assert development["date_of_transfer"].max() < test["date_of_transfer"].min()
    assert set(development["transaction_id"]).isdisjoint(set(test["transaction_id"]))
    assert development[MODEL_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert test[MODEL_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert np.isfinite(development[MODEL_FEATURES + ["log_price"]].select_dtypes("number")).all().all()
    assert np.isfinite(test[MODEL_FEATURES + ["log_price"]].select_dtypes("number")).all().all()
    assert len(baseline_predictions) == len(test)
    assert baseline_predictions["transaction_id"].tolist() == test["transaction_id"].tolist()
    assert baseline_cv["evaluation_period"].nunique() == 3

    fold_definitions = [
        ("Fold 1", "2022-12", "2023-01", "2023-04"),
        ("Fold 2", "2023-04", "2023-05", "2023-08"),
        ("Fold 3", "2023-08", "2023-09", "2023-12"),
    ]
    cv_rows: list[dict[str, object]] = []
    base_pipeline = build_pipeline()
    for fold_name, train_end, validation_start, validation_end in fold_definitions:
        fold_train = development.loc[development["sale_month"].le(train_end)].copy()
        fold_validation = development.loc[
            development["sale_month"].between(validation_start, validation_end)
        ].copy()
        assert fold_train["date_of_transfer"].max() < fold_validation["date_of_transfer"].min()

        model = clone(base_pipeline)
        model.fit(fold_train[MODEL_FEATURES], fold_train["log_price"])
        train_log_prediction = model.predict(fold_train[MODEL_FEATURES])
        fold_smearing = smearing_factor(
            fold_train["log_price"].to_numpy(), train_log_prediction
        )
        validation_log_prediction = model.predict(fold_validation[MODEL_FEATURES])
        direct_prediction = np.exp(validation_log_prediction)
        smearing_prediction = direct_prediction * fold_smearing

        for row in [
            metric_row(
                ENRICHED_NAME,
                fold_name,
                fold_validation["price"].to_numpy(),
                direct_prediction,
                fold_validation["log_price"].to_numpy(),
                validation_log_prediction,
            ),
            metric_row(
                ENRICHED_SMEAR_NAME,
                fold_name,
                fold_validation["price"].to_numpy(),
                smearing_prediction,
                fold_validation["log_price"].to_numpy(),
                validation_log_prediction,
            ),
        ]:
            row.update(
                {
                    "training_start": fold_train["date_of_transfer"].min().date().isoformat(),
                    "training_end": fold_train["date_of_transfer"].max().date().isoformat(),
                    "training_rows": len(fold_train),
                    "validation_start": fold_validation["date_of_transfer"].min().date().isoformat(),
                    "validation_end": fold_validation["date_of_transfer"].max().date().isoformat(),
                    "smearing_factor": fold_smearing
                    if row["model"] == ENRICHED_SMEAR_NAME
                    else np.nan,
                }
            )
            cv_rows.append(row)
    cv_metrics = pd.DataFrame(cv_rows)
    cv_metrics.to_csv(CV_METRICS, index=False, float_format="%.6f")

    final_model = clone(base_pipeline)
    final_model.fit(development[MODEL_FEATURES], development["log_price"])
    development_log_prediction = final_model.predict(development[MODEL_FEATURES])
    final_smearing = smearing_factor(
        development["log_price"].to_numpy(), development_log_prediction
    )
    test_log_prediction = final_model.predict(test[MODEL_FEATURES])
    test_price_prediction = np.exp(test_log_prediction)
    test_smearing_price_prediction = test_price_prediction * final_smearing

    test_metrics = pd.DataFrame(
        [
            metric_row(
                ENRICHED_NAME,
                "2024 holdout",
                test["price"].to_numpy(),
                test_price_prediction,
                test["log_price"].to_numpy(),
                test_log_prediction,
            ),
            metric_row(
                ENRICHED_SMEAR_NAME,
                "2024 holdout",
                test["price"].to_numpy(),
                test_smearing_price_prediction,
                test["log_price"].to_numpy(),
                test_log_prediction,
            ),
        ]
    )
    test_metrics.to_csv(TEST_METRICS, index=False, float_format="%.6f")

    preprocessor = final_model.named_steps["preprocessor"]
    regressor = final_model.named_steps["regressor"]
    coefficient_table = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "coefficient_log_price": regressor.coef_,
        }
    )
    coefficient_table["approximate_percent_change"] = (
        np.exp(coefficient_table["coefficient_log_price"]) - 1
    ) * 100
    coefficient_table["interpretation"] = coefficient_table["feature"].map(
        {
            "property_type_F": "Flat/Maisonette relative to Detached",
            "property_type_S": "Semi-detached relative to Detached",
            "property_type_T": "Terraced relative to Detached",
            "duration_L": "Leasehold relative to Freehold",
            "old_new_Y": "New build relative to Established",
            "month_index": "One additional month from January 2022",
            "month_sin": "Seasonal sine component; not interpreted alone",
            "month_cos": "Seasonal cosine component; not interpreted alone",
            "log1p_total_crime": "One-unit increase in log(1 + monthly LSOA crime)",
            "claimant_rate_percent": "One percentage-point increase in claimant rate",
        }
    )
    intercept = pd.DataFrame(
        [
            {
                "feature": "intercept",
                "coefficient_log_price": regressor.intercept_,
                "approximate_percent_change": np.nan,
                "interpretation": "Reference: Detached, Freehold, Established at time origin",
            }
        ]
    )
    coefficient_table = pd.concat([intercept, coefficient_table], ignore_index=True)
    coefficient_table.to_csv(COEFFICIENTS, index=False, float_format="%.8f")

    coefficient_lookup = coefficient_table.set_index("feature")
    effect_rows: list[dict[str, object]] = []
    for feature, original_feature, unit_definition in [
        (
            "log1p_total_crime",
            "total_crime",
            "One-unit increase in log(1 + monthly LSOA crime)",
        ),
        (
            "claimant_rate_percent",
            "claimant_rate_percent",
            "One percentage-point increase in claimant rate",
        ),
    ]:
        coefficient = float(coefficient_lookup.loc[feature, "coefficient_log_price"])
        standard_deviation = float(development[feature].std())
        effect_rows.append(
            {
                "feature": feature,
                "source_variable": original_feature,
                "coefficient_log_price": coefficient,
                "unit_definition": unit_definition,
                "approximate_percent_change_per_unit": (np.exp(coefficient) - 1) * 100,
                "development_standard_deviation": standard_deviation,
                "approximate_percent_change_per_one_sd": (
                    np.exp(coefficient * standard_deviation) - 1
                )
                * 100,
                "development_p25": development[feature].quantile(0.25),
                "development_median": development[feature].median(),
                "development_p75": development[feature].quantile(0.75),
            }
        )
    neighbourhood_effects = pd.DataFrame(effect_rows)
    neighbourhood_effects.to_csv(NEIGHBOURHOOD_EFFECTS, index=False, float_format="%.8f")

    predictions = test[
        [
            "transaction_id",
            "date_of_transfer",
            "sale_month",
            "price",
            "property_type",
            "property_type_label",
            "duration",
            "old_new",
            "analysis_lsoa_code",
            "total_crime",
            "claimant_rate_percent",
        ]
    ].copy()
    predictions["log1p_total_crime"] = test["log1p_total_crime"].to_numpy()
    predictions["log_price"] = test["log_price"].to_numpy()
    predictions["predicted_log_price"] = test_log_prediction
    predictions["predicted_price"] = test_price_prediction
    predictions["smearing_predicted_price"] = test_smearing_price_prediction
    predictions["baseline_predicted_log_price"] = baseline_predictions[
        "predicted_log_price"
    ].to_numpy()
    predictions["baseline_predicted_price"] = baseline_predictions[
        "predicted_price"
    ].to_numpy()
    predictions["prediction_error"] = predictions["predicted_price"] - predictions["price"]
    predictions["absolute_error"] = predictions["prediction_error"].abs()
    predictions["baseline_absolute_error"] = (
        predictions["baseline_predicted_price"] - predictions["price"]
    ).abs()
    predictions["absolute_error_improvement_vs_baseline"] = (
        predictions["baseline_absolute_error"] - predictions["absolute_error"]
    )
    predictions["absolute_percentage_error_percent"] = (
        predictions["absolute_error"] / predictions["price"] * 100
    )
    predictions.to_csv(TEST_PREDICTIONS, index=False, float_format="%.6f")

    error_by_type = pd.concat(
        [
            grouped_price_metrics(predictions, "property_type_label", ENRICHED_NAME),
            grouped_price_metrics(predictions, "property_type_label", ENRICHED_SMEAR_NAME),
        ],
        ignore_index=True,
    )
    error_by_type.to_csv(ERROR_BY_TYPE, index=False, float_format="%.6f")

    monthly_rows: list[dict[str, object]] = []
    for sale_month, group in predictions.groupby("sale_month", observed=True):
        row = metric_row(
            ENRICHED_NAME,
            str(sale_month),
            group["price"].to_numpy(),
            group["predicted_price"].to_numpy(),
            group["log_price"].to_numpy(),
            group["predicted_log_price"].to_numpy(),
        )
        row.update(
            {
                "sale_month": sale_month,
                "actual_median_price": group["price"].median(),
                "baseline_predicted_median_price": group["baseline_predicted_price"].median(),
                "enriched_predicted_median_price": group["predicted_price"].median(),
                "actual_mean_price": group["price"].mean(),
                "baseline_predicted_mean_price": group["baseline_predicted_price"].mean(),
                "enriched_predicted_mean_price": group["predicted_price"].mean(),
            }
        )
        monthly_rows.append(row)
    monthly_performance = pd.DataFrame(monthly_rows).sort_values("sale_month")
    monthly_performance.to_csv(MONTHLY_PERFORMANCE, index=False, float_format="%.6f")

    comparison_metrics = pd.concat(
        [
            baseline_test.loc[
                baseline_test["model"].isin([NAIVE_NAME, BASELINE_NAME, BASELINE_SMEAR_NAME])
            ].copy(),
            test_metrics.copy(),
        ],
        ignore_index=True,
    )
    property_baseline = comparison_metrics.loc[
        comparison_metrics["model"].eq(BASELINE_NAME)
    ].iloc[0]
    comparison_metrics["mae_change_vs_property_baseline_gbp"] = (
        comparison_metrics["mae_gbp"] - property_baseline["mae_gbp"]
    )
    comparison_metrics["mae_improvement_vs_property_baseline_percent"] = (
        (property_baseline["mae_gbp"] - comparison_metrics["mae_gbp"])
        / property_baseline["mae_gbp"]
        * 100
    )
    comparison_metrics["rmse_change_vs_property_baseline_gbp"] = (
        comparison_metrics["rmse_gbp"] - property_baseline["rmse_gbp"]
    )
    comparison_metrics["rmse_improvement_vs_property_baseline_percent"] = (
        (property_baseline["rmse_gbp"] - comparison_metrics["rmse_gbp"])
        / property_baseline["rmse_gbp"]
        * 100
    )
    comparison_metrics["r2_price_change_vs_property_baseline"] = (
        comparison_metrics["r2_price"] - property_baseline["r2_price"]
    )
    comparison_metrics.to_csv(MODEL_COMPARISON, index=False, float_format="%.6f")

    enriched_direct = comparison_metrics.loc[
        comparison_metrics["model"].eq(ENRICHED_NAME)
    ].iloc[0]
    enriched_smear = comparison_metrics.loc[
        comparison_metrics["model"].eq(ENRICHED_SMEAR_NAME)
    ].iloc[0]

    set_chart_style()
    fig, axis = plt.subplots(figsize=(8.4, 7.0))
    for property_type in ["Detached", "Flat/Maisonette", "Semi-detached", "Terraced"]:
        group = predictions.loc[predictions["property_type_label"].eq(property_type)]
        axis.scatter(
            group["price"],
            group["predicted_price"],
            s=17,
            alpha=0.42,
            color=TYPE_COLORS[property_type],
            label=property_type,
            edgecolor="none",
        )
    chart_min = min(predictions["price"].min(), predictions["predicted_price"].min())
    chart_max = max(predictions["price"].max(), predictions["predicted_price"].max())
    axis.plot(
        [chart_min, chart_max],
        [chart_min, chart_max],
        color=NAVY,
        linestyle="--",
        linewidth=1.6,
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Actual 2024 sale price")
    axis.set_ylabel("Predicted sale price")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axis.grid()
    axis.legend(frameon=False, ncol=2, loc="upper left")
    axis.text(
        0.97,
        0.05,
        f"MAE {money(enriched_direct['mae_gbp'])}\n"
        f"RMSE {money(enriched_direct['rmse_gbp'])}\n"
        f"R² {enriched_direct['r2_price']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=NAVY,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": LIGHT_GREY},
    )
    axis.set_title("Enriched linear model: actual versus predicted", pad=14)
    fig.suptitle("2024 holdout predictions", fontsize=12, color=GREY, y=0.96)
    fig.tight_layout()
    fig.savefig(ACTUAL_PREDICTED_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7))
    chart_keys = [BASELINE_NAME, ENRICHED_NAME, ENRICHED_SMEAR_NAME]
    chart_labels = ["Property-only\ndirect exp", "Enriched\ndirect exp", "Enriched\nsmearing"]
    chart_data = comparison_metrics.set_index("model").loc[chart_keys]
    x = np.arange(len(chart_labels))
    width = 0.32
    mae_values = chart_data["mae_gbp"].to_numpy()
    rmse_values = chart_data["rmse_gbp"].to_numpy()
    bars_mae = axes[0].bar(x - width / 2, mae_values, width, label="MAE", color=BLUE)
    bars_rmse = axes[0].bar(x + width / 2, rmse_values, width, label="RMSE", color=ORANGE)
    axes[0].set_xticks(x, chart_labels)
    axes[0].set_ylabel("Error (£)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[0].set_title("2024 error comparison")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    for bars in [bars_mae, bars_rmse]:
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rmse_values) * 0.025,
                f"£{bar.get_height() / 1_000:.0f}k",
                ha="center",
                va="bottom",
                fontsize=9,
                color=NAVY,
            )
    axes[0].set_ylim(0, max(rmse_values) * 1.20)

    r2_values = chart_data["r2_price"].to_numpy()
    r2_bars = axes[1].bar(x, r2_values, width=0.55, color=[GREY, TEAL, PURPLE])
    axes[1].set_xticks(x, chart_labels)
    axes[1].set_ylabel("Price R²")
    axes[1].set_title("Explained price variation")
    axes[1].grid(axis="y")
    axes[1].set_ylim(0, max(r2_values) * 1.28)
    for bar in r2_bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(r2_values) * 0.035,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=NAVY,
        )
    fig.suptitle(
        "Do neighbourhood conditions improve the linear model?",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(COMPARISON_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    joblib.dump(
        {
            "pipeline": final_model,
            "smearing_factor": final_smearing,
            "model_features": MODEL_FEATURES,
            "target": "log(price)",
            "training_period": "2022-01 to 2023-12",
        },
        MODEL_FILE,
    )

    cv_combined = pd.concat(
        [
            baseline_cv.loc[baseline_cv["model"].eq(BASELINE_NAME)].copy(),
            cv_metrics.loc[cv_metrics["model"].eq(ENRICHED_NAME)].copy(),
        ],
        ignore_index=True,
    )
    cv_summary = cv_combined.groupby("model", as_index=False).agg(
        mean_mae_gbp=("mae_gbp", "mean"),
        mean_rmse_gbp=("rmse_gbp", "mean"),
        mean_r2_price=("r2_price", "mean"),
        mean_r2_log_price=("r2_log_price", "mean"),
    ).set_index("model")
    cv_mae_improvement = (
        cv_summary.loc[BASELINE_NAME, "mean_mae_gbp"]
        - cv_summary.loc[ENRICHED_NAME, "mean_mae_gbp"]
    ) / cv_summary.loc[BASELINE_NAME, "mean_mae_gbp"] * 100
    cv_rmse_improvement = (
        cv_summary.loc[BASELINE_NAME, "mean_rmse_gbp"]
        - cv_summary.loc[ENRICHED_NAME, "mean_rmse_gbp"]
    ) / cv_summary.loc[BASELINE_NAME, "mean_rmse_gbp"] * 100
    crime_effect = neighbourhood_effects.set_index("feature").loc["log1p_total_crime"]
    claimant_effect = neighbourhood_effects.set_index("feature").loc[
        "claimant_rate_percent"
    ]

    findings = f"""# Enriched linear model: crime and claimant conditions

## Purpose

This checkpoint tests whether two neighbourhood-condition variables improve 2024 house-price predictions beyond the property-only linear baseline:

- `log1p(total_crime)`: the natural logarithm of one plus the number of police-recorded crimes in the property's 2011 LSOA during the sale month.
- `claimant_rate_percent`: claimant count as a percentage of the fixed 2020 working-age population in the property's 2011 LSOA.

Everything else is held constant: Category A transactions, `log(price)` target, property and time features, linear regression, expanding-window validation, and the 2024 holdout.

## Comparable sample

| Dataset | Period | Transactions |
|---|---|---:|
| Development | 2022–2023 | {len(development):,} |
| Test | 2024 | {len(test):,} |

Both neighbourhood variables are present for every transaction. Adding them therefore does not alter the modelling sample.

## Time-based validation

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only linear regression | {money(cv_summary.loc[BASELINE_NAME, 'mean_mae_gbp'])} | {money(cv_summary.loc[BASELINE_NAME, 'mean_rmse_gbp'])} | {cv_summary.loc[BASELINE_NAME, 'mean_r2_price']:.3f} |
| Enriched linear regression | {money(cv_summary.loc[ENRICHED_NAME, 'mean_mae_gbp'])} | {money(cv_summary.loc[ENRICHED_NAME, 'mean_rmse_gbp'])} | {cv_summary.loc[ENRICHED_NAME, 'mean_r2_price']:.3f} |

Across the three validation periods, enrichment reduces mean MAE by **{cv_mae_improvement:.1f}%** and mean RMSE by **{cv_rmse_improvement:.1f}%**. The enriched direct transformation has lower MAE than its smeared alternative in every validation fold, so direct exponentiation remains the primary prediction method.

## Final 2024 comparison

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only linear regression | {money(property_baseline['mae_gbp'])} | {money(property_baseline['rmse_gbp'])} | {property_baseline['r2_price']:.3f} | {property_baseline['r2_log_price']:.3f} |
| Enriched linear regression | {money(enriched_direct['mae_gbp'])} | {money(enriched_direct['rmse_gbp'])} | {enriched_direct['r2_price']:.3f} | {enriched_direct['r2_log_price']:.3f} |
| Enriched linear regression with smearing | {money(enriched_smear['mae_gbp'])} | {money(enriched_smear['rmse_gbp'])} | {enriched_smear['r2_price']:.3f} | {enriched_smear['r2_log_price']:.3f} |

Relative to the property-only model, the direct enriched model reduces MAE by **{enriched_direct['mae_improvement_vs_property_baseline_percent']:.1f}%** ({-enriched_direct['mae_change_vs_property_baseline_gbp']:,.0f} pounds) and reduces RMSE by **{enriched_direct['rmse_improvement_vs_property_baseline_percent']:.1f}%** ({-enriched_direct['rmse_change_vs_property_baseline_gbp']:,.0f} pounds). Its price R² increases by **{enriched_direct['r2_price_change_vs_property_baseline']:+.3f}**. The smeared enriched result reduces RMSE by **{enriched_smear['rmse_improvement_vs_property_baseline_percent']:.1f}%**, but has higher MAE than direct exponentiation.

## Neighbourhood coefficients

Holding the included property and time features constant:

- A one-standard-deviation increase in `log1p(total_crime)` is associated with an approximate **{crime_effect['approximate_percent_change_per_one_sd']:+.1f}%** change in predicted price.
- A one-standard-deviation increase in `claimant_rate_percent` is associated with an approximate **{claimant_effect['approximate_percent_change_per_one_sd']:+.1f}%** change in predicted price.

These are conditional associations, not causal effects. Crime, claimant rate and omitted aspects of location are related, so the coefficients should not be interpreted as the effect of changing crime or claimant levels while everything else in the real world remains fixed.

## Interpretation

This is the first direct test of the research question. The comparison shows whether the two neighbourhood-condition variables add predictive information under the same linear specification. It does not yet show whether they improve nonlinear machine-learning models, and it does not include an LSOA identifier or detailed structural property attributes.

## Limitations

- Crime is measured for the entire sale month; for a strictly prospective prediction system, a lagged crime measure would avoid using incidents recorded after an early-month sale.
- Claimant rates use a fixed 2020 working-age population denominator.
- Neighbourhood variables may proxy for unobserved location quality, housing stock and amenities.
- Land Registry data still lacks bedrooms, floor area, condition and detailed amenities.
- Linear regression assumes additive, constant effects and cannot automatically learn interactions or thresholds.

## Next modelling experiment

Apply the same property-only versus enriched feature comparison to nonlinear tree-based models. A random forest and a gradient-boosting model can test whether crime and claimant conditions become more useful when interactions and nonlinear relationships are allowed.
"""
    FINDINGS.write_text(findings, encoding="utf-8")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": ENRICHED_NAME,
        "estimator": "sklearn.linear_model.LinearRegression",
        "sklearn_version": sklearn.__version__,
        "cohort": "PPD Category A",
        "target": "log(price)",
        "primary_back_transformation": "exp(predicted_log_price); selected using validation MAE and compared directly with the property-only baseline",
        "sensitivity_back_transformation": "exp(predicted_log_price) multiplied by development-set smearing factor",
        "smearing_factor": final_smearing,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "neighbourhood_features": {
            "log1p_total_crime": "log(1 + total_crime) in 2011 LSOA and sale month",
            "claimant_rate_percent": "claimant count / 2020 working-age population * 100",
        },
        "development_period": "2022-01-04 to 2023-12-22",
        "test_period": "2024-01-02 to 2024-12-31",
        "development_rows": len(development),
        "test_rows": len(test),
        "randomness": "none; deterministic estimator and chronological split",
        "model_file": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
        "model_sha256": sha256(MODEL_FILE),
    }
    MODEL_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    output_files = [
        CV_METRICS,
        TEST_METRICS,
        COEFFICIENTS,
        TEST_PREDICTIONS,
        ERROR_BY_TYPE,
        MONTHLY_PERFORMANCE,
        MODEL_COMPARISON,
        NEIGHBOURHOOD_EFFECTS,
        ACTUAL_PREDICTED_CHART,
        COMPARISON_CHART,
        FINDINGS,
        MODEL_METADATA,
        MODEL_FILE,
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(DATASET.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(DATASET),
        "baseline_cross_validation_metrics_sha256": sha256(BASELINE_CV),
        "baseline_test_metrics_sha256": sha256(BASELINE_TEST),
        "baseline_predictions_sha256": sha256(BASELINE_PREDICTIONS),
        "checks": {
            "source_rows": len(source),
            "category_a_rows": len(cohort),
            "development_rows": len(development),
            "test_rows": len(test),
            "split_overlap_transaction_ids": len(
                set(development["transaction_id"]).intersection(set(test["transaction_id"]))
            ),
            "development_missing_model_values": int(
                development[MODEL_FEATURES + ["log_price"]].isna().sum().sum()
            ),
            "test_missing_model_values": int(
                test[MODEL_FEATURES + ["log_price"]].isna().sum().sum()
            ),
            "prediction_rows": len(predictions),
            "prediction_duplicate_transaction_ids": int(
                predictions["transaction_id"].duplicated().sum()
            ),
            "prediction_nonfinite_values": int(
                (
                    ~np.isfinite(
                        predictions[
                            [
                                "predicted_log_price",
                                "predicted_price",
                                "smearing_predicted_price",
                            ]
                        ]
                    )
                ).sum().sum()
            ),
            "baseline_transaction_order_match": bool(
                baseline_predictions["transaction_id"].tolist()
                == predictions["transaction_id"].tolist()
            ),
            "cross_validation_folds": len(fold_definitions),
        },
        "test_metrics": {
            "mae_gbp": float(enriched_direct["mae_gbp"]),
            "rmse_gbp": float(enriched_direct["rmse_gbp"]),
            "r2_price": float(enriched_direct["r2_price"]),
            "r2_log_price": float(enriched_direct["r2_log_price"]),
            "mae_improvement_vs_property_baseline_percent": float(
                enriched_direct["mae_improvement_vs_property_baseline_percent"]
            ),
            "rmse_improvement_vs_property_baseline_percent": float(
                enriched_direct["rmse_improvement_vs_property_baseline_percent"]
            ),
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Development rows: {len(development):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Smearing factor: {final_smearing:.6f}")
    print(f"Property-only MAE: {money(property_baseline['mae_gbp'])}")
    print(f"Enriched MAE: {money(enriched_direct['mae_gbp'])}")
    print(
        "Enriched MAE improvement: "
        f"{enriched_direct['mae_improvement_vs_property_baseline_percent']:.2f}%"
    )
    print(f"Property-only RMSE: {money(property_baseline['rmse_gbp'])}")
    print(f"Enriched RMSE: {money(enriched_direct['rmse_gbp'])}")
    print(f"Property-only price R2: {property_baseline['r2_price']:.4f}")
    print(f"Enriched price R2: {enriched_direct['r2_price']:.4f}")
    print(f"Saved outputs to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
