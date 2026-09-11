"""Model checkpoint 1: property-only linear baseline with a 2024 holdout."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
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
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "models"
MODEL_FOLDER = PROJECT_ROOT / "models"

SPLIT_SUMMARY = OUTPUT_FOLDER / "04_baseline_split_summary.csv"
CV_METRICS = OUTPUT_FOLDER / "04_baseline_cross_validation_metrics.csv"
TEST_METRICS = OUTPUT_FOLDER / "04_baseline_test_metrics.csv"
COEFFICIENTS = OUTPUT_FOLDER / "04_baseline_coefficients.csv"
TEST_PREDICTIONS = OUTPUT_FOLDER / "04_baseline_2024_predictions.csv"
ERROR_BY_TYPE = OUTPUT_FOLDER / "04_baseline_error_by_property_type.csv"
MONTHLY_PERFORMANCE = OUTPUT_FOLDER / "04_baseline_monthly_performance.csv"
ACTUAL_PREDICTED_CHART = OUTPUT_FOLDER / "04_baseline_actual_vs_predicted.png"
PERFORMANCE_CHART = OUTPUT_FOLDER / "04_baseline_model_performance.png"
FINDINGS = OUTPUT_FOLDER / "04_property_only_baseline_findings.md"
MODEL_METADATA = OUTPUT_FOLDER / "04_property_only_baseline.metadata.json"
MANIFEST = OUTPUT_FOLDER / "04_property_only_baseline.validation.json"
MODEL_FILE = MODEL_FOLDER / "04_property_only_linear_baseline.joblib"

CATEGORICAL_FEATURES = ["property_type", "duration", "old_new"]
NUMERIC_FEATURES = ["month_index", "month_sin", "month_cos"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

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
    return engineered


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
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
    rows: list[dict[str, object]] = []
    column_map = {
        "Training-median benchmark": (
            "naive_predicted_price",
            "naive_predicted_log_price",
        ),
        "Property-only linear regression": (
            "predicted_price",
            "predicted_log_price",
        ),
        "Property-only linear regression with smearing": (
            "smearing_predicted_price",
            "predicted_log_price",
        ),
    }
    prediction_column, log_prediction_column = column_map[model_name]
    for group_name, group in data.groupby(group_column, observed=True):
        row = metric_row(
            model_name,
            str(group_name),
            group["price"].to_numpy(),
            group[prediction_column].to_numpy(),
            group["log_price"].to_numpy(),
            group[log_prediction_column].to_numpy(),
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

    assert len(source) == 21_968
    assert len(cohort) == 18_670
    assert len(development) == 12_785
    assert len(test) == 5_885
    assert development["date_of_transfer"].max() < test["date_of_transfer"].min()
    assert development[MODEL_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert test[MODEL_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert set(development["transaction_id"]).isdisjoint(set(test["transaction_id"]))

    split_rows: list[dict[str, object]] = []
    for split_name, data in [
        ("Full Category A cohort", cohort),
        ("Development: 2022-2023", development),
        ("Test: 2024", test),
    ]:
        split_rows.append(
            {
                "split": split_name,
                "rows": len(data),
                "date_min": data["date_of_transfer"].min().date().isoformat(),
                "date_max": data["date_of_transfer"].max().date().isoformat(),
                "median_price": data["price"].median(),
                "mean_price": data["price"].mean(),
                "p01_price": data["price"].quantile(0.01),
                "p99_price": data["price"].quantile(0.99),
                "maximum_price": data["price"].max(),
                "detached_rows": int(data["property_type"].eq("D").sum()),
                "flat_rows": int(data["property_type"].eq("F").sum()),
                "semi_detached_rows": int(data["property_type"].eq("S").sum()),
                "terraced_rows": int(data["property_type"].eq("T").sum()),
                "freehold_rows": int(data["duration"].eq("F").sum()),
                "leasehold_rows": int(data["duration"].eq("L").sum()),
                "new_build_rows": int(data["old_new"].eq("Y").sum()),
            }
        )
    split_summary = pd.DataFrame(split_rows)
    split_summary.to_csv(SPLIT_SUMMARY, index=False, float_format="%.2f")

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
        validation_direct_price_prediction = np.exp(validation_log_prediction)
        validation_smearing_price_prediction = (
            validation_direct_price_prediction * fold_smearing
        )

        naive_price = float(fold_train["price"].median())
        naive_log = float(np.log(naive_price))
        naive_price_predictions = np.full(len(fold_validation), naive_price)
        naive_log_predictions = np.full(len(fold_validation), naive_log)

        for row in [
            metric_row(
                "Training-median benchmark",
                fold_name,
                fold_validation["price"].to_numpy(),
                naive_price_predictions,
                fold_validation["log_price"].to_numpy(),
                naive_log_predictions,
            ),
            metric_row(
                "Property-only linear regression",
                fold_name,
                fold_validation["price"].to_numpy(),
                validation_direct_price_prediction,
                fold_validation["log_price"].to_numpy(),
                validation_log_prediction,
            ),
            metric_row(
                "Property-only linear regression with smearing",
                fold_name,
                fold_validation["price"].to_numpy(),
                validation_smearing_price_prediction,
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
                    if row["model"] == "Property-only linear regression with smearing"
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

    naive_price = float(development["price"].median())
    naive_log = float(np.log(naive_price))
    naive_test_price = np.full(len(test), naive_price)
    naive_test_log = np.full(len(test), naive_log)

    test_metric_rows = [
        metric_row(
            "Training-median benchmark",
            "2024 holdout",
            test["price"].to_numpy(),
            naive_test_price,
            test["log_price"].to_numpy(),
            naive_test_log,
        ),
        metric_row(
            "Property-only linear regression",
            "2024 holdout",
            test["price"].to_numpy(),
            test_price_prediction,
            test["log_price"].to_numpy(),
            test_log_prediction,
        ),
        metric_row(
            "Property-only linear regression with smearing",
            "2024 holdout",
            test["price"].to_numpy(),
            test_smearing_price_prediction,
            test["log_price"].to_numpy(),
            test_log_prediction,
        ),
    ]
    test_metrics = pd.DataFrame(test_metric_rows)
    test_metrics.to_csv(TEST_METRICS, index=False, float_format="%.6f")

    preprocessor = final_model.named_steps["preprocessor"]
    regressor = final_model.named_steps["regressor"]
    feature_names = preprocessor.get_feature_names_out()
    coefficient_table = pd.DataFrame(
        {
            "feature": feature_names,
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
        ]
    ].copy()
    predictions["log_price"] = test["log_price"].to_numpy()
    predictions["predicted_log_price"] = test_log_prediction
    predictions["predicted_price"] = test_price_prediction
    predictions["smearing_predicted_price"] = test_smearing_price_prediction
    predictions["naive_predicted_log_price"] = naive_test_log
    predictions["naive_predicted_price"] = naive_test_price
    predictions["prediction_error"] = predictions["predicted_price"] - predictions["price"]
    predictions["absolute_error"] = predictions["prediction_error"].abs()
    predictions["absolute_percentage_error_percent"] = (
        predictions["absolute_error"] / predictions["price"] * 100
    )
    predictions.to_csv(TEST_PREDICTIONS, index=False, float_format="%.6f")

    error_by_type = pd.concat(
        [
            grouped_price_metrics(
                predictions, "property_type_label", "Training-median benchmark"
            ),
            grouped_price_metrics(
                predictions, "property_type_label", "Property-only linear regression"
            ),
            grouped_price_metrics(
                predictions,
                "property_type_label",
                "Property-only linear regression with smearing",
            ),
        ],
        ignore_index=True,
    )
    error_by_type.to_csv(ERROR_BY_TYPE, index=False, float_format="%.6f")

    monthly_rows: list[dict[str, object]] = []
    for sale_month, group in predictions.groupby("sale_month", observed=True):
        metrics = metric_row(
            "Property-only linear regression",
            sale_month,
            group["price"].to_numpy(),
            group["predicted_price"].to_numpy(),
            group["log_price"].to_numpy(),
            group["predicted_log_price"].to_numpy(),
        )
        metrics.update(
            {
                "sale_month": sale_month,
                "actual_median_price": group["price"].median(),
                "predicted_median_price": group["predicted_price"].median(),
                "actual_mean_price": group["price"].mean(),
                "predicted_mean_price": group["predicted_price"].mean(),
            }
        )
        monthly_rows.append(metrics)
    monthly_performance = pd.DataFrame(monthly_rows).sort_values("sale_month")
    monthly_performance.to_csv(MONTHLY_PERFORMANCE, index=False, float_format="%.6f")

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
    axis.plot([chart_min, chart_max], [chart_min, chart_max], color=NAVY, linestyle="--", linewidth=1.6)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel("Actual 2024 sale price")
    axis.set_ylabel("Predicted sale price")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axis.grid()
    axis.legend(frameon=False, ncol=2, loc="upper left")
    linear_test = test_metrics.loc[
        test_metrics["model"].eq("Property-only linear regression")
    ].iloc[0]
    smeared_test = test_metrics.loc[
        test_metrics["model"].eq("Property-only linear regression with smearing")
    ].iloc[0]
    axis.text(
        0.97,
        0.05,
        f"MAE {money(linear_test['mae_gbp'])}\nRMSE {money(linear_test['rmse_gbp'])}\nR² {linear_test['r2_price']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=NAVY,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": LIGHT_GREY},
    )
    axis.set_title("Property-only linear baseline: actual versus predicted", pad=14)
    fig.suptitle(
        "2024 holdout predictions",
        fontsize=12,
        color=GREY,
        y=0.96,
    )
    fig.tight_layout()
    fig.savefig(ACTUAL_PREDICTED_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7))
    comparison = test_metrics.set_index("model")
    model_labels = ["Training\nmedian", "Linear\ndirect exp", "Linear\nsmearing"]
    model_keys = [
        "Training-median benchmark",
        "Property-only linear regression",
        "Property-only linear regression with smearing",
    ]
    x = np.arange(len(model_labels))
    width = 0.32
    mae_values = comparison.loc[model_keys, "mae_gbp"].to_numpy()
    rmse_values = comparison.loc[model_keys, "rmse_gbp"].to_numpy()
    bars_mae = axes[0].bar(x - width / 2, mae_values, width, label="MAE", color=BLUE)
    bars_rmse = axes[0].bar(x + width / 2, rmse_values, width, label="RMSE", color=ORANGE)
    axes[0].set_xticks(x, model_labels)
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

    monthly_chart = monthly_performance.copy()
    monthly_chart["month_date"] = pd.to_datetime(monthly_chart["sale_month"] + "-01")
    axes[1].plot(
        monthly_chart["month_date"],
        monthly_chart["actual_median_price"],
        color=NAVY,
        linewidth=2.5,
        marker="o",
        label="Actual median",
    )
    axes[1].plot(
        monthly_chart["month_date"],
        monthly_chart["predicted_median_price"],
        color=TEAL,
        linewidth=2.5,
        marker="o",
        label="Predicted median",
    )
    axes[1].set_title("Monthly median calibration")
    axes[1].set_xlabel("2024 sale month")
    axes[1].set_ylabel("Median price")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False)

    fig.suptitle(
        "Property-only baseline performance",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(PERFORMANCE_CHART, dpi=220, bbox_inches="tight")
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

    cv_summary = (
        cv_metrics.groupby("model", as_index=False)
        .agg(
            mean_mae_gbp=("mae_gbp", "mean"),
            mean_rmse_gbp=("rmse_gbp", "mean"),
            mean_r2_price=("r2_price", "mean"),
            mean_r2_log_price=("r2_log_price", "mean"),
        )
        .set_index("model")
    )
    naive_test = comparison.loc["Training-median benchmark"]
    mae_improvement = (naive_test["mae_gbp"] - linear_test["mae_gbp"]) / naive_test["mae_gbp"] * 100
    rmse_improvement = (
        (naive_test["rmse_gbp"] - smeared_test["rmse_gbp"])
        / naive_test["rmse_gbp"]
        * 100
    )

    coefficient_lookup = coefficient_table.set_index("feature")
    findings = f"""# Property-only linear baseline

## Purpose

This checkpoint establishes the baseline prediction performance before crime and claimant variables are introduced. It uses only basic property and time information:

- Property type: detached, semi-detached, terraced or flat/maisonette.
- Tenure: freehold or leasehold.
- New-build status: new build or established property.
- A continuous month index and cyclical month-of-year terms.

The model predicts `log(price)`. Two back-transformations are evaluated: direct exponentiation, which estimates a conditional median, and a training-derived smearing correction, which shifts predictions towards an expected pound value. No postcode, LSOA, crime or claimant feature is included.

## Chronological split

| Dataset | Period | Transactions | Median price |
|---|---|---:|---:|
| Development | 2022–2023 | {len(development):,} | {money(development['price'].median())} |
| Test | 2024 | {len(test):,} | {money(test['price'].median())} |

There is no transaction overlap, and every development transaction occurs before every test transaction. Model fitting and the optional smearing correction use development data only.

## Time-based validation

Three expanding-window validation folds were used within 2022–2023. Every fold begins with at least 12 months of training data so that all seasons are represented. The average results were:

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Training-median benchmark | {money(cv_summary.loc['Training-median benchmark', 'mean_mae_gbp'])} | {money(cv_summary.loc['Training-median benchmark', 'mean_rmse_gbp'])} | {cv_summary.loc['Training-median benchmark', 'mean_r2_price']:.3f} |
| Property-only linear regression | {money(cv_summary.loc['Property-only linear regression', 'mean_mae_gbp'])} | {money(cv_summary.loc['Property-only linear regression', 'mean_rmse_gbp'])} | {cv_summary.loc['Property-only linear regression', 'mean_r2_price']:.3f} |
| Property-only linear regression with smearing | {money(cv_summary.loc['Property-only linear regression with smearing', 'mean_mae_gbp'])} | {money(cv_summary.loc['Property-only linear regression with smearing', 'mean_rmse_gbp'])} | {cv_summary.loc['Property-only linear regression with smearing', 'mean_r2_price']:.3f} |

## Final 2024 holdout result

| Model | MAE | RMSE | Price R² | Log-price R² | Median absolute percentage error |
|---|---:|---:|---:|---:|---:|
| Training-median benchmark | {money(naive_test['mae_gbp'])} | {money(naive_test['rmse_gbp'])} | {naive_test['r2_price']:.3f} | {naive_test['r2_log_price']:.3f} | {naive_test['median_absolute_percentage_error_percent']:.1f}% |
| Property-only linear regression | {money(linear_test['mae_gbp'])} | {money(linear_test['rmse_gbp'])} | {linear_test['r2_price']:.3f} | {linear_test['r2_log_price']:.3f} | {linear_test['median_absolute_percentage_error_percent']:.1f}% |
| Property-only linear regression with smearing | {money(smeared_test['mae_gbp'])} | {money(smeared_test['rmse_gbp'])} | {smeared_test['r2_price']:.3f} | {smeared_test['r2_log_price']:.3f} | {smeared_test['median_absolute_percentage_error_percent']:.1f}% |

Relative to predicting the development-set median for every property, direct exponentiation reduces MAE by **{mae_improvement:.1f}%**. Smearing produces the lower RMSE, reducing it by **{rmse_improvement:.1f}%**. Because MAE is less dominated by unusually expensive properties, direct exponentiation is retained as the primary baseline prediction; the smeared result remains a reported sensitivity measure.

## Coefficient interpretation

The categorical reference property is detached, freehold and established. Holding the other included features constant:

- Flat/maisonette coefficient: **{coefficient_lookup.loc['property_type_F', 'coefficient_log_price']:.3f}**, approximately **{coefficient_lookup.loc['property_type_F', 'approximate_percent_change']:.1f}%** relative to detached.
- Semi-detached coefficient: **{coefficient_lookup.loc['property_type_S', 'coefficient_log_price']:.3f}**, approximately **{coefficient_lookup.loc['property_type_S', 'approximate_percent_change']:.1f}%** relative to detached.
- Terraced coefficient: **{coefficient_lookup.loc['property_type_T', 'coefficient_log_price']:.3f}**, approximately **{coefficient_lookup.loc['property_type_T', 'approximate_percent_change']:.1f}%** relative to detached.
- Leasehold coefficient: **{coefficient_lookup.loc['duration_L', 'coefficient_log_price']:.3f}**, approximately **{coefficient_lookup.loc['duration_L', 'approximate_percent_change']:.1f}%** relative to freehold.
- New-build coefficient: **{coefficient_lookup.loc['old_new_Y', 'coefficient_log_price']:.3f}**, approximately **{coefficient_lookup.loc['old_new_Y', 'approximate_percent_change']:.1f}%** relative to established property.

These coefficients describe conditional associations within this additive model. They should not be interpreted causally, especially because property type and tenure are related and the model does not yet include floor area, bedrooms, exact location or other important housing characteristics.

## Limitations

- The Land Registry data does not provide bedrooms, floor area, condition or detailed amenities.
- An additive linear model cannot automatically learn interactions or nonlinear effects.
- Predictions for unusually expensive properties remain difficult, so RMSE is substantially influenced by the upper price tail.
- Direct exponentiation and smearing optimise different aspects of performance: the former gives lower MAE, while the latter gives lower RMSE.
- This baseline contains no explicit neighbourhood location or neighbourhood-condition feature.

## Next modelling experiment

Fit the same linear model on the same 12,785 development transactions and evaluate the same 5,885 test transactions after adding:

- `log1p(total_crime)`
- `claimant_rate_percent`

The difference in 2024 MAE, RMSE and R² will provide the first direct answer to whether the neighbourhood variables improve predictions beyond property and time characteristics.
"""
    FINDINGS.write_text(findings, encoding="utf-8")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "Property-only linear regression",
        "estimator": "sklearn.linear_model.LinearRegression",
        "sklearn_version": sklearn.__version__,
        "cohort": "PPD Category A",
        "target": "log(price)",
        "primary_back_transformation": "exp(predicted_log_price); selected using validation MAE",
        "sensitivity_back_transformation": "exp(predicted_log_price) multiplied by development-set smearing factor",
        "smearing_factor": final_smearing,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "category_references": {
            "property_type": "D (Detached)",
            "duration": "F (Freehold)",
            "old_new": "N (Established)",
        },
        "development_period": "2022-01-04 to 2023-12-22",
        "test_period": "2024-01-02 to 2024-12-31",
        "development_rows": len(development),
        "test_rows": len(test),
        "randomness": "none; deterministic estimator and chronological split",
        "neighbourhood_features_included": False,
        "model_file": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
        "model_sha256": sha256(MODEL_FILE),
    }
    MODEL_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    output_files = [
        SPLIT_SUMMARY,
        CV_METRICS,
        TEST_METRICS,
        COEFFICIENTS,
        TEST_PREDICTIONS,
        ERROR_BY_TYPE,
        MONTHLY_PERFORMANCE,
        ACTUAL_PREDICTED_CHART,
        PERFORMANCE_CHART,
        FINDINGS,
        MODEL_METADATA,
        MODEL_FILE,
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(DATASET.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(DATASET),
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
            "cross_validation_folds": len(fold_definitions),
        },
        "test_metrics": {
            "mae_gbp": float(linear_test["mae_gbp"]),
            "rmse_gbp": float(linear_test["rmse_gbp"]),
            "r2_price": float(linear_test["r2_price"]),
            "r2_log_price": float(linear_test["r2_log_price"]),
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
    print(f"2024 MAE: {money(linear_test['mae_gbp'])}")
    print(f"2024 RMSE (direct exp): {money(linear_test['rmse_gbp'])}")
    print(f"2024 RMSE (smearing): {money(smeared_test['rmse_gbp'])}")
    print(f"2024 price R2: {linear_test['r2_price']:.4f}")
    print(f"2024 log-price R2: {linear_test['r2_log_price']:.4f}")
    print(f"Saved outputs to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
