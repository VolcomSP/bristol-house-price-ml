"""Model checkpoint 3: tuned property-only random forest with a 2024 holdout."""

from __future__ import annotations

import hashlib
import itertools
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
from sklearn.ensemble import RandomForestRegressor
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
LINEAR_TEST = PROJECT_ROOT / "outputs" / "models" / "04_baseline_test_metrics.csv"
LINEAR_CV = PROJECT_ROOT / "outputs" / "models" / "04_baseline_cross_validation_metrics.csv"
LINEAR_PREDICTIONS = PROJECT_ROOT / "outputs" / "models" / "04_baseline_2024_predictions.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "models"
MODEL_FOLDER = PROJECT_ROOT / "models"

SEARCH_RESULTS = OUTPUT_FOLDER / "06_property_rf_hyperparameter_search.csv"
CV_METRICS = OUTPUT_FOLDER / "06_property_rf_cross_validation_metrics.csv"
TEST_METRICS = OUTPUT_FOLDER / "06_property_rf_test_metrics.csv"
FEATURE_IMPORTANCE = OUTPUT_FOLDER / "06_property_rf_feature_importance.csv"
TEST_PREDICTIONS = OUTPUT_FOLDER / "06_property_rf_2024_predictions.csv"
ERROR_BY_TYPE = OUTPUT_FOLDER / "06_property_rf_error_by_property_type.csv"
MONTHLY_PERFORMANCE = OUTPUT_FOLDER / "06_property_rf_monthly_performance.csv"
MODEL_COMPARISON = OUTPUT_FOLDER / "06_property_rf_vs_linear_comparison.csv"
ACTUAL_PREDICTED_CHART = OUTPUT_FOLDER / "06_property_rf_actual_vs_predicted.jpg"
PERFORMANCE_CHART = OUTPUT_FOLDER / "06_property_rf_performance.png"
FINDINGS = OUTPUT_FOLDER / "06_property_rf_findings.md"
MODEL_METADATA = OUTPUT_FOLDER / "06_property_rf.metadata.json"
MANIFEST = OUTPUT_FOLDER / "06_property_rf.validation.json"
MODEL_FILE = MODEL_FOLDER / "06_property_only_random_forest.joblib"

CATEGORICAL_FEATURES = ["property_type", "duration", "old_new"]
NUMERIC_FEATURES = ["month_index", "month_sin", "month_cos"]
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

NAIVE_NAME = "Training-median benchmark"
LINEAR_NAME = "Property-only linear regression"
LINEAR_SMEAR_NAME = "Property-only linear regression with smearing"
RF_NAME = "Property-only random forest"
RF_SMEAR_NAME = "Property-only random forest with smearing"

RANDOM_STATE = 42
N_ESTIMATORS = 240
PARAMETER_GRID = {
    "max_depth": [8, 12, None],
    "min_samples_leaf": [5, 20],
    "max_features": [0.7, 1.0],
}
FOLD_DEFINITIONS = [
    ("Fold 1", "2022-12", "2023-01", "2023-04"),
    ("Fold 2", "2023-04", "2023-05", "2023-08"),
    ("Fold 3", "2023-08", "2023-09", "2023-12"),
]

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


def build_pipeline(parameters: dict[str, object]) -> Pipeline:
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
    forest = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        criterion="squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        max_depth=parameters["max_depth"],
        min_samples_leaf=parameters["min_samples_leaf"],
        max_features=parameters["max_features"],
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", forest)])


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


def parameter_combinations() -> list[dict[str, object]]:
    return [
        {
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
        }
        for max_depth, min_samples_leaf, max_features in itertools.product(
            PARAMETER_GRID["max_depth"],
            PARAMETER_GRID["min_samples_leaf"],
            PARAMETER_GRID["max_features"],
        )
    ]


def grouped_price_metrics(
    data: pd.DataFrame, group_column: str, model_name: str
) -> pd.DataFrame:
    prediction_column = {
        RF_NAME: "predicted_price",
        RF_SMEAR_NAME: "smearing_predicted_price",
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
    linear_cv = pd.read_csv(LINEAR_CV)
    linear_test = pd.read_csv(LINEAR_TEST)
    linear_predictions = pd.read_csv(LINEAR_PREDICTIONS)

    assert len(source) == 21_968
    assert len(cohort) == 18_670
    assert len(development) == 12_785
    assert len(test) == 5_885
    assert development["date_of_transfer"].max() < test["date_of_transfer"].min()
    assert set(development["transaction_id"]).isdisjoint(set(test["transaction_id"]))
    assert development[MODEL_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert test[MODEL_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert linear_predictions["transaction_id"].tolist() == test["transaction_id"].tolist()

    fold_data: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for fold_name, train_end, validation_start, validation_end in FOLD_DEFINITIONS:
        fold_train = development.loc[development["sale_month"].le(train_end)].copy()
        fold_validation = development.loc[
            development["sale_month"].between(validation_start, validation_end)
        ].copy()
        assert fold_train["date_of_transfer"].max() < fold_validation["date_of_transfer"].min()
        fold_data.append((fold_name, fold_train, fold_validation))

    search_rows: list[dict[str, object]] = []
    combinations = parameter_combinations()
    for candidate_id, parameters in enumerate(combinations, start=1):
        fold_metrics: list[dict[str, object]] = []
        for fold_name, fold_train, fold_validation in fold_data:
            model = build_pipeline(parameters)
            model.fit(fold_train[MODEL_FEATURES], fold_train["log_price"])
            predicted_log = model.predict(fold_validation[MODEL_FEATURES])
            predicted_price = np.exp(predicted_log)
            fold_metrics.append(
                metric_row(
                    RF_NAME,
                    fold_name,
                    fold_validation["price"].to_numpy(),
                    predicted_price,
                    fold_validation["log_price"].to_numpy(),
                    predicted_log,
                )
            )
        fold_table = pd.DataFrame(fold_metrics)
        search_rows.append(
            {
                "candidate_id": candidate_id,
                "n_estimators": N_ESTIMATORS,
                "max_depth": "None" if parameters["max_depth"] is None else parameters["max_depth"],
                "min_samples_leaf": parameters["min_samples_leaf"],
                "max_features": parameters["max_features"],
                "mean_validation_mae_gbp": fold_table["mae_gbp"].mean(),
                "mean_validation_rmse_gbp": fold_table["rmse_gbp"].mean(),
                "mean_validation_r2_price": fold_table["r2_price"].mean(),
                "mean_validation_r2_log_price": fold_table["r2_log_price"].mean(),
                "fold_1_mae_gbp": fold_table.loc[
                    fold_table["evaluation_period"].eq("Fold 1"), "mae_gbp"
                ].iloc[0],
                "fold_2_mae_gbp": fold_table.loc[
                    fold_table["evaluation_period"].eq("Fold 2"), "mae_gbp"
                ].iloc[0],
                "fold_3_mae_gbp": fold_table.loc[
                    fold_table["evaluation_period"].eq("Fold 3"), "mae_gbp"
                ].iloc[0],
            }
        )
    search_results = pd.DataFrame(search_rows).sort_values(
        ["mean_validation_mae_gbp", "mean_validation_rmse_gbp", "candidate_id"]
    ).reset_index(drop=True)
    search_results.insert(0, "validation_rank", np.arange(1, len(search_results) + 1))
    search_results.to_csv(SEARCH_RESULTS, index=False, float_format="%.6f")

    winner = search_results.iloc[0]
    selected_parameters = {
        "max_depth": None if pd.isna(winner["max_depth"]) else int(float(winner["max_depth"])),
        "min_samples_leaf": int(winner["min_samples_leaf"]),
        "max_features": float(winner["max_features"]),
    }

    cv_rows: list[dict[str, object]] = []
    for fold_name, fold_train, fold_validation in fold_data:
        model = build_pipeline(selected_parameters)
        model.fit(fold_train[MODEL_FEATURES], fold_train["log_price"])
        training_log_prediction = model.predict(fold_train[MODEL_FEATURES])
        fold_smearing = smearing_factor(
            fold_train["log_price"].to_numpy(), training_log_prediction
        )
        validation_log_prediction = model.predict(fold_validation[MODEL_FEATURES])
        direct_prediction = np.exp(validation_log_prediction)
        smearing_prediction = direct_prediction * fold_smearing
        for row in [
            metric_row(
                RF_NAME,
                fold_name,
                fold_validation["price"].to_numpy(),
                direct_prediction,
                fold_validation["log_price"].to_numpy(),
                validation_log_prediction,
            ),
            metric_row(
                RF_SMEAR_NAME,
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
                    if row["model"] == RF_SMEAR_NAME
                    else np.nan,
                }
            )
            cv_rows.append(row)
    cv_metrics = pd.DataFrame(cv_rows)
    cv_metrics.to_csv(CV_METRICS, index=False, float_format="%.6f")

    final_model = build_pipeline(selected_parameters)
    final_model.fit(development[MODEL_FEATURES], development["log_price"])
    development_log_prediction = final_model.predict(development[MODEL_FEATURES])
    final_smearing = smearing_factor(
        development["log_price"].to_numpy(), development_log_prediction
    )
    test_log_prediction = final_model.predict(test[MODEL_FEATURES])
    test_price_prediction = np.exp(test_log_prediction)
    test_smearing_prediction = test_price_prediction * final_smearing

    test_metrics = pd.DataFrame(
        [
            metric_row(
                RF_NAME,
                "2024 holdout",
                test["price"].to_numpy(),
                test_price_prediction,
                test["log_price"].to_numpy(),
                test_log_prediction,
            ),
            metric_row(
                RF_SMEAR_NAME,
                "2024 holdout",
                test["price"].to_numpy(),
                test_smearing_prediction,
                test["log_price"].to_numpy(),
                test_log_prediction,
            ),
        ]
    )
    test_metrics.to_csv(TEST_METRICS, index=False, float_format="%.6f")

    preprocessor = final_model.named_steps["preprocessor"]
    forest = final_model.named_steps["regressor"]
    feature_importance = pd.DataFrame(
        {
            "feature": preprocessor.get_feature_names_out(),
            "importance": forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False, ignore_index=True)
    feature_importance.insert(0, "rank", np.arange(1, len(feature_importance) + 1))
    feature_importance["importance_percent"] = feature_importance["importance"] * 100
    feature_importance["interpretation"] = feature_importance["feature"].map(
        {
            "property_type_F": "Flat/Maisonette indicator",
            "property_type_S": "Semi-detached indicator",
            "property_type_T": "Terraced indicator",
            "duration_L": "Leasehold indicator",
            "old_new_Y": "New-build indicator",
            "month_index": "Continuous month from January 2022",
            "month_sin": "Seasonal sine component",
            "month_cos": "Seasonal cosine component",
        }
    )
    feature_importance.to_csv(FEATURE_IMPORTANCE, index=False, float_format="%.8f")

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
    predictions["smearing_predicted_price"] = test_smearing_prediction
    predictions["linear_predicted_log_price"] = linear_predictions[
        "predicted_log_price"
    ].to_numpy()
    predictions["linear_predicted_price"] = linear_predictions["predicted_price"].to_numpy()
    predictions["prediction_error"] = predictions["predicted_price"] - predictions["price"]
    predictions["absolute_error"] = predictions["prediction_error"].abs()
    predictions["linear_absolute_error"] = (
        predictions["linear_predicted_price"] - predictions["price"]
    ).abs()
    predictions["absolute_error_improvement_vs_linear"] = (
        predictions["linear_absolute_error"] - predictions["absolute_error"]
    )
    predictions["absolute_percentage_error_percent"] = (
        predictions["absolute_error"] / predictions["price"] * 100
    )
    predictions.to_csv(TEST_PREDICTIONS, index=False, float_format="%.6f")

    error_by_type = pd.concat(
        [
            grouped_price_metrics(predictions, "property_type_label", RF_NAME),
            grouped_price_metrics(predictions, "property_type_label", RF_SMEAR_NAME),
        ],
        ignore_index=True,
    )
    error_by_type.to_csv(ERROR_BY_TYPE, index=False, float_format="%.6f")

    monthly_rows: list[dict[str, object]] = []
    for sale_month, group in predictions.groupby("sale_month", observed=True):
        row = metric_row(
            RF_NAME,
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
                "linear_predicted_median_price": group["linear_predicted_price"].median(),
                "random_forest_predicted_median_price": group["predicted_price"].median(),
                "actual_mean_price": group["price"].mean(),
                "linear_predicted_mean_price": group["linear_predicted_price"].mean(),
                "random_forest_predicted_mean_price": group["predicted_price"].mean(),
            }
        )
        monthly_rows.append(row)
    monthly_performance = pd.DataFrame(monthly_rows).sort_values("sale_month")
    monthly_performance.to_csv(MONTHLY_PERFORMANCE, index=False, float_format="%.6f")

    comparison = pd.concat(
        [
            linear_test.loc[
                linear_test["model"].isin([NAIVE_NAME, LINEAR_NAME, LINEAR_SMEAR_NAME])
            ].copy(),
            test_metrics.copy(),
        ],
        ignore_index=True,
    )
    linear_direct = comparison.loc[comparison["model"].eq(LINEAR_NAME)].iloc[0]
    comparison["mae_change_vs_linear_gbp"] = comparison["mae_gbp"] - linear_direct["mae_gbp"]
    comparison["mae_improvement_vs_linear_percent"] = (
        (linear_direct["mae_gbp"] - comparison["mae_gbp"]) / linear_direct["mae_gbp"] * 100
    )
    comparison["rmse_change_vs_linear_gbp"] = (
        comparison["rmse_gbp"] - linear_direct["rmse_gbp"]
    )
    comparison["rmse_improvement_vs_linear_percent"] = (
        (linear_direct["rmse_gbp"] - comparison["rmse_gbp"])
        / linear_direct["rmse_gbp"]
        * 100
    )
    comparison["r2_price_change_vs_linear"] = comparison["r2_price"] - linear_direct["r2_price"]
    comparison.to_csv(MODEL_COMPARISON, index=False, float_format="%.6f")

    rf_direct = comparison.loc[comparison["model"].eq(RF_NAME)].iloc[0]
    rf_smear = comparison.loc[comparison["model"].eq(RF_SMEAR_NAME)].iloc[0]

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
        f"MAE {money(rf_direct['mae_gbp'])}\n"
        f"RMSE {money(rf_direct['rmse_gbp'])}\n"
        f"R² {rf_direct['r2_price']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=NAVY,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": LIGHT_GREY},
    )
    axis.set_title("Property-only random forest: actual versus predicted", pad=14)
    fig.suptitle("2024 holdout predictions", fontsize=12, color=GREY, y=0.96)
    fig.tight_layout()
    fig.savefig(
        ACTUAL_PREDICTED_CHART,
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
        format="jpg",
    )
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7))
    chart_keys = [LINEAR_NAME, RF_NAME, RF_SMEAR_NAME]
    chart_labels = ["Linear\ndirect exp", "Random forest\ndirect exp", "Random forest\nsmearing"]
    chart_data = comparison.set_index("model").loc[chart_keys]
    x = np.arange(len(chart_labels))
    width = 0.32
    mae_values = chart_data["mae_gbp"].to_numpy()
    rmse_values = chart_data["rmse_gbp"].to_numpy()
    bars_mae = axes[0].bar(x - width / 2, mae_values, width, label="MAE", color=BLUE)
    bars_rmse = axes[0].bar(x + width / 2, rmse_values, width, label="RMSE", color=ORANGE)
    axes[0].set_xticks(x, chart_labels)
    axes[0].set_ylabel("Error (£)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[0].set_title("2024 property-only error comparison")
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
    lower = min(0.0, float(r2_values.min()) * 1.15)
    upper = max(0.05, float(r2_values.max()) * 1.28)
    axes[1].set_ylim(lower, upper)
    for bar in r2_bars:
        offset = max(abs(upper - lower) * 0.025, 0.005)
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + offset,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            color=NAVY,
        )
    fig.suptitle(
        "Does a nonlinear model improve the property-only baseline?",
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
            "selected_parameters": selected_parameters,
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "training_period": "2022-01 to 2023-12",
        },
        MODEL_FILE,
    )

    cv_summary = cv_metrics.groupby("model", as_index=False).agg(
        mean_mae_gbp=("mae_gbp", "mean"),
        mean_rmse_gbp=("rmse_gbp", "mean"),
        mean_r2_price=("r2_price", "mean"),
        mean_r2_log_price=("r2_log_price", "mean"),
    ).set_index("model")
    linear_cv_direct = linear_cv.loc[linear_cv["model"].eq(LINEAR_NAME)]
    linear_cv_summary = pd.Series(
        {
            "mean_mae_gbp": linear_cv_direct["mae_gbp"].mean(),
            "mean_rmse_gbp": linear_cv_direct["rmse_gbp"].mean(),
            "mean_r2_price": linear_cv_direct["r2_price"].mean(),
            "mean_r2_log_price": linear_cv_direct["r2_log_price"].mean(),
        }
    )
    validation_mae_improvement = (
        linear_cv_summary["mean_mae_gbp"] - cv_summary.loc[RF_NAME, "mean_mae_gbp"]
    ) / linear_cv_summary["mean_mae_gbp"] * 100
    top_features = feature_importance.head(3)
    top_feature_text = ", ".join(
        f"`{row.feature}` ({row.importance_percent:.1f}%)"
        for row in top_features.itertuples(index=False)
    )

    findings = f"""# Property-only random forest

## Purpose

This checkpoint tests whether a nonlinear random forest improves house-price predictions when it receives exactly the same property and time features as the property-only linear model:

- Property type.
- Tenure.
- New-build status.
- Continuous month index and cyclical month-of-year terms.

No postcode, LSOA identifier, crime or claimant feature is included. The model predicts `log(price)` and uses the same Category A cohort, 2022–2023 development data, expanding-window validation and unseen 2024 holdout.

## Hyperparameter selection

Twelve candidate settings were evaluated across all three development-period folds. The selection criterion was mean validation MAE using direct exponentiation. The selected settings were:

- Trees: **{N_ESTIMATORS}**.
- Maximum depth: **{selected_parameters['max_depth']}**.
- Minimum observations per leaf: **{selected_parameters['min_samples_leaf']}**.
- Candidate feature proportion per split: **{selected_parameters['max_features']}**.
- Random seed: **{RANDOM_STATE}**.

The 2024 test data was not used during parameter selection.

## Time-based validation

| Prediction method | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only linear regression | {money(linear_cv_summary['mean_mae_gbp'])} | {money(linear_cv_summary['mean_rmse_gbp'])} | {linear_cv_summary['mean_r2_price']:.3f} |
| Random forest, direct exponentiation | {money(cv_summary.loc[RF_NAME, 'mean_mae_gbp'])} | {money(cv_summary.loc[RF_NAME, 'mean_rmse_gbp'])} | {cv_summary.loc[RF_NAME, 'mean_r2_price']:.3f} |
| Random forest, smearing | {money(cv_summary.loc[RF_SMEAR_NAME, 'mean_mae_gbp'])} | {money(cv_summary.loc[RF_SMEAR_NAME, 'mean_rmse_gbp'])} | {cv_summary.loc[RF_SMEAR_NAME, 'mean_r2_price']:.3f} |

The direct random forest changes mean validation MAE by **{validation_mae_improvement:+.1f}%** relative to the linear model. It therefore does not provide a meaningful validation improvement with this feature set.

## Final 2024 comparison

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only linear regression | {money(linear_direct['mae_gbp'])} | {money(linear_direct['rmse_gbp'])} | {linear_direct['r2_price']:.3f} | {linear_direct['r2_log_price']:.3f} |
| Property-only random forest | {money(rf_direct['mae_gbp'])} | {money(rf_direct['rmse_gbp'])} | {rf_direct['r2_price']:.3f} | {rf_direct['r2_log_price']:.3f} |
| Property-only random forest with smearing | {money(rf_smear['mae_gbp'])} | {money(rf_smear['rmse_gbp'])} | {rf_smear['r2_price']:.3f} | {rf_smear['r2_log_price']:.3f} |

Relative to the property-only linear model, the direct random forest changes MAE by **{rf_direct['mae_change_vs_linear_gbp']:+,.0f} pounds** ({rf_direct['mae_improvement_vs_linear_percent']:+.1f}%) and changes RMSE by **{rf_direct['rmse_change_vs_linear_gbp']:+,.0f} pounds** ({rf_direct['rmse_improvement_vs_linear_percent']:+.1f}%). Its price R² changes by **{rf_direct['r2_price_change_vs_linear']:+.3f}**. These differences are negligible, with the random forest marginally worse on the primary metrics.

## Feature importance

The three most important transformed inputs were {top_feature_text}. Random-forest impurity importance measures how often and how effectively a variable splits the training observations. It does not show causality and can favour continuous variables or features with more possible split points.

## Interpretation

This experiment isolates the effect of changing the algorithm. Because the feature set contains no detailed location or structural property information, a more flexible algorithm may still have limited information from which to learn. The result should therefore be assessed against the linear property-only model, not against the later enriched models.

## Limitations

- Land Registry data does not include bedrooms, floor area, property condition or detailed amenities.
- No postcode or neighbourhood identifier is included.
- Random forests cannot extrapolate beyond patterns observed in their training data.
- Impurity-based feature importance is descriptive and may be biased.
- Direct exponentiation and smearing optimise different error characteristics.

## Next modelling experiment

Fit an enriched random forest using the selected modelling procedure and add `log1p(total_crime)` and `claimant_rate_percent`. Comparing property-only and enriched random forests will test whether the neighbourhood variables provide predictive value under a nonlinear specification.
"""
    FINDINGS.write_text(findings, encoding="utf-8")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": RF_NAME,
        "estimator": "sklearn.ensemble.RandomForestRegressor",
        "sklearn_version": sklearn.__version__,
        "cohort": "PPD Category A",
        "target": "log(price)",
        "primary_back_transformation": "exp(predicted_log_price); parameter selection uses validation MAE",
        "sensitivity_back_transformation": "exp(predicted_log_price) multiplied by development-set smearing factor",
        "smearing_factor": final_smearing,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "neighbourhood_features_included": False,
        "parameter_grid": PARAMETER_GRID,
        "selected_parameters": selected_parameters,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "development_period": "2022-01-04 to 2023-12-22",
        "test_period": "2024-01-02 to 2024-12-31",
        "development_rows": len(development),
        "test_rows": len(test),
        "model_file": str(MODEL_FILE.relative_to(PROJECT_ROOT)),
        "model_sha256": sha256(MODEL_FILE),
    }
    MODEL_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    output_files = [
        SEARCH_RESULTS,
        CV_METRICS,
        TEST_METRICS,
        FEATURE_IMPORTANCE,
        TEST_PREDICTIONS,
        ERROR_BY_TYPE,
        MONTHLY_PERFORMANCE,
        MODEL_COMPARISON,
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
        "linear_cross_validation_metrics_sha256": sha256(LINEAR_CV),
        "linear_test_metrics_sha256": sha256(LINEAR_TEST),
        "linear_predictions_sha256": sha256(LINEAR_PREDICTIONS),
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
            "parameter_candidates": len(combinations),
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
                                "predicted_log_price",
                                "predicted_price",
                                "smearing_predicted_price",
                            ]
                        ]
                    )
                ).sum().sum()
            ),
            "linear_transaction_order_match": bool(
                linear_predictions["transaction_id"].tolist()
                == predictions["transaction_id"].tolist()
            ),
        },
        "test_metrics": {
            "mae_gbp": float(rf_direct["mae_gbp"]),
            "rmse_gbp": float(rf_direct["rmse_gbp"]),
            "r2_price": float(rf_direct["r2_price"]),
            "r2_log_price": float(rf_direct["r2_log_price"]),
            "mae_improvement_vs_linear_percent": float(
                rf_direct["mae_improvement_vs_linear_percent"]
            ),
            "rmse_improvement_vs_linear_percent": float(
                rf_direct["rmse_improvement_vs_linear_percent"]
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
    print(f"Candidates evaluated: {len(combinations)}")
    print(f"Selected parameters: {selected_parameters}")
    print(f"Smearing factor: {final_smearing:.6f}")
    print(f"Linear MAE: {money(linear_direct['mae_gbp'])}")
    print(f"Random forest MAE: {money(rf_direct['mae_gbp'])}")
    print(f"Random forest RMSE: {money(rf_direct['rmse_gbp'])}")
    print(f"Random forest price R2: {rf_direct['r2_price']:.4f}")
    print(f"Saved outputs to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
