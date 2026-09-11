"""Model checkpoint 5: property-only versus enriched gradient boosting.

Both feature sets use the same Category A cohort, log-price target,
expanding-window validation folds and 2024 holdout. Hyperparameters are tuned
only on 2022-2023 validation periods. The resulting comparison isolates the
predictive contribution of monthly LSOA crime and claimant-rate information.
"""

from __future__ import annotations

import hashlib
import importlib.util
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
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
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
MODEL_OUTPUTS = PROJECT_ROOT / "outputs" / "models"
MODEL_FOLDER = PROJECT_ROOT / "models"

PROPERTY_LINEAR_TEST = MODEL_OUTPUTS / "04_baseline_test_metrics.csv"
ENRICHED_LINEAR_TEST = MODEL_OUTPUTS / "05_enriched_test_metrics.csv"
PROPERTY_RF_TEST = MODEL_OUTPUTS / "06_property_rf_test_metrics.csv"
ENRICHED_RF_TEST = MODEL_OUTPUTS / "07_enriched_rf_test_metrics.csv"
ENRICHED_LINEAR_CV = MODEL_OUTPUTS / "05_enriched_cross_validation_metrics.csv"
ENRICHED_RF_CV = MODEL_OUTPUTS / "07_enriched_rf_cross_validation_metrics.csv"

SEARCH_RESULTS = MODEL_OUTPUTS / "08_gb_hyperparameter_search.csv"
CV_METRICS = MODEL_OUTPUTS / "08_gb_cross_validation_metrics.csv"
TEST_METRICS = MODEL_OUTPUTS / "08_gb_test_metrics.csv"
PERMUTATION_IMPORTANCE = MODEL_OUTPUTS / "08_gb_permutation_importance.csv"
TEST_PREDICTIONS = MODEL_OUTPUTS / "08_gb_2024_predictions.csv"
ERROR_BY_TYPE = MODEL_OUTPUTS / "08_gb_error_by_property_type.csv"
MONTHLY_PERFORMANCE = MODEL_OUTPUTS / "08_gb_monthly_performance.csv"
MODEL_COMPARISON = MODEL_OUTPUTS / "08_gb_all_model_comparison.csv"
ACTUAL_PREDICTED_CHART = MODEL_OUTPUTS / "08_enriched_gb_actual_vs_predicted.png"
PERFORMANCE_CHART = MODEL_OUTPUTS / "08_gb_model_performance.png"
IMPORTANCE_CHART = MODEL_OUTPUTS / "08_enriched_gb_permutation_importance.png"
FINDINGS = MODEL_OUTPUTS / "08_gradient_boosting_findings.md"
MODEL_METADATA = MODEL_OUTPUTS / "08_gradient_boosting.metadata.json"
MANIFEST = MODEL_OUTPUTS / "08_gradient_boosting.validation.json"
PROPERTY_MODEL_FILE = MODEL_FOLDER / "08_property_gradient_boosting.joblib"
ENRICHED_MODEL_FILE = MODEL_FOLDER / "08_enriched_gradient_boosting.joblib"

CATEGORICAL_FEATURES = ["property_type", "duration", "old_new"]
PROPERTY_NUMERIC_FEATURES = ["month_index", "month_sin", "month_cos"]
NEIGHBOURHOOD_FEATURES = ["log_total_crime", "claimant_rate_percent"]
PROPERTY_FEATURES = CATEGORICAL_FEATURES + PROPERTY_NUMERIC_FEATURES
ENRICHED_FEATURES = PROPERTY_FEATURES + NEIGHBOURHOOD_FEATURES

PROPERTY_GB_NAME = "Property-only gradient boosting"
PROPERTY_GB_SMEAR_NAME = "Property-only gradient boosting with smearing"
ENRICHED_GB_NAME = "Enriched gradient boosting"
ENRICHED_GB_SMEAR_NAME = "Enriched gradient boosting with smearing"

PROPERTY_LINEAR_NAME = "Property-only linear regression"
ENRICHED_LINEAR_NAME = "Enriched linear regression"
PROPERTY_RF_NAME = "Property-only random forest"
ENRICHED_RF_NAME = "Enriched random forest"

RANDOM_STATE = 42
MAX_ITER = 300
L2_REGULARIZATION = 1.0
PARAMETER_GRID = {
    "learning_rate": [0.03, 0.07],
    "max_leaf_nodes": [15, 31],
    "min_samples_leaf": [20, 50],
}
FOLD_DEFINITIONS = BASE.FOLD_DEFINITIONS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def money(value: float) -> str:
    return f"£{value:,.0f}"


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    engineered = BASE.engineer_features(data)
    engineered["log_total_crime"] = np.log1p(engineered["total_crime"])
    return engineered


def build_pipeline(
    features: list[str], parameters: dict[str, float | int]
) -> Pipeline:
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
    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=float(parameters["learning_rate"]),
        max_iter=MAX_ITER,
        max_leaf_nodes=int(parameters["max_leaf_nodes"]),
        min_samples_leaf=int(parameters["min_samples_leaf"]),
        l2_regularization=L2_REGULARIZATION,
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])


def parameter_combinations() -> list[dict[str, float | int]]:
    return [
        {
            "learning_rate": learning_rate,
            "max_leaf_nodes": max_leaf_nodes,
            "min_samples_leaf": min_samples_leaf,
        }
        for learning_rate, max_leaf_nodes, min_samples_leaf in itertools.product(
            PARAMETER_GRID["learning_rate"],
            PARAMETER_GRID["max_leaf_nodes"],
            PARAMETER_GRID["min_samples_leaf"],
        )
    ]


def direct_and_smearing_names(feature_set: str) -> tuple[str, str]:
    if feature_set == "property_only":
        return PROPERTY_GB_NAME, PROPERTY_GB_SMEAR_NAME
    if feature_set == "enriched":
        return ENRICHED_GB_NAME, ENRICHED_GB_SMEAR_NAME
    raise ValueError(f"Unknown feature set: {feature_set}")


def tune_feature_set(
    feature_set: str,
    features: list[str],
    fold_data: list[tuple[str, pd.DataFrame, pd.DataFrame]],
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    direct_name, _ = direct_and_smearing_names(feature_set)
    rows: list[dict[str, object]] = []
    combinations = parameter_combinations()
    for candidate_id, parameters in enumerate(combinations, start=1):
        fold_metrics: list[dict[str, object]] = []
        for fold_name, fold_train, fold_validation in fold_data:
            model = build_pipeline(features, parameters)
            model.fit(fold_train[features], fold_train["log_price"])
            predicted_log = model.predict(fold_validation[features])
            predicted_price = np.exp(predicted_log)
            fold_metrics.append(
                BASE.metric_row(
                    direct_name,
                    fold_name,
                    fold_validation["price"].to_numpy(),
                    predicted_price,
                    fold_validation["log_price"].to_numpy(),
                    predicted_log,
                )
            )
        fold_table = pd.DataFrame(fold_metrics)
        rows.append(
            {
                "feature_set": feature_set,
                "candidate_id": candidate_id,
                "max_iter": MAX_ITER,
                "learning_rate": parameters["learning_rate"],
                "max_leaf_nodes": parameters["max_leaf_nodes"],
                "min_samples_leaf": parameters["min_samples_leaf"],
                "l2_regularization": L2_REGULARIZATION,
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
    search = pd.DataFrame(rows).sort_values(
        ["mean_validation_mae_gbp", "mean_validation_rmse_gbp", "candidate_id"]
    ).reset_index(drop=True)
    search.insert(1, "feature_set_rank", np.arange(1, len(search) + 1))
    winner = search.iloc[0]
    selected = {
        "learning_rate": float(winner["learning_rate"]),
        "max_leaf_nodes": int(winner["max_leaf_nodes"]),
        "min_samples_leaf": int(winner["min_samples_leaf"]),
    }
    return search, selected


def evaluate_cross_validation(
    feature_set: str,
    features: list[str],
    parameters: dict[str, float | int],
    fold_data: list[tuple[str, pd.DataFrame, pd.DataFrame]],
) -> pd.DataFrame:
    direct_name, smear_name = direct_and_smearing_names(feature_set)
    rows: list[dict[str, object]] = []
    for fold_name, fold_train, fold_validation in fold_data:
        model = build_pipeline(features, parameters)
        model.fit(fold_train[features], fold_train["log_price"])
        training_log_prediction = model.predict(fold_train[features])
        fold_smearing = BASE.smearing_factor(
            fold_train["log_price"].to_numpy(), training_log_prediction
        )
        validation_log_prediction = model.predict(fold_validation[features])
        direct_prediction = np.exp(validation_log_prediction)
        smearing_prediction = direct_prediction * fold_smearing
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
                    "feature_set": feature_set,
                    "training_start": fold_train["date_of_transfer"].min().date().isoformat(),
                    "training_end": fold_train["date_of_transfer"].max().date().isoformat(),
                    "training_rows": len(fold_train),
                    "validation_start": fold_validation["date_of_transfer"].min().date().isoformat(),
                    "validation_end": fold_validation["date_of_transfer"].max().date().isoformat(),
                    "smearing_factor": fold_smearing if row["model"] == smear_name else np.nan,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def fit_and_test(
    feature_set: str,
    features: list[str],
    parameters: dict[str, float | int],
    development: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[Pipeline, float, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    direct_name, smear_name = direct_and_smearing_names(feature_set)
    model = build_pipeline(features, parameters)
    model.fit(development[features], development["log_price"])
    development_log_prediction = model.predict(development[features])
    smearing = BASE.smearing_factor(
        development["log_price"].to_numpy(), development_log_prediction
    )
    test_log_prediction = model.predict(test[features])
    direct_prediction = np.exp(test_log_prediction)
    smearing_prediction = direct_prediction * smearing
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
    metrics.insert(2, "feature_set", feature_set)
    return model, smearing, metrics, test_log_prediction, direct_prediction, smearing_prediction


def permutation_table(
    feature_set: str,
    model: Pipeline,
    features: list[str],
    test: pd.DataFrame,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        test[features],
        test["log_price"],
        scoring="neg_mean_absolute_error",
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    table = pd.DataFrame(
        {
            "feature_set": feature_set,
            "feature": features,
            "log_mae_increase_mean": result.importances_mean,
            "log_mae_increase_std": result.importances_std,
        }
    ).sort_values("log_mae_increase_mean", ascending=False, ignore_index=True)
    table.insert(1, "rank", np.arange(1, len(table) + 1))
    return table


def grouped_metrics(
    predictions: pd.DataFrame, model_name: str, prediction_column: str
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for property_type, group in predictions.groupby("property_type_label", observed=True):
        row = BASE.metric_row(
            model_name,
            str(property_type),
            group["price"].to_numpy(),
            group[prediction_column].to_numpy(),
            group["log_price"].to_numpy(),
            group[f"{prediction_column}_log"].to_numpy(),
        )
        row["property_type_label"] = property_type
        rows.append(row)
    return pd.DataFrame(rows)


def direct_metric(table: pd.DataFrame, model_name: str) -> pd.Series:
    return table.loc[table["model"].eq(model_name)].iloc[0]


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


def main() -> None:
    MODEL_OUTPUTS.mkdir(parents=True, exist_ok=True)
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
    assert set(development["transaction_id"]).isdisjoint(set(test["transaction_id"]))
    assert development[ENRICHED_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert test[ENRICHED_FEATURES + ["log_price"]].isna().sum().sum() == 0

    fold_data: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for fold_name, train_end, validation_start, validation_end in FOLD_DEFINITIONS:
        fold_train = development.loc[development["sale_month"].le(train_end)].copy()
        fold_validation = development.loc[
            development["sale_month"].between(validation_start, validation_end)
        ].copy()
        assert fold_train["date_of_transfer"].max() < fold_validation["date_of_transfer"].min()
        fold_data.append((fold_name, fold_train, fold_validation))

    property_search, property_parameters = tune_feature_set(
        "property_only", PROPERTY_FEATURES, fold_data
    )
    enriched_search, enriched_parameters = tune_feature_set(
        "enriched", ENRICHED_FEATURES, fold_data
    )
    search_results = pd.concat([property_search, enriched_search], ignore_index=True)
    search_results.to_csv(SEARCH_RESULTS, index=False, float_format="%.6f")

    property_cv = evaluate_cross_validation(
        "property_only", PROPERTY_FEATURES, property_parameters, fold_data
    )
    enriched_cv = evaluate_cross_validation(
        "enriched", ENRICHED_FEATURES, enriched_parameters, fold_data
    )
    cv_metrics = pd.concat([property_cv, enriched_cv], ignore_index=True)
    cv_metrics.to_csv(CV_METRICS, index=False, float_format="%.6f")

    (
        property_model,
        property_smearing,
        property_test_metrics,
        property_log_prediction,
        property_prediction,
        property_smearing_prediction,
    ) = fit_and_test(
        "property_only", PROPERTY_FEATURES, property_parameters, development, test
    )
    (
        enriched_model,
        enriched_smearing,
        enriched_test_metrics,
        enriched_log_prediction,
        enriched_prediction,
        enriched_smearing_prediction,
    ) = fit_and_test(
        "enriched", ENRICHED_FEATURES, enriched_parameters, development, test
    )
    test_metrics = pd.concat(
        [property_test_metrics, enriched_test_metrics], ignore_index=True
    )
    test_metrics.to_csv(TEST_METRICS, index=False, float_format="%.6f")

    importance = pd.concat(
        [
            permutation_table("property_only", property_model, PROPERTY_FEATURES, test),
            permutation_table("enriched", enriched_model, ENRICHED_FEATURES, test),
        ],
        ignore_index=True,
    )
    importance.to_csv(PERMUTATION_IMPORTANCE, index=False, float_format="%.8f")

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
    predictions["log_total_crime"] = test["log_total_crime"].to_numpy()
    predictions["log_price"] = test["log_price"].to_numpy()
    predictions["property_gb_predicted_log_price"] = property_log_prediction
    predictions["property_gb_predicted_price"] = property_prediction
    predictions["property_gb_smearing_predicted_price"] = property_smearing_prediction
    predictions["enriched_gb_predicted_log_price"] = enriched_log_prediction
    predictions["enriched_gb_predicted_price"] = enriched_prediction
    predictions["enriched_gb_smearing_predicted_price"] = enriched_smearing_prediction
    predictions["property_gb_absolute_error"] = (
        predictions["property_gb_predicted_price"] - predictions["price"]
    ).abs()
    predictions["enriched_gb_absolute_error"] = (
        predictions["enriched_gb_predicted_price"] - predictions["price"]
    ).abs()
    predictions["absolute_error_improvement_from_enrichment"] = (
        predictions["property_gb_absolute_error"]
        - predictions["enriched_gb_absolute_error"]
    )
    predictions["enriched_gb_absolute_percentage_error_percent"] = (
        predictions["enriched_gb_absolute_error"] / predictions["price"] * 100
    )
    predictions.to_csv(TEST_PREDICTIONS, index=False, float_format="%.6f")

    property_metric_data = predictions.copy()
    property_metric_data["property_gb_predicted_price_log"] = property_log_prediction
    enriched_metric_data = predictions.copy()
    enriched_metric_data["enriched_gb_predicted_price_log"] = enriched_log_prediction
    error_by_type = pd.concat(
        [
            grouped_metrics(
                property_metric_data,
                PROPERTY_GB_NAME,
                "property_gb_predicted_price",
            ),
            grouped_metrics(
                enriched_metric_data,
                ENRICHED_GB_NAME,
                "enriched_gb_predicted_price",
            ),
        ],
        ignore_index=True,
    )
    error_by_type.to_csv(ERROR_BY_TYPE, index=False, float_format="%.6f")

    monthly_rows: list[dict[str, object]] = []
    for sale_month, group in predictions.groupby("sale_month", observed=True):
        for model_name, predicted_price_column, predicted_log_column in [
            (
                PROPERTY_GB_NAME,
                "property_gb_predicted_price",
                "property_gb_predicted_log_price",
            ),
            (
                ENRICHED_GB_NAME,
                "enriched_gb_predicted_price",
                "enriched_gb_predicted_log_price",
            ),
        ]:
            row = BASE.metric_row(
                model_name,
                str(sale_month),
                group["price"].to_numpy(),
                group[predicted_price_column].to_numpy(),
                group["log_price"].to_numpy(),
                group[predicted_log_column].to_numpy(),
            )
            row.update(
                {
                    "sale_month": sale_month,
                    "actual_median_price": group["price"].median(),
                    "predicted_median_price": group[predicted_price_column].median(),
                    "actual_mean_price": group["price"].mean(),
                    "predicted_mean_price": group[predicted_price_column].mean(),
                }
            )
            monthly_rows.append(row)
    monthly_performance = pd.DataFrame(monthly_rows).sort_values(["sale_month", "model"])
    monthly_performance.to_csv(MONTHLY_PERFORMANCE, index=False, float_format="%.6f")

    previous_tables = [
        (pd.read_csv(PROPERTY_LINEAR_TEST), PROPERTY_LINEAR_NAME),
        (pd.read_csv(ENRICHED_LINEAR_TEST), ENRICHED_LINEAR_NAME),
        (pd.read_csv(PROPERTY_RF_TEST), PROPERTY_RF_NAME),
        (pd.read_csv(ENRICHED_RF_TEST), ENRICHED_RF_NAME),
    ]
    comparison_rows = [
        table.loc[table["model"].eq(name)].copy() for table, name in previous_tables
    ]
    comparison_rows.append(
        test_metrics.loc[
            test_metrics["model"].isin([PROPERTY_GB_NAME, ENRICHED_GB_NAME])
        ].copy()
    )
    model_comparison = pd.concat(comparison_rows, ignore_index=True)
    property_gb_direct = direct_metric(test_metrics, PROPERTY_GB_NAME)
    enriched_gb_direct = direct_metric(test_metrics, ENRICHED_GB_NAME)
    model_comparison["mae_change_vs_property_gb_gbp"] = (
        model_comparison["mae_gbp"] - property_gb_direct["mae_gbp"]
    )
    model_comparison["mae_improvement_vs_property_gb_percent"] = (
        (property_gb_direct["mae_gbp"] - model_comparison["mae_gbp"])
        / property_gb_direct["mae_gbp"]
        * 100
    )
    model_comparison["rmse_change_vs_property_gb_gbp"] = (
        model_comparison["rmse_gbp"] - property_gb_direct["rmse_gbp"]
    )
    model_comparison["rmse_improvement_vs_property_gb_percent"] = (
        (property_gb_direct["rmse_gbp"] - model_comparison["rmse_gbp"])
        / property_gb_direct["rmse_gbp"]
        * 100
    )
    model_comparison["r2_price_change_vs_property_gb"] = (
        model_comparison["r2_price"] - property_gb_direct["r2_price"]
    )
    model_comparison.to_csv(MODEL_COMPARISON, index=False, float_format="%.6f")

    BASE.set_chart_style()
    fig, axis = plt.subplots(figsize=(8.4, 7.0))
    for property_type in ["Detached", "Flat/Maisonette", "Semi-detached", "Terraced"]:
        group = predictions.loc[predictions["property_type_label"].eq(property_type)]
        axis.scatter(
            group["price"],
            group["enriched_gb_predicted_price"],
            s=17,
            alpha=0.42,
            color=BASE.TYPE_COLORS[property_type],
            label=property_type,
            edgecolor="none",
        )
    chart_min = min(
        predictions["price"].min(), predictions["enriched_gb_predicted_price"].min()
    )
    chart_max = max(
        predictions["price"].max(), predictions["enriched_gb_predicted_price"].max()
    )
    axis.plot(
        [chart_min, chart_max],
        [chart_min, chart_max],
        color=BASE.NAVY,
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
        f"MAE {money(enriched_gb_direct['mae_gbp'])}\n"
        f"RMSE {money(enriched_gb_direct['rmse_gbp'])}\n"
        f"R² {enriched_gb_direct['r2_price']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=BASE.NAVY,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": BASE.LIGHT_GREY},
    )
    axis.set_title("Enriched gradient boosting: actual versus predicted", pad=14)
    fig.suptitle("2024 holdout predictions", fontsize=12, color=BASE.GREY, y=0.96)
    fig.tight_layout()
    fig.savefig(ACTUAL_PREDICTED_CHART, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    chart_models = [
        PROPERTY_LINEAR_NAME,
        PROPERTY_RF_NAME,
        PROPERTY_GB_NAME,
        ENRICHED_LINEAR_NAME,
        ENRICHED_RF_NAME,
        ENRICHED_GB_NAME,
    ]
    chart_labels = [
        "Property\nlinear",
        "Property\nforest",
        "Property\nboosting",
        "Enriched\nlinear",
        "Enriched\nforest",
        "Enriched\nboosting",
    ]
    chart_data = model_comparison.set_index("model").loc[chart_models]
    x = np.arange(len(chart_labels))
    width = 0.34
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 5.9))
    mae_values = chart_data["mae_gbp"].to_numpy()
    rmse_values = chart_data["rmse_gbp"].to_numpy()
    bars_mae = axes[0].bar(x - width / 2, mae_values, width, label="MAE", color=BASE.BLUE)
    bars_rmse = axes[0].bar(x + width / 2, rmse_values, width, label="RMSE", color=BASE.ORANGE)
    axes[0].set_xticks(x, chart_labels)
    axes[0].set_ylabel("Error (£)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[0].set_title("2024 prediction error")
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    for bars in [bars_mae, bars_rmse]:
        for bar in bars:
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + max(rmse_values) * 0.018,
                f"£{bar.get_height() / 1_000:.0f}k",
                ha="center",
                va="bottom",
                fontsize=8,
                color=BASE.NAVY,
            )
    axes[0].set_ylim(0, max(rmse_values) * 1.19)

    r2_values = chart_data["r2_price"].to_numpy()
    colors = [BASE.GREY, BASE.GREY, BASE.GREY, BASE.BLUE, BASE.TEAL, BASE.PURPLE]
    r2_bars = axes[1].bar(x, r2_values, width=0.62, color=colors)
    axes[1].set_xticks(x, chart_labels)
    axes[1].set_ylabel("Price R²")
    axes[1].set_title("Explained price variation")
    axes[1].grid(axis="y")
    axes[1].set_ylim(0, max(r2_values) * 1.25)
    for bar in r2_bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(r2_values) * 0.02,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color=BASE.NAVY,
        )
    fig.suptitle(
        "Property-only and enriched model comparison",
        fontsize=17,
        fontweight="bold",
        color=BASE.NAVY,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(PERFORMANCE_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    enriched_importance = importance.loc[
        importance["feature_set"].eq("enriched")
    ].sort_values("log_mae_increase_mean", ascending=True)
    fig, axis = plt.subplots(figsize=(9.0, 5.8))
    axis.barh(
        enriched_importance["feature"],
        enriched_importance["log_mae_increase_mean"],
        xerr=enriched_importance["log_mae_increase_std"],
        color=BASE.TEAL,
        alpha=0.9,
        error_kw={"ecolor": BASE.GREY, "capsize": 3},
    )
    axis.set_xlabel("Increase in log-price MAE after permutation")
    axis.set_title("Enriched gradient boosting: permutation importance")
    axis.grid(axis="x")
    fig.tight_layout()
    fig.savefig(IMPORTANCE_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    joblib.dump(
        {
            "pipeline": property_model,
            "smearing_factor": property_smearing,
            "model_features": PROPERTY_FEATURES,
            "target": "log(price)",
            "selected_parameters": property_parameters,
            "max_iter": MAX_ITER,
            "l2_regularization": L2_REGULARIZATION,
            "random_state": RANDOM_STATE,
            "training_period": "2022-01 to 2023-12",
        },
        PROPERTY_MODEL_FILE,
    )
    joblib.dump(
        {
            "pipeline": enriched_model,
            "smearing_factor": enriched_smearing,
            "model_features": ENRICHED_FEATURES,
            "target": "log(price)",
            "selected_parameters": enriched_parameters,
            "max_iter": MAX_ITER,
            "l2_regularization": L2_REGULARIZATION,
            "random_state": RANDOM_STATE,
            "training_period": "2022-01 to 2023-12",
        },
        ENRICHED_MODEL_FILE,
    )

    enriched_linear_cv = pd.read_csv(ENRICHED_LINEAR_CV)
    enriched_rf_cv = pd.read_csv(ENRICHED_RF_CV)
    validation_comparison = pd.DataFrame(
        [
            {
                "model": ENRICHED_LINEAR_NAME,
                **mean_cv_metric(enriched_linear_cv, ENRICHED_LINEAR_NAME).to_dict(),
            },
            {
                "model": ENRICHED_RF_NAME,
                **mean_cv_metric(enriched_rf_cv, ENRICHED_RF_NAME).to_dict(),
            },
            {
                "model": ENRICHED_GB_NAME,
                **mean_cv_metric(cv_metrics, ENRICHED_GB_NAME).to_dict(),
            },
        ]
    ).sort_values(["mae_gbp", "rmse_gbp"]).reset_index(drop=True)
    selected_final_model = validation_comparison.iloc[0]["model"]

    property_cv_summary = mean_cv_metric(cv_metrics, PROPERTY_GB_NAME)
    enriched_cv_summary = mean_cv_metric(cv_metrics, ENRICHED_GB_NAME)
    validation_mae_improvement = (
        (property_cv_summary["mae_gbp"] - enriched_cv_summary["mae_gbp"])
        / property_cv_summary["mae_gbp"]
        * 100
    )
    validation_rmse_improvement = (
        (property_cv_summary["rmse_gbp"] - enriched_cv_summary["rmse_gbp"])
        / property_cv_summary["rmse_gbp"]
        * 100
    )
    test_mae_improvement = (
        (property_gb_direct["mae_gbp"] - enriched_gb_direct["mae_gbp"])
        / property_gb_direct["mae_gbp"]
        * 100
    )
    test_rmse_improvement = (
        (property_gb_direct["rmse_gbp"] - enriched_gb_direct["rmse_gbp"])
        / property_gb_direct["rmse_gbp"]
        * 100
    )
    top_importance_text = ", ".join(
        f"`{row.feature}` ({row.log_mae_increase_mean:.4f})"
        for row in importance.loc[importance["feature_set"].eq("enriched")]
        .head(5)
        .itertuples(index=False)
    )

    enriched_linear_direct = direct_metric(pd.read_csv(ENRICHED_LINEAR_TEST), ENRICHED_LINEAR_NAME)
    enriched_rf_direct = direct_metric(pd.read_csv(ENRICHED_RF_TEST), ENRICHED_RF_NAME)
    findings = f"""# Gradient-boosting comparison

## Purpose

This checkpoint compares property-only and enriched histogram gradient-boosting models. Both predict `log(price)` using the same Category A transactions, chronological folds and 2024 holdout. The enriched model adds `log1p(total_crime)` and `claimant_rate_percent`.

## Model design

Eight predefined parameter settings were assessed separately for each feature set. All models used {MAX_ITER} boosting iterations, L2 regularisation of {L2_REGULARIZATION}, early stopping disabled and random seed {RANDOM_STATE}. Parameter selection used mean validation MAE; 2024 was not used for tuning.

Selected property-only parameters: learning rate **{property_parameters['learning_rate']}**, maximum leaves **{property_parameters['max_leaf_nodes']}**, minimum leaf size **{property_parameters['min_samples_leaf']}**.

Selected enriched parameters: learning rate **{enriched_parameters['learning_rate']}**, maximum leaves **{enriched_parameters['max_leaf_nodes']}**, minimum leaf size **{enriched_parameters['min_samples_leaf']}**.

## Time-based validation

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only gradient boosting | {money(property_cv_summary['mae_gbp'])} | {money(property_cv_summary['rmse_gbp'])} | {property_cv_summary['r2_price']:.3f} |
| Enriched gradient boosting | {money(enriched_cv_summary['mae_gbp'])} | {money(enriched_cv_summary['rmse_gbp'])} | {enriched_cv_summary['r2_price']:.3f} |

Enrichment reduces mean validation MAE by **{validation_mae_improvement:.1f}%** and mean validation RMSE by **{validation_rmse_improvement:.1f}%**.

## Final 2024 comparison

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only gradient boosting | {money(property_gb_direct['mae_gbp'])} | {money(property_gb_direct['rmse_gbp'])} | {property_gb_direct['r2_price']:.3f} | {property_gb_direct['r2_log_price']:.3f} |
| Enriched linear regression | {money(enriched_linear_direct['mae_gbp'])} | {money(enriched_linear_direct['rmse_gbp'])} | {enriched_linear_direct['r2_price']:.3f} | {enriched_linear_direct['r2_log_price']:.3f} |
| Enriched random forest | {money(enriched_rf_direct['mae_gbp'])} | {money(enriched_rf_direct['rmse_gbp'])} | {enriched_rf_direct['r2_price']:.3f} | {enriched_rf_direct['r2_log_price']:.3f} |
| Enriched gradient boosting | {money(enriched_gb_direct['mae_gbp'])} | {money(enriched_gb_direct['rmse_gbp'])} | {enriched_gb_direct['r2_price']:.3f} | {enriched_gb_direct['r2_log_price']:.3f} |

Compared with property-only gradient boosting, enrichment reduces 2024 MAE by **{test_mae_improvement:.1f}%** and RMSE by **{test_rmse_improvement:.1f}%**. Price R² changes by **{enriched_gb_direct['r2_price'] - property_gb_direct['r2_price']:+.3f}**.

## Model selection

The planned final-model candidate is **{selected_final_model}**, selected using the lowest mean validation MAE across the three enriched algorithms. This rule avoids choosing solely from the 2024 holdout result. A leakage-sensitivity check should still be completed before the dashboard is finalised.

## Permutation importance

The largest enriched-model permutation effects were {top_importance_text}. The values are increases in log-price MAE after shuffling a feature. They measure predictive reliance, not causation.

## Limitations

- The Land Registry data lacks bedrooms, floor area, condition and detailed amenities.
- Contemporaneous monthly crime can include incidents occurring after an early-month sale.
- The claimant-rate denominator is the fixed 2020 working-age population.
- Neighbourhood variables may proxy for omitted location and housing-stock characteristics.
- The same 2024 holdout is reported for several pre-planned model families; final selection therefore uses validation performance.

## Next checkpoint

Run a lagged-crime sensitivity experiment for the selected enriched model. This will show whether the predictive improvement remains when each transaction uses crime information available before the sale month.
"""
    FINDINGS.write_text(findings, encoding="utf-8")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "estimator": "sklearn.ensemble.HistGradientBoostingRegressor",
        "sklearn_version": sklearn.__version__,
        "cohort": "PPD Category A",
        "target": "log(price)",
        "property_features": PROPERTY_FEATURES,
        "enriched_features": ENRICHED_FEATURES,
        "neighbourhood_features": NEIGHBOURHOOD_FEATURES,
        "crime_transformation": "log1p(total_crime)",
        "parameter_grid": PARAMETER_GRID,
        "max_iter": MAX_ITER,
        "l2_regularization": L2_REGULARIZATION,
        "early_stopping": False,
        "random_state": RANDOM_STATE,
        "selected_property_parameters": property_parameters,
        "selected_enriched_parameters": enriched_parameters,
        "final_model_selection_rule": "lowest mean expanding-window validation MAE among enriched algorithms",
        "selected_final_model_candidate": selected_final_model,
        "development_period": "2022-01-04 to 2023-12-22",
        "test_period": "2024-01-02 to 2024-12-31",
        "development_rows": len(development),
        "test_rows": len(test),
        "model_files": {
            "property_only": str(PROPERTY_MODEL_FILE.relative_to(PROJECT_ROOT)),
            "enriched": str(ENRICHED_MODEL_FILE.relative_to(PROJECT_ROOT)),
        },
        "model_sha256": {
            "property_only": sha256(PROPERTY_MODEL_FILE),
            "enriched": sha256(ENRICHED_MODEL_FILE),
        },
        "validation_model_comparison": validation_comparison.to_dict(orient="records"),
    }
    MODEL_METADATA.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    output_files = [
        SEARCH_RESULTS,
        CV_METRICS,
        TEST_METRICS,
        PERMUTATION_IMPORTANCE,
        TEST_PREDICTIONS,
        ERROR_BY_TYPE,
        MONTHLY_PERFORMANCE,
        MODEL_COMPARISON,
        ACTUAL_PREDICTED_CHART,
        PERFORMANCE_CHART,
        IMPORTANCE_CHART,
        FINDINGS,
        MODEL_METADATA,
        PROPERTY_MODEL_FILE,
        ENRICHED_MODEL_FILE,
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
                development[ENRICHED_FEATURES + ["log_price"]].isna().sum().sum()
            ),
            "test_missing_model_values": int(
                test[ENRICHED_FEATURES + ["log_price"]].isna().sum().sum()
            ),
            "parameter_candidates_per_feature_set": len(parameter_combinations()),
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
                                "property_gb_predicted_log_price",
                                "property_gb_predicted_price",
                                "enriched_gb_predicted_log_price",
                                "enriched_gb_predicted_price",
                            ]
                        ]
                    )
                ).sum().sum()
            ),
            "permutation_importance_rows": len(importance),
        },
        "selected_parameters": {
            "property_only": property_parameters,
            "enriched": enriched_parameters,
        },
        "test_metrics": {
            "property_only": {
                "mae_gbp": float(property_gb_direct["mae_gbp"]),
                "rmse_gbp": float(property_gb_direct["rmse_gbp"]),
                "r2_price": float(property_gb_direct["r2_price"]),
            },
            "enriched": {
                "mae_gbp": float(enriched_gb_direct["mae_gbp"]),
                "rmse_gbp": float(enriched_gb_direct["rmse_gbp"]),
                "r2_price": float(enriched_gb_direct["r2_price"]),
            },
            "mae_improvement_from_enrichment_percent": float(test_mae_improvement),
            "rmse_improvement_from_enrichment_percent": float(test_rmse_improvement),
        },
        "selected_final_model_candidate": selected_final_model,
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Development rows: {len(development):,}")
    print(f"Test rows: {len(test):,}")
    print(f"Candidates per feature set: {len(parameter_combinations())}")
    print(f"Property parameters: {property_parameters}")
    print(f"Enriched parameters: {enriched_parameters}")
    print(f"Property GB MAE: {money(property_gb_direct['mae_gbp'])}")
    print(f"Enriched GB MAE: {money(enriched_gb_direct['mae_gbp'])}")
    print(f"Enriched GB RMSE: {money(enriched_gb_direct['rmse_gbp'])}")
    print(f"Enriched GB price R2: {enriched_gb_direct['r2_price']:.4f}")
    print(f"Selected final-model candidate: {selected_final_model}")


if __name__ == "__main__":
    main()
