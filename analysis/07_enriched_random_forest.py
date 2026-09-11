"""Model checkpoint 4: enriched random forest with a 2024 holdout.

This experiment keeps the property-only random-forest design unchanged and
adds two neighbourhood-condition variables: monthly LSOA crime and claimant
rate. Hyperparameters are selected using expanding-window validation within
2022-2023; the 2024 holdout is used once for final comparison.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODULE_PATH = Path(__file__).with_name("06_property_only_random_forest.py")
BASE_SPEC = importlib.util.spec_from_file_location("property_rf_checkpoint", BASE_MODULE_PATH)
if BASE_SPEC is None or BASE_SPEC.loader is None:
    raise ImportError(f"Cannot load shared model helpers from {BASE_MODULE_PATH}")
BASE = importlib.util.module_from_spec(BASE_SPEC)
BASE_SPEC.loader.exec_module(BASE)

DATASET = PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2022_2024.csv"
PROPERTY_RF_TEST = PROJECT_ROOT / "outputs" / "models" / "06_property_rf_test_metrics.csv"
PROPERTY_RF_CV = PROJECT_ROOT / "outputs" / "models" / "06_property_rf_cross_validation_metrics.csv"
PROPERTY_RF_PREDICTIONS = PROJECT_ROOT / "outputs" / "models" / "06_property_rf_2024_predictions.csv"
ENRICHED_LINEAR_TEST = PROJECT_ROOT / "outputs" / "models" / "05_enriched_test_metrics.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "models"
MODEL_FOLDER = PROJECT_ROOT / "models"

SEARCH_RESULTS = OUTPUT_FOLDER / "07_enriched_rf_hyperparameter_search.csv"
CV_METRICS = OUTPUT_FOLDER / "07_enriched_rf_cross_validation_metrics.csv"
TEST_METRICS = OUTPUT_FOLDER / "07_enriched_rf_test_metrics.csv"
FEATURE_IMPORTANCE = OUTPUT_FOLDER / "07_enriched_rf_feature_importance.csv"
TEST_PREDICTIONS = OUTPUT_FOLDER / "07_enriched_rf_2024_predictions.csv"
ERROR_BY_TYPE = OUTPUT_FOLDER / "07_enriched_rf_error_by_property_type.csv"
MONTHLY_PERFORMANCE = OUTPUT_FOLDER / "07_enriched_rf_monthly_performance.csv"
MODEL_COMPARISON = OUTPUT_FOLDER / "07_enriched_rf_model_comparison.csv"
ACTUAL_PREDICTED_CHART = OUTPUT_FOLDER / "07_enriched_rf_actual_vs_predicted.png"
PERFORMANCE_CHART = OUTPUT_FOLDER / "07_enriched_rf_performance.png"
FINDINGS = OUTPUT_FOLDER / "07_enriched_rf_findings.md"
MODEL_METADATA = OUTPUT_FOLDER / "07_enriched_rf.metadata.json"
MANIFEST = OUTPUT_FOLDER / "07_enriched_rf.validation.json"
MODEL_FILE = MODEL_FOLDER / "07_enriched_random_forest.joblib"

CATEGORICAL_FEATURES = ["property_type", "duration", "old_new"]
PROPERTY_NUMERIC_FEATURES = ["month_index", "month_sin", "month_cos"]
NEIGHBOURHOOD_FEATURES = ["log_total_crime", "claimant_rate_percent"]
NUMERIC_FEATURES = PROPERTY_NUMERIC_FEATURES + NEIGHBOURHOOD_FEATURES
MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

PROPERTY_RF_NAME = "Property-only random forest"
ENRICHED_LINEAR_NAME = "Enriched linear regression"
ENRICHED_RF_NAME = "Enriched random forest"
ENRICHED_RF_SMEAR_NAME = "Enriched random forest with smearing"

RANDOM_STATE = BASE.RANDOM_STATE
N_ESTIMATORS = BASE.N_ESTIMATORS
PARAMETER_GRID = BASE.PARAMETER_GRID
FOLD_DEFINITIONS = BASE.FOLD_DEFINITIONS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    engineered = BASE.engineer_features(data)
    engineered["log_total_crime"] = np.log1p(engineered["total_crime"])
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


def money(value: float) -> str:
    return f"£{value:,.0f}"


def percent_improvement(reference: float, candidate: float) -> float:
    return (reference - candidate) / reference * 100


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(DATASET, low_memory=False)
    cohort = engineer_features(source.loc[source["ppd_category"].eq("A")].copy())
    cohort["log_price"] = np.log(cohort["price"])
    cohort = cohort.sort_values(["date_of_transfer", "transaction_id"]).reset_index(drop=True)
    development = cohort.loc[cohort["sale_year"].le(2023)].copy()
    test = cohort.loc[cohort["sale_year"].eq(2024)].copy()

    property_rf_test = pd.read_csv(PROPERTY_RF_TEST)
    property_rf_cv = pd.read_csv(PROPERTY_RF_CV)
    property_rf_predictions = pd.read_csv(PROPERTY_RF_PREDICTIONS)
    enriched_linear_test = pd.read_csv(ENRICHED_LINEAR_TEST)

    assert len(source) == 21_968
    assert len(cohort) == 18_670
    assert len(development) == 12_785
    assert len(test) == 5_885
    assert development["date_of_transfer"].max() < test["date_of_transfer"].min()
    assert set(development["transaction_id"]).isdisjoint(set(test["transaction_id"]))
    assert development[MODEL_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert test[MODEL_FEATURES + ["log_price"]].isna().sum().sum() == 0
    assert (development["total_crime"] >= 0).all() and (test["total_crime"] >= 0).all()
    assert (development["claimant_rate_percent"] >= 0).all()
    assert (test["claimant_rate_percent"] >= 0).all()
    assert property_rf_predictions["transaction_id"].tolist() == test["transaction_id"].tolist()

    fold_data: list[tuple[str, pd.DataFrame, pd.DataFrame]] = []
    for fold_name, train_end, validation_start, validation_end in FOLD_DEFINITIONS:
        fold_train = development.loc[development["sale_month"].le(train_end)].copy()
        fold_validation = development.loc[
            development["sale_month"].between(validation_start, validation_end)
        ].copy()
        assert fold_train["date_of_transfer"].max() < fold_validation["date_of_transfer"].min()
        fold_data.append((fold_name, fold_train, fold_validation))

    combinations = parameter_combinations()
    search_rows: list[dict[str, object]] = []
    for candidate_id, parameters in enumerate(combinations, start=1):
        fold_metrics: list[dict[str, object]] = []
        for fold_name, fold_train, fold_validation in fold_data:
            model = build_pipeline(parameters)
            model.fit(fold_train[MODEL_FEATURES], fold_train["log_price"])
            predicted_log = model.predict(fold_validation[MODEL_FEATURES])
            predicted_price = np.exp(predicted_log)
            fold_metrics.append(
                BASE.metric_row(
                    ENRICHED_RF_NAME,
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
        fold_smearing = BASE.smearing_factor(
            fold_train["log_price"].to_numpy(), training_log_prediction
        )
        validation_log_prediction = model.predict(fold_validation[MODEL_FEATURES])
        direct_prediction = np.exp(validation_log_prediction)
        smearing_prediction = direct_prediction * fold_smearing
        for row in [
            BASE.metric_row(
                ENRICHED_RF_NAME,
                fold_name,
                fold_validation["price"].to_numpy(),
                direct_prediction,
                fold_validation["log_price"].to_numpy(),
                validation_log_prediction,
            ),
            BASE.metric_row(
                ENRICHED_RF_SMEAR_NAME,
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
                    if row["model"] == ENRICHED_RF_SMEAR_NAME
                    else np.nan,
                }
            )
            cv_rows.append(row)
    cv_metrics = pd.DataFrame(cv_rows)
    cv_metrics.to_csv(CV_METRICS, index=False, float_format="%.6f")

    final_model = build_pipeline(selected_parameters)
    final_model.fit(development[MODEL_FEATURES], development["log_price"])
    development_log_prediction = final_model.predict(development[MODEL_FEATURES])
    final_smearing = BASE.smearing_factor(
        development["log_price"].to_numpy(), development_log_prediction
    )
    test_log_prediction = final_model.predict(test[MODEL_FEATURES])
    test_price_prediction = np.exp(test_log_prediction)
    test_smearing_prediction = test_price_prediction * final_smearing

    test_metrics = pd.DataFrame(
        [
            BASE.metric_row(
                ENRICHED_RF_NAME,
                "2024 holdout",
                test["price"].to_numpy(),
                test_price_prediction,
                test["log_price"].to_numpy(),
                test_log_prediction,
            ),
            BASE.metric_row(
                ENRICHED_RF_SMEAR_NAME,
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
            "log_total_crime": "Log monthly crime count in the 2011 LSOA",
            "claimant_rate_percent": "Monthly claimant rate in the 2011 LSOA",
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
            "total_crime",
            "claimant_rate_percent",
        ]
    ].copy()
    predictions["log_total_crime"] = test["log_total_crime"].to_numpy()
    predictions["log_price"] = test["log_price"].to_numpy()
    predictions["predicted_log_price"] = test_log_prediction
    predictions["predicted_price"] = test_price_prediction
    predictions["smearing_predicted_price"] = test_smearing_prediction
    predictions["property_rf_predicted_price"] = property_rf_predictions[
        "predicted_price"
    ].to_numpy()
    predictions["prediction_error"] = predictions["predicted_price"] - predictions["price"]
    predictions["absolute_error"] = predictions["prediction_error"].abs()
    predictions["property_rf_absolute_error"] = (
        predictions["property_rf_predicted_price"] - predictions["price"]
    ).abs()
    predictions["absolute_error_improvement_vs_property_rf"] = (
        predictions["property_rf_absolute_error"] - predictions["absolute_error"]
    )
    predictions["absolute_percentage_error_percent"] = (
        predictions["absolute_error"] / predictions["price"] * 100
    )
    predictions.to_csv(TEST_PREDICTIONS, index=False, float_format="%.6f")

    error_rows: list[dict[str, object]] = []
    for model_name, prediction_column in [
        (ENRICHED_RF_NAME, "predicted_price"),
        (ENRICHED_RF_SMEAR_NAME, "smearing_predicted_price"),
    ]:
        for property_type, group in predictions.groupby("property_type_label", observed=True):
            row = BASE.metric_row(
                model_name,
                str(property_type),
                group["price"].to_numpy(),
                group[prediction_column].to_numpy(),
                group["log_price"].to_numpy(),
                group["predicted_log_price"].to_numpy(),
            )
            row["property_type_label"] = property_type
            error_rows.append(row)
    error_by_type = pd.DataFrame(error_rows)
    error_by_type.to_csv(ERROR_BY_TYPE, index=False, float_format="%.6f")

    monthly_rows: list[dict[str, object]] = []
    for sale_month, group in predictions.groupby("sale_month", observed=True):
        row = BASE.metric_row(
            ENRICHED_RF_NAME,
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
                "property_rf_predicted_median_price": group[
                    "property_rf_predicted_price"
                ].median(),
                "enriched_rf_predicted_median_price": group["predicted_price"].median(),
                "actual_mean_price": group["price"].mean(),
                "property_rf_predicted_mean_price": group[
                    "property_rf_predicted_price"
                ].mean(),
                "enriched_rf_predicted_mean_price": group["predicted_price"].mean(),
            }
        )
        monthly_rows.append(row)
    monthly_performance = pd.DataFrame(monthly_rows).sort_values("sale_month")
    monthly_performance.to_csv(MONTHLY_PERFORMANCE, index=False, float_format="%.6f")

    property_rf_direct = property_rf_test.loc[
        property_rf_test["model"].eq(PROPERTY_RF_NAME)
    ].iloc[0]
    enriched_linear_direct = enriched_linear_test.loc[
        enriched_linear_test["model"].eq(ENRICHED_LINEAR_NAME)
    ].iloc[0]
    enriched_rf_direct = test_metrics.loc[test_metrics["model"].eq(ENRICHED_RF_NAME)].iloc[0]
    enriched_rf_smear = test_metrics.loc[
        test_metrics["model"].eq(ENRICHED_RF_SMEAR_NAME)
    ].iloc[0]

    comparison = pd.concat(
        [
            property_rf_test.loc[property_rf_test["model"].eq(PROPERTY_RF_NAME)].copy(),
            enriched_linear_test.loc[
                enriched_linear_test["model"].eq(ENRICHED_LINEAR_NAME)
            ].copy(),
            test_metrics.copy(),
        ],
        ignore_index=True,
    )
    comparison["mae_change_vs_property_rf_gbp"] = (
        comparison["mae_gbp"] - property_rf_direct["mae_gbp"]
    )
    comparison["mae_improvement_vs_property_rf_percent"] = (
        (property_rf_direct["mae_gbp"] - comparison["mae_gbp"])
        / property_rf_direct["mae_gbp"]
        * 100
    )
    comparison["rmse_change_vs_property_rf_gbp"] = (
        comparison["rmse_gbp"] - property_rf_direct["rmse_gbp"]
    )
    comparison["rmse_improvement_vs_property_rf_percent"] = (
        (property_rf_direct["rmse_gbp"] - comparison["rmse_gbp"])
        / property_rf_direct["rmse_gbp"]
        * 100
    )
    comparison["r2_price_change_vs_property_rf"] = (
        comparison["r2_price"] - property_rf_direct["r2_price"]
    )
    comparison.to_csv(MODEL_COMPARISON, index=False, float_format="%.6f")

    BASE.set_chart_style()
    fig, axis = plt.subplots(figsize=(8.4, 7.0))
    for property_type in ["Detached", "Flat/Maisonette", "Semi-detached", "Terraced"]:
        group = predictions.loc[predictions["property_type_label"].eq(property_type)]
        axis.scatter(
            group["price"],
            group["predicted_price"],
            s=17,
            alpha=0.42,
            color=BASE.TYPE_COLORS[property_type],
            label=property_type,
            edgecolor="none",
        )
    chart_min = min(predictions["price"].min(), predictions["predicted_price"].min())
    chart_max = max(predictions["price"].max(), predictions["predicted_price"].max())
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
        f"MAE {money(enriched_rf_direct['mae_gbp'])}\n"
        f"RMSE {money(enriched_rf_direct['rmse_gbp'])}\n"
        f"R² {enriched_rf_direct['r2_price']:.3f}",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        color=BASE.NAVY,
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": "white", "edgecolor": BASE.LIGHT_GREY},
    )
    axis.set_title("Enriched random forest: actual versus predicted", pad=14)
    fig.suptitle("2024 holdout predictions", fontsize=12, color=BASE.GREY, y=0.96)
    fig.tight_layout()
    fig.savefig(ACTUAL_PREDICTED_CHART, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    chart_models = [PROPERTY_RF_NAME, ENRICHED_LINEAR_NAME, ENRICHED_RF_NAME]
    chart_labels = ["Property-only\nrandom forest", "Enriched\nlinear", "Enriched\nrandom forest"]
    chart_data = comparison.set_index("model").loc[chart_models]
    x = np.arange(len(chart_labels))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7))
    mae_values = chart_data["mae_gbp"].to_numpy()
    rmse_values = chart_data["rmse_gbp"].to_numpy()
    bars_mae = axes[0].bar(x - width / 2, mae_values, width, label="MAE", color=BASE.BLUE)
    bars_rmse = axes[0].bar(x + width / 2, rmse_values, width, label="RMSE", color=BASE.ORANGE)
    axes[0].set_xticks(x, chart_labels)
    axes[0].set_ylabel("Error (£)")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[0].set_title("2024 model error comparison")
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
                color=BASE.NAVY,
            )
    axes[0].set_ylim(0, max(rmse_values) * 1.20)

    r2_values = chart_data["r2_price"].to_numpy()
    r2_bars = axes[1].bar(x, r2_values, width=0.55, color=[BASE.GREY, BASE.BLUE, BASE.TEAL])
    axes[1].set_xticks(x, chart_labels)
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
        "Do neighbourhood conditions improve nonlinear predictions?",
        fontsize=17,
        fontweight="bold",
        color=BASE.NAVY,
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
    property_cv_direct = property_rf_cv.loc[property_rf_cv["model"].eq(PROPERTY_RF_NAME)]
    property_cv_summary = pd.Series(
        {
            "mean_mae_gbp": property_cv_direct["mae_gbp"].mean(),
            "mean_rmse_gbp": property_cv_direct["rmse_gbp"].mean(),
            "mean_r2_price": property_cv_direct["r2_price"].mean(),
        }
    )
    validation_mae_improvement = percent_improvement(
        property_cv_summary["mean_mae_gbp"],
        cv_summary.loc[ENRICHED_RF_NAME, "mean_mae_gbp"],
    )
    validation_rmse_improvement = percent_improvement(
        property_cv_summary["mean_rmse_gbp"],
        cv_summary.loc[ENRICHED_RF_NAME, "mean_rmse_gbp"],
    )
    test_mae_improvement = percent_improvement(
        property_rf_direct["mae_gbp"], enriched_rf_direct["mae_gbp"]
    )
    test_rmse_improvement = percent_improvement(
        property_rf_direct["rmse_gbp"], enriched_rf_direct["rmse_gbp"]
    )
    top_feature_text = ", ".join(
        f"`{row.feature}` ({row.importance_percent:.1f}%)"
        for row in feature_importance.head(5).itertuples(index=False)
    )

    findings = f"""# Enriched random forest

## Purpose

This checkpoint tests the research question under a nonlinear specification. It adds monthly 2011-LSOA crime and claimant rate to the property-only random forest while keeping the cohort, target, chronological split, validation folds and tuning procedure unchanged.

## Comparable sample and leakage control

| Dataset | Period | Transactions |
|---|---|---:|
| Development | 2022–2023 | {len(development):,} |
| Test | 2024 | {len(test):,} |

The 2024 records were not used for hyperparameter selection. Crime and claimant variables are complete for every modelling transaction, so enrichment does not change the sample.

## Hyperparameter selection

Twelve settings were tested using mean validation MAE. The selected settings were {N_ESTIMATORS} trees, maximum depth **{selected_parameters['max_depth']}**, minimum leaf size **{selected_parameters['min_samples_leaf']}**, and maximum feature proportion **{selected_parameters['max_features']}**.

## Time-based validation

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only random forest | {money(property_cv_summary['mean_mae_gbp'])} | {money(property_cv_summary['mean_rmse_gbp'])} | {property_cv_summary['mean_r2_price']:.3f} |
| Enriched random forest | {money(cv_summary.loc[ENRICHED_RF_NAME, 'mean_mae_gbp'])} | {money(cv_summary.loc[ENRICHED_RF_NAME, 'mean_rmse_gbp'])} | {cv_summary.loc[ENRICHED_RF_NAME, 'mean_r2_price']:.3f} |

Adding the neighbourhood variables reduces mean validation MAE by **{validation_mae_improvement:.1f}%** and mean validation RMSE by **{validation_rmse_improvement:.1f}%**.

## Final 2024 comparison

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only random forest | {money(property_rf_direct['mae_gbp'])} | {money(property_rf_direct['rmse_gbp'])} | {property_rf_direct['r2_price']:.3f} | {property_rf_direct['r2_log_price']:.3f} |
| Enriched linear regression | {money(enriched_linear_direct['mae_gbp'])} | {money(enriched_linear_direct['rmse_gbp'])} | {enriched_linear_direct['r2_price']:.3f} | {enriched_linear_direct['r2_log_price']:.3f} |
| Enriched random forest | {money(enriched_rf_direct['mae_gbp'])} | {money(enriched_rf_direct['rmse_gbp'])} | {enriched_rf_direct['r2_price']:.3f} | {enriched_rf_direct['r2_log_price']:.3f} |
| Enriched random forest with smearing | {money(enriched_rf_smear['mae_gbp'])} | {money(enriched_rf_smear['rmse_gbp'])} | {enriched_rf_smear['r2_price']:.3f} | {enriched_rf_smear['r2_log_price']:.3f} |

Against the property-only random forest, the direct enriched model reduces 2024 MAE by **{test_mae_improvement:.1f}%** and RMSE by **{test_rmse_improvement:.1f}%**. Its price R² changes by **{enriched_rf_direct['r2_price'] - property_rf_direct['r2_price']:+.3f}**.

## Feature importance

The five largest impurity importances are {top_feature_text}. These values show how the fitted forest used variables for prediction; they do not demonstrate causal effects.

## Interpretation

This experiment supplies a nonlinear test of whether crime and claimant conditions contain predictive information beyond basic property and time variables. The fair comparison is the change from the property-only random forest under the same data split and tuning procedure.

## Limitations

- The Land Registry data lacks bedrooms, floor area, condition and detailed amenities.
- Monthly crime can include incidents occurring after an early-month sale, so a lagged sensitivity test is still advisable for a strictly prospective interpretation.
- The claimant-rate denominator is the fixed 2020 working-age population.
- Neighbourhood conditions may proxy for omitted aspects of location and housing stock.
- Impurity importance is descriptive and can favour variables with more potential split points.

## Next modelling experiment

Fit property-only and enriched gradient-boosting models using the same chronological design. This will provide a second nonlinear comparison before selecting the final model for the dashboard.
"""
    FINDINGS.write_text(findings, encoding="utf-8")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": ENRICHED_RF_NAME,
        "estimator": "sklearn.ensemble.RandomForestRegressor",
        "sklearn_version": sklearn.__version__,
        "cohort": "PPD Category A",
        "target": "log(price)",
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "neighbourhood_features": NEIGHBOURHOOD_FEATURES,
        "crime_transformation": "log1p(total_crime)",
        "claimant_rate_definition": "monthly claimant count / fixed 2020 working-age population * 100",
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
        "reference_files": {
            str(PROPERTY_RF_TEST.relative_to(PROJECT_ROOT)): sha256(PROPERTY_RF_TEST),
            str(PROPERTY_RF_CV.relative_to(PROJECT_ROOT)): sha256(PROPERTY_RF_CV),
            str(PROPERTY_RF_PREDICTIONS.relative_to(PROJECT_ROOT)): sha256(
                PROPERTY_RF_PREDICTIONS
            ),
            str(ENRICHED_LINEAR_TEST.relative_to(PROJECT_ROOT)): sha256(
                ENRICHED_LINEAR_TEST
            ),
        },
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
            "property_rf_transaction_order_match": bool(
                property_rf_predictions["transaction_id"].tolist()
                == predictions["transaction_id"].tolist()
            ),
        },
        "test_metrics": {
            "mae_gbp": float(enriched_rf_direct["mae_gbp"]),
            "rmse_gbp": float(enriched_rf_direct["rmse_gbp"]),
            "r2_price": float(enriched_rf_direct["r2_price"]),
            "r2_log_price": float(enriched_rf_direct["r2_log_price"]),
            "mae_improvement_vs_property_rf_percent": float(test_mae_improvement),
            "rmse_improvement_vs_property_rf_percent": float(test_rmse_improvement),
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
    print(f"Property-only RF MAE: {money(property_rf_direct['mae_gbp'])}")
    print(f"Enriched RF MAE: {money(enriched_rf_direct['mae_gbp'])}")
    print(f"Enriched RF RMSE: {money(enriched_rf_direct['rmse_gbp'])}")
    print(f"Enriched RF price R2: {enriched_rf_direct['r2_price']:.4f}")
    print(f"Saved outputs to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
