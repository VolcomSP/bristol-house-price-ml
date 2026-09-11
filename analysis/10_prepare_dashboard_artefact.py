"""Prepare the validated data bundle used by the Streamlit artefact.

The dashboard is deliberately restricted to historical 2024 estimates.  A
postcode supplies the 2011 LSOA and the requested sale month supplies the
previous month's crime and claimant-rate observations.  The selected lagged
random-forest model is never refitted by this script.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONSPD = PROJECT_ROOT / "verified_sources" / "ONSPD_MAY_2026_UK_BS_complete.csv"
CRIME = PROJECT_ROOT / "outputs" / "candidate" / "crime_features_2022_2024_lsoa11_candidate.csv"
CLAIMANT = PROJECT_ROOT / "outputs" / "candidate" / "claimant_features_2022_2024_lsoa11_candidate.csv"
TEST_METRICS = PROJECT_ROOT / "outputs" / "models" / "09_lagged_sensitivity_test_metrics.csv"
MODEL = PROJECT_ROOT / "models" / "09_lagged_enriched_random_forest.joblib"

DASHBOARD = PROJECT_ROOT / "dashboard"
DATA_DIR = DASHBOARD / "data"
POSTCODE_OUTPUT = DATA_DIR / "bristol_postcode_lsoa11.csv"
FEATURE_OUTPUT = DATA_DIR / "lagged_neighbourhood_features_2024.csv"
METRICS_OUTPUT = DATA_DIR / "model_performance_2024.csv"
CASES_OUTPUT = DATA_DIR / "dashboard_validation_cases.csv"
MODEL_CARD = DASHBOARD / "model_card.json"
VALIDATION = DASHBOARD / "dashboard_validation.json"
APP = DASHBOARD / "app.py"
PREDICTION_MODULE = DASHBOARD / "prediction.py"
REQUIREMENTS = DASHBOARD / "requirements.txt"
README = DASHBOARD / "README.md"
EVALUATION = DASHBOARD / "ARTEFACT_EVALUATION.md"
TEST_MODULE = PROJECT_ROOT / "tests" / "test_dashboard_prediction.py"
STREAMLIT_CONFIG = PROJECT_ROOT / ".streamlit" / "config.toml"

BRISTOL_LAD = "E06000023"
SUPPORTED_MONTHS = pd.period_range("2024-01", "2024-12", freq="M")
PROPERTY_FEATURES = [
    "property_type",
    "duration",
    "old_new",
    "month_index",
    "month_sin",
    "month_cos",
    "log_lagged_total_crime",
    "lagged_claimant_rate_percent",
]
VALIDATION_CASE_INPUTS = [
    {
        "case_id": "case_01",
        "postcode": "BS58DA",
        "sale_month": "2024-01",
        "property_type": "D",
        "duration": "F",
        "old_new": "N",
        "analysis_lsoa_code": "E01014677",
        "expected_prediction_gbp": 420_983.499939,
    },
    {
        "case_id": "case_02",
        "postcode": "BS130QW",
        "sale_month": "2024-01",
        "property_type": "S",
        "duration": "F",
        "old_new": "N",
        "analysis_lsoa_code": "E01032515",
        "expected_prediction_gbp": 276_606.136521,
    },
    {
        "case_id": "case_03",
        "postcode": "BS67PL",
        "sale_month": "2024-01",
        "property_type": "T",
        "duration": "F",
        "old_new": "N",
        "analysis_lsoa_code": "E01014619",
        "expected_prediction_gbp": 560_622.843068,
    },
    {
        "case_id": "case_04",
        "postcode": "BS20JB",
        "sale_month": "2024-01",
        "property_type": "F",
        "duration": "L",
        "old_new": "N",
        "analysis_lsoa_code": "E01014540",
        "expected_prediction_gbp": 289_818.502988,
    },
    {
        "case_id": "case_05",
        "postcode": "BS79YG",
        "sale_month": "2024-01",
        "property_type": "S",
        "duration": "F",
        "old_new": "Y",
        "analysis_lsoa_code": "E01014660",
        "expected_prediction_gbp": 316_904.144136,
    },
    {
        "case_id": "case_06",
        "postcode": "BS31GH",
        "sale_month": "2024-12",
        "property_type": "F",
        "duration": "L",
        "old_new": "N",
        "analysis_lsoa_code": "E01033359",
        "expected_prediction_gbp": 244_815.380677,
    },
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def postcode_key(value: str) -> str:
    return re.sub(r"\s+", "", str(value).upper())


def prepare_postcodes(valid_lsoas: set[str]) -> pd.DataFrame:
    postcodes = pd.read_csv(
        ONSPD,
        usecols=["pcds", "dointr", "doterm", "lad25cd", "lsoa11cd"],
        low_memory=False,
    )
    postcodes = postcodes.loc[
        postcodes["lad25cd"].eq(BRISTOL_LAD)
        & postcodes["lsoa11cd"].isin(valid_lsoas)
    ].copy()
    postcodes["postcode_key"] = postcodes["pcds"].map(postcode_key)
    postcodes["postcode"] = postcodes["postcode_key"].str[:-3] + " " + postcodes["postcode_key"].str[-3:]
    postcodes["postcode_status"] = np.where(postcodes["doterm"].isna(), "Active", "Terminated")
    postcodes = postcodes.rename(
        columns={
            "lsoa11cd": "analysis_lsoa_code",
            "dointr": "postcode_introduction_yyyymm",
            "doterm": "postcode_termination_yyyymm",
        }
    )[
        [
            "postcode_key",
            "postcode",
            "analysis_lsoa_code",
            "postcode_status",
            "postcode_introduction_yyyymm",
            "postcode_termination_yyyymm",
        ]
    ]
    postcodes["postcode_introduction_yyyymm"] = pd.to_numeric(
        postcodes["postcode_introduction_yyyymm"], errors="coerce"
    ).astype("Int64")
    postcodes["postcode_termination_yyyymm"] = pd.to_numeric(
        postcodes["postcode_termination_yyyymm"], errors="coerce"
    ).astype("Int64")
    postcodes = postcodes.sort_values("postcode_key").reset_index(drop=True)

    assert len(postcodes) == 17_844
    assert postcodes["postcode_key"].duplicated().sum() == 0
    assert postcodes["analysis_lsoa_code"].nunique() == 263
    assert postcodes.isna().sum()[["postcode_key", "postcode", "analysis_lsoa_code"]].sum() == 0
    return postcodes


def prepare_lagged_features() -> pd.DataFrame:
    crime = pd.read_csv(CRIME)
    claimant = pd.read_csv(CLAIMANT)
    crime["month"] = pd.PeriodIndex(crime["month"], freq="M")
    claimant["month"] = pd.PeriodIndex(claimant["month"], freq="M")

    features = crime[["lsoa_code", "month", "total_crime"]].merge(
        claimant[["lsoa11cd", "lsoa11_name", "month", "claimant_rate_percent"]],
        left_on=["lsoa_code", "month"],
        right_on=["lsoa11cd", "month"],
        how="inner",
        validate="one_to_one",
    )
    features["sale_month"] = features["month"] + 1
    features = features.loc[features["sale_month"].isin(SUPPORTED_MONTHS)].copy()
    features = features.rename(
        columns={
            "lsoa_code": "analysis_lsoa_code",
            "lsoa11_name": "analysis_lsoa_name",
            "month": "neighbourhood_source_month",
            "total_crime": "lagged_total_crime",
            "claimant_rate_percent": "lagged_claimant_rate_percent",
        }
    ).drop(columns="lsoa11cd")
    features["sale_month"] = features["sale_month"].astype(str)
    features["neighbourhood_source_month"] = features["neighbourhood_source_month"].astype(str)
    features = features[
        [
            "sale_month",
            "neighbourhood_source_month",
            "analysis_lsoa_code",
            "analysis_lsoa_name",
            "lagged_total_crime",
            "lagged_claimant_rate_percent",
        ]
    ].sort_values(["sale_month", "analysis_lsoa_code"]).reset_index(drop=True)

    assert len(features) == 12 * 263
    assert features[["sale_month", "analysis_lsoa_code"]].duplicated().sum() == 0
    assert features["analysis_lsoa_code"].nunique() == 263
    assert features.isna().sum().sum() == 0
    expected = (pd.PeriodIndex(features["sale_month"], freq="M") - 1).astype(str)
    assert (features["neighbourhood_source_month"].to_numpy() == expected.to_numpy()).all()
    return features


def prepare_metrics() -> pd.DataFrame:
    metrics = pd.read_csv(TEST_METRICS)
    selected_names = {
        "Property-only random forest, common cohort": "Property-only random forest",
        "Lagged enriched random forest": "Lagged enriched random forest",
    }
    metrics = metrics.loc[metrics["model"].isin(selected_names)].copy()
    metrics["model"] = metrics["model"].map(selected_names)
    metrics = metrics[
        [
            "model",
            "rows",
            "mae_gbp",
            "rmse_gbp",
            "median_absolute_percentage_error_percent",
            "r2_price",
        ]
    ].reset_index(drop=True)
    property_mae = float(metrics.loc[metrics["model"].eq("Property-only random forest"), "mae_gbp"].iloc[0])
    metrics["mae_improvement_vs_property_percent"] = (
        (property_mae - metrics["mae_gbp"]) / property_mae * 100
    )
    assert len(metrics) == 2
    return metrics


def model_input(record: pd.Series, feature: pd.Series) -> pd.DataFrame:
    period = pd.Period(record["sale_month"], freq="M")
    month_number = period.month
    return pd.DataFrame(
        [
            {
                "property_type": record["property_type"],
                "duration": record["duration"],
                "old_new": record["old_new"],
                "month_index": float((period.year - 2022) * 12 + month_number - 1),
                "month_sin": float(np.sin(2 * np.pi * month_number / 12)),
                "month_cos": float(np.cos(2 * np.pi * month_number / 12)),
                "log_lagged_total_crime": float(np.log1p(feature["lagged_total_crime"])),
                "lagged_claimant_rate_percent": float(feature["lagged_claimant_rate_percent"]),
            }
        ],
        columns=PROPERTY_FEATURES,
    )


def prepare_validation_cases(
    postcodes: pd.DataFrame, features: pd.DataFrame, model_bundle: dict[str, object]
) -> pd.DataFrame:
    postcode_index = postcodes.set_index("postcode_key")
    feature_index = features.set_index(["sale_month", "analysis_lsoa_code"])
    pipeline = model_bundle["pipeline"]
    rows: list[dict[str, object]] = []
    for values in VALIDATION_CASE_INPUTS:
        record = pd.Series(values)
        postcode_record = postcode_index.loc[postcode_key(record["postcode"])]
        assert postcode_record["analysis_lsoa_code"] == record["analysis_lsoa_code"]
        feature = feature_index.loc[(record["sale_month"], record["analysis_lsoa_code"])]
        prediction = float(np.exp(pipeline.predict(model_input(record, feature))[0]))
        expected = float(record["expected_prediction_gbp"])
        assert abs(prediction - expected) <= 0.01
        rows.append(dict(values))
    cases = pd.DataFrame(rows).drop_duplicates("case_id").reset_index(drop=True)
    assert len(cases) >= 5
    return cases


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    model_bundle = joblib.load(MODEL)
    assert model_bundle["model_features"] == PROPERTY_FEATURES
    assert model_bundle["neighbourhood_timing"] == "one calendar month before sale month"

    features = prepare_lagged_features()
    postcodes = prepare_postcodes(set(features["analysis_lsoa_code"]))
    metrics = prepare_metrics()
    cases = prepare_validation_cases(postcodes, features, model_bundle)

    postcodes.to_csv(POSTCODE_OUTPUT, index=False)
    features.to_csv(FEATURE_OUTPUT, index=False, float_format="%.8f")
    metrics.to_csv(METRICS_OUTPUT, index=False, float_format="%.6f")
    cases.to_csv(CASES_OUTPUT, index=False, float_format="%.6f")

    lagged_metrics = metrics.loc[metrics["model"].eq("Lagged enriched random forest")].iloc[0]
    model_card = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "project": "Predicting house prices in Bristol using machine learning and socio-economic data",
        "selected_model": "Lagged enriched random forest",
        "estimator": "sklearn.ensemble.RandomForestRegressor",
        "target": "log(price)",
        "inverse_transformation": "exp(predicted log price); no smearing correction for the headline estimate",
        "training_period": model_bundle["training_period"],
        "dashboard_supported_period": "2024-01 to 2024-12",
        "dashboard_scope": "Historical Bristol sale-price exploration; not a current professional valuation",
        "user_inputs": ["postcode", "property type", "tenure", "new-build status", "sale month"],
        "derived_inputs": ["2011 LSOA", "previous-month total crime", "previous-month claimant rate", "time features"],
        "model_features": model_bundle["model_features"],
        "model_file": str(MODEL.relative_to(PROJECT_ROOT)),
        "model_sha256": sha256(MODEL),
        "test_rows": int(lagged_metrics["rows"]),
        "test_mae_gbp": float(lagged_metrics["mae_gbp"]),
        "test_rmse_gbp": float(lagged_metrics["rmse_gbp"]),
        "test_r2_price": float(lagged_metrics["r2_price"]),
        "test_median_absolute_percentage_error_percent": float(
            lagged_metrics["median_absolute_percentage_error_percent"]
        ),
        "limitations": [
            "The Land Registry data does not include bedrooms, floor area, condition or exact property amenities.",
            "Crime and claimant inputs describe the surrounding LSOA, not the individual property.",
            "The dashboard supports the independently tested 2024 period and is not a live 2026 valuation service.",
            "The results are predictive associations and do not establish causal effects.",
        ],
    }
    MODEL_CARD.write_text(json.dumps(model_card, indent=2) + "\n", encoding="utf-8")

    outputs = [POSTCODE_OUTPUT, FEATURE_OUTPUT, METRICS_OUTPUT, CASES_OUTPUT, MODEL_CARD]
    artefact_files = [
        Path(__file__),
        APP,
        PREDICTION_MODULE,
        REQUIREMENTS,
        README,
        EVALUATION,
        TEST_MODULE,
        STREAMLIT_CONFIG,
        MODEL,
        *outputs,
    ]
    validation = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checks": {
            "postcode_rows": len(postcodes),
            "active_postcodes": int(postcodes["postcode_status"].eq("Active").sum()),
            "terminated_postcodes": int(postcodes["postcode_status"].eq("Terminated").sum()),
            "duplicate_postcode_keys": int(postcodes["postcode_key"].duplicated().sum()),
            "lsoa_count": int(features["analysis_lsoa_code"].nunique()),
            "supported_sale_months": int(features["sale_month"].nunique()),
            "lagged_feature_rows": len(features),
            "missing_lagged_feature_values": int(features.isna().sum().sum()),
            "validation_case_rows": len(cases),
            "model_feature_order_matches": model_bundle["model_features"] == PROPERTY_FEATURES,
        },
        "sources": {
            str(path.relative_to(PROJECT_ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [ONSPD, CRIME, CLAIMANT, TEST_METRICS, MODEL]
        },
        "outputs": {
            str(path.relative_to(DASHBOARD)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
        },
        "artefact_files": {
            str(path.relative_to(PROJECT_ROOT)): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in artefact_files
        },
    }
    VALIDATION.write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")

    print(f"Prepared {len(postcodes):,} Bristol postcode mappings")
    print(f"Prepared {len(features):,} lagged feature rows across 12 sale months")
    print(f"Validated {len(cases):,} saved-model prediction cases")
    print(f"Dashboard model: {model_card['selected_model']}")


if __name__ == "__main__":
    main()
