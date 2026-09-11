"""Pure prediction functions used by the dashboard and its tests."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "09_lagged_enriched_random_forest.joblib"
DEFAULT_POSTCODE_PATH = DASHBOARD_DIR / "data" / "bristol_postcode_lsoa11.csv"
DEFAULT_FEATURE_PATH = DASHBOARD_DIR / "data" / "lagged_neighbourhood_features_2024.csv"

SUPPORTED_PROPERTY_TYPES = {"D", "S", "T", "F"}
SUPPORTED_DURATIONS = {"F", "L"}
SUPPORTED_NEW_BUILD = {"Y", "N"}
EXPECTED_FEATURES = [
    "property_type",
    "duration",
    "old_new",
    "month_index",
    "month_sin",
    "month_cos",
    "log_lagged_total_crime",
    "lagged_claimant_rate_percent",
]


class DashboardInputError(ValueError):
    """Raised when a dashboard request is outside the validated scope."""


@dataclass(frozen=True)
class DashboardResources:
    model_bundle: dict[str, object]
    postcodes: pd.DataFrame
    features: pd.DataFrame


def normalize_postcode(value: str) -> str:
    key = re.sub(r"\s+", "", str(value).upper())
    if not re.fullmatch(r"BS\d{1,2}[A-Z]?\d[A-Z]{2}", key):
        raise DashboardInputError("Enter a complete Bristol postcode, for example BS1 1AD.")
    return key


def load_resources(
    model_path: Path = DEFAULT_MODEL_PATH,
    postcode_path: Path = DEFAULT_POSTCODE_PATH,
    feature_path: Path = DEFAULT_FEATURE_PATH,
) -> DashboardResources:
    model_bundle = joblib.load(model_path)
    if model_bundle.get("model_features") != EXPECTED_FEATURES:
        raise RuntimeError("The saved model feature definition does not match the dashboard.")

    postcodes = pd.read_csv(postcode_path, dtype={"postcode_key": str})
    features = pd.read_csv(feature_path, dtype={"sale_month": str})
    if postcodes["postcode_key"].duplicated().any():
        raise RuntimeError("The postcode lookup contains duplicate keys.")
    if features[["sale_month", "analysis_lsoa_code"]].duplicated().any():
        raise RuntimeError("The neighbourhood lookup contains duplicate keys.")
    return DashboardResources(model_bundle=model_bundle, postcodes=postcodes, features=features)


def build_model_input(
    property_type: str,
    duration: str,
    old_new: str,
    sale_month: str,
    lagged_total_crime: float,
    lagged_claimant_rate_percent: float,
) -> pd.DataFrame:
    if property_type not in SUPPORTED_PROPERTY_TYPES:
        raise DashboardInputError("Select one of the property types represented in the model.")
    if duration not in SUPPORTED_DURATIONS:
        raise DashboardInputError("Select freehold or leasehold.")
    if old_new not in SUPPORTED_NEW_BUILD:
        raise DashboardInputError("Select an existing property or new build.")

    period = pd.Period(sale_month, freq="M")
    if period.year != 2024:
        raise DashboardInputError("This validated dashboard supports January to December 2024 only.")
    if lagged_total_crime < 0 or lagged_claimant_rate_percent < 0:
        raise DashboardInputError("Neighbourhood inputs cannot be negative.")

    month_number = period.month
    values = {
        "property_type": property_type,
        "duration": duration,
        "old_new": old_new,
        "month_index": float((period.year - 2022) * 12 + month_number - 1),
        "month_sin": float(np.sin(2 * np.pi * month_number / 12)),
        "month_cos": float(np.cos(2 * np.pi * month_number / 12)),
        "log_lagged_total_crime": float(np.log1p(lagged_total_crime)),
        "lagged_claimant_rate_percent": float(lagged_claimant_rate_percent),
    }
    return pd.DataFrame([values], columns=EXPECTED_FEATURES)


def predict_property(
    resources: DashboardResources,
    postcode: str,
    property_type: str,
    duration: str,
    old_new: str,
    sale_month: str,
) -> dict[str, object]:
    key = normalize_postcode(postcode)
    postcode_match = resources.postcodes.loc[resources.postcodes["postcode_key"].eq(key)]
    if postcode_match.empty:
        raise DashboardInputError(
            "That postcode is not mapped to the Bristol local-authority area in the project lookup."
        )
    postcode_record = postcode_match.iloc[0]

    feature_match = resources.features.loc[
        resources.features["sale_month"].eq(sale_month)
        & resources.features["analysis_lsoa_code"].eq(postcode_record["analysis_lsoa_code"])
    ]
    if feature_match.empty:
        raise DashboardInputError("Neighbourhood data is unavailable for that postcode and month.")
    feature_record = feature_match.iloc[0]

    model_frame = build_model_input(
        property_type=property_type,
        duration=duration,
        old_new=old_new,
        sale_month=sale_month,
        lagged_total_crime=float(feature_record["lagged_total_crime"]),
        lagged_claimant_rate_percent=float(feature_record["lagged_claimant_rate_percent"]),
    )
    predicted_log_price = float(resources.model_bundle["pipeline"].predict(model_frame)[0])
    predicted_price = float(np.exp(predicted_log_price))
    bias_adjusted_price = predicted_price * float(resources.model_bundle["smearing_factor"])

    return {
        "predicted_price": predicted_price,
        "bias_adjusted_price": bias_adjusted_price,
        "predicted_log_price": predicted_log_price,
        "postcode": postcode_record["postcode"],
        "postcode_status": postcode_record["postcode_status"],
        "analysis_lsoa_code": postcode_record["analysis_lsoa_code"],
        "analysis_lsoa_name": feature_record["analysis_lsoa_name"],
        "sale_month": sale_month,
        "neighbourhood_source_month": feature_record["neighbourhood_source_month"],
        "lagged_total_crime": int(feature_record["lagged_total_crime"]),
        "lagged_claimant_rate_percent": float(feature_record["lagged_claimant_rate_percent"]),
        "model_input": model_frame,
    }
