"""Streamlit interface for the Bristol house-price artefact."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import pandas as pd
import streamlit as st

from prediction import DashboardInputError, load_resources, predict_property


DASHBOARD_DIR = Path(__file__).resolve().parent
MODEL_CARD_PATH = DASHBOARD_DIR / "model_card.json"
METRICS_PATH = DASHBOARD_DIR / "data" / "model_performance_2024.csv"

PROPERTY_TYPES = {
    "Terraced": "T",
    "Semi-detached": "S",
    "Detached": "D",
    "Flat or maisonette": "F",
}
TENURES = {"Freehold": "F", "Leasehold": "L"}
BUILD_STATUS = {"Existing property": "N", "New build": "Y"}
SALE_MONTHS = pd.period_range("2024-01", "2024-12", freq="M")
SALE_MONTH_LABELS = {period.strftime("%B %Y"): str(period) for period in SALE_MONTHS}


@st.cache_resource
def cached_resources():
    return load_resources()


@st.cache_data
def cached_reference_data():
    card = json.loads(MODEL_CARD_PATH.read_text(encoding="utf-8"))
    metrics = pd.read_csv(METRICS_PATH)
    return card, metrics


st.set_page_config(page_title="Bristol House Price Explorer", layout="wide")

st.title("Bristol House Price Explorer")
st.write(
    "Explore an estimated 2024 Bristol sale price using property details and the "
    "previous month's neighbourhood crime and claimant conditions."
)

resources = cached_resources()
model_card, metrics = cached_reference_data()

prediction_tab, evidence_tab, method_tab = st.tabs(
    ["Price estimate", "Model evidence", "Method and limitations"]
)

with prediction_tab:
    st.subheader("Property details")
    with st.form("prediction_form"):
        left, right = st.columns(2)
        with left:
            postcode = st.text_input("Bristol postcode", value="BS1 1AD")
            property_label = st.selectbox("Property type", list(PROPERTY_TYPES), index=0)
            tenure_label = st.selectbox("Tenure", list(TENURES), index=0)
        with right:
            build_label = st.selectbox("Build status", list(BUILD_STATUS), index=0)
            month_label = st.selectbox("Estimated sale month", list(SALE_MONTH_LABELS), index=0)
            st.write("The tool uses the preceding month's neighbourhood observations.")
        submitted = st.form_submit_button("Estimate price", width="stretch")

    if submitted:
        try:
            result = predict_property(
                resources=resources,
                postcode=postcode,
                property_type=PROPERTY_TYPES[property_label],
                duration=TENURES[tenure_label],
                old_new=BUILD_STATUS[build_label],
                sale_month=SALE_MONTH_LABELS[month_label],
            )
        except DashboardInputError as exc:
            st.markdown(f"**The estimate could not be produced:** {exc}")
        else:
            st.subheader("Estimated sale price")
            estimate, typical_error, fit = st.columns(3)
            estimate.metric("Model estimate", f"£{result['predicted_price']:,.0f}")
            typical_error.metric("2024 test MAE", f"£{model_card['test_mae_gbp']:,.0f}")
            fit.metric("2024 test R²", f"{model_card['test_r2_price']:.3f}")
            st.caption(
                "The test MAE is a model-wide typical absolute error, not a confidence interval "
                "for this individual property."
            )

            st.subheader("Neighbourhood information used")
            context = pd.DataFrame(
                {
                    "Item": [
                        "Postcode",
                        "2011 LSOA",
                        "Neighbourhood data month",
                        "Total recorded crime",
                        "Claimant rate",
                    ],
                    "Value": [
                        result["postcode"],
                        f"{result['analysis_lsoa_name']} ({result['analysis_lsoa_code']})",
                        pd.Period(result["neighbourhood_source_month"], freq="M").strftime("%B %Y"),
                        f"{result['lagged_total_crime']:,}",
                        f"{result['lagged_claimant_rate_percent']:.2f}%",
                    ],
                }
            )
            st.dataframe(context, hide_index=True, width="stretch")

            if result["postcode_status"] == "Terminated":
                st.write(
                    "The supplied postcode is marked as terminated in the May 2026 postcode directory, "
                    "but its historical 2011 neighbourhood mapping remains available."
                )

with evidence_tab:
    st.subheader("Performance on unseen 2024 transactions")
    display = metrics.rename(
        columns={
            "model": "Model",
            "mae_gbp": "MAE (£)",
            "rmse_gbp": "RMSE (£)",
            "r2_price": "R²",
            "median_absolute_percentage_error_percent": "Median absolute percentage error",
            "mae_improvement_vs_property_percent": "MAE improvement",
        }
    )[
        ["Model", "MAE (£)", "RMSE (£)", "R²", "Median absolute percentage error", "MAE improvement"]
    ].copy()
    display["MAE (£)"] = display["MAE (£)"].map(lambda value: f"£{value:,.0f}")
    display["RMSE (£)"] = display["RMSE (£)"].map(lambda value: f"£{value:,.0f}")
    display["R²"] = display["R²"].map(lambda value: f"{value:.3f}")
    display["Median absolute percentage error"] = display[
        "Median absolute percentage error"
    ].map(lambda value: f"{value:.1f}%")
    display["MAE improvement"] = display["MAE improvement"].map(lambda value: f"{value:.1f}%")
    st.dataframe(display, hide_index=True, width="stretch")

    chart_data = metrics.set_index("model")[["mae_gbp"]].rename(columns={"mae_gbp": "MAE (£)"})
    st.bar_chart(chart_data, color="#176B87", horizontal=True)
    st.write(
        "Adding the previous month's neighbourhood variables reduced 2024 MAE by "
        f"{metrics.loc[metrics['model'].eq('Lagged enriched random forest'), 'mae_improvement_vs_property_percent'].iloc[0]:.1f}% "
        "relative to the property-only random forest on the same transactions."
    )

with method_tab:
    st.subheader("How the estimate is produced")
    st.write(
        "The postcode is converted to a 2011 Lower Layer Super Output Area. The app then obtains "
        "crime and claimant-rate values from one calendar month before the selected sale month. "
        "Those values are combined with property type, tenure, new-build status and time features."
    )
    st.write(
        "The selected model is a random forest trained on Category A Bristol Land Registry sales "
        "from February 2022 to December 2023 and tested on 5,885 transactions from 2024."
    )

    st.subheader("Important limitations")
    for limitation in model_card["limitations"]:
        st.write(f"- {limitation}")

    st.write(
        "This tool is an academic research artefact. It is not an estate-agent valuation, mortgage "
        "valuation or recommendation to buy or sell a property."
    )
