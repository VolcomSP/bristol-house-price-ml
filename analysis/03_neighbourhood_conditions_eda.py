"""EDA checkpoint 3: crime, claimant conditions and Bristol sale prices."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROPERTY_DATASET = PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2022_2024.csv"
CRIME_DATASET = (
    PROJECT_ROOT / "outputs" / "candidate" / "crime_features_2022_2024_lsoa11_candidate.csv"
)
CLAIMANT_DATASET = (
    PROJECT_ROOT / "outputs" / "candidate" / "claimant_features_2022_2024_lsoa11_candidate.csv"
)
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "eda"

DISTRIBUTION_SUMMARY = OUTPUT_FOLDER / "03_neighbourhood_feature_distribution_summary.csv"
CORRELATION_SUMMARY = OUTPUT_FOLDER / "03_neighbourhood_correlation_summary.csv"
TYPE_CORRELATIONS = OUTPUT_FOLDER / "03_property_type_neighbourhood_correlations.csv"
LSOA_SUMMARY = OUTPUT_FOLDER / "03_lsoa_neighbourhood_summary.csv"
QUINTILE_SUMMARY = OUTPUT_FOLDER / "03_neighbourhood_quintile_summary.csv"
MONTHLY_SUMMARY = OUTPUT_FOLDER / "03_monthly_neighbourhood_summary.csv"
DISTRIBUTION_CHART = OUTPUT_FOLDER / "03_neighbourhood_feature_distributions.png"
RELATIONSHIP_CHART = OUTPUT_FOLDER / "03_lsoa_price_relationships.png"
QUINTILE_CHART = OUTPUT_FOLDER / "03_neighbourhood_quintile_prices.png"
MONTHLY_CHART = OUTPUT_FOLDER / "03_monthly_neighbourhood_features.png"
FINDINGS = OUTPUT_FOLDER / "03_neighbourhood_conditions_findings.md"
MANIFEST = OUTPUT_FOLDER / "03_neighbourhood_conditions_eda.validation.json"

FEATURES = [
    "total_crime",
    "log1p_total_crime",
    "crime_per_1000_working_age_proxy",
    "claimant_rate_percent",
]

NAVY = "#172554"
BLUE = "#2563EB"
TEAL = "#0F766E"
ORANGE = "#D97706"
PURPLE = "#7C3AED"
GREY = "#64748B"
LIGHT_GREY = "#E2E8F0"


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


def describe_features(data: pd.DataFrame, analysis_level: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for feature in FEATURES:
        values = data[feature]
        quantiles = values.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        rows.append(
            {
                "analysis_level": analysis_level,
                "feature": feature,
                "observations": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "mean": values.mean(),
                "standard_deviation": values.std(),
                "minimum": values.min(),
                "p01": quantiles.loc[0.01],
                "p05": quantiles.loc[0.05],
                "p25": quantiles.loc[0.25],
                "median": quantiles.loc[0.50],
                "p75": quantiles.loc[0.75],
                "p95": quantiles.loc[0.95],
                "p99": quantiles.loc[0.99],
                "maximum": values.max(),
                "skewness": values.skew(),
            }
        )
    return pd.DataFrame(rows)


def correlation_rows(
    data: pd.DataFrame,
    analysis_level: str,
    target: str,
    features: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for feature in features:
        pair = data[[feature, target]].dropna()
        rows.append(
            {
                "analysis_level": analysis_level,
                "x_variable": feature,
                "y_variable": target,
                "observations": len(pair),
                "spearman_correlation": pair[feature].corr(pair[target], method="spearman"),
                "pearson_correlation": pair[feature].corr(pair[target], method="pearson"),
            }
        )
    return rows


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    source = pd.read_csv(PROPERTY_DATASET, low_memory=False)
    transactions = source.loc[source["ppd_category"].eq("A")].copy()
    transactions["log_price"] = np.log(transactions["price"])
    transactions["log1p_total_crime"] = np.log1p(transactions["total_crime"])
    transactions["crime_per_1000_working_age_proxy"] = (
        transactions["total_crime"]
        / transactions["working_age_population_2020"]
        * 1_000
    )

    crime = pd.read_csv(CRIME_DATASET, low_memory=False)
    claimant = pd.read_csv(CLAIMANT_DATASET, low_memory=False)
    panel = crime.merge(
        claimant,
        left_on=["lsoa_code", "month"],
        right_on=["lsoa11cd", "month"],
        how="inner",
        validate="one_to_one",
    )
    panel["month_date"] = pd.to_datetime(panel["month"] + "-01")
    panel["log1p_total_crime"] = np.log1p(panel["total_crime"])
    panel["crime_per_1000_working_age_proxy"] = (
        panel["total_crime"] / panel["working_age_population_2020"] * 1_000
    )

    assert len(source) == 21_968
    assert len(transactions) == 18_670
    assert len(panel) == 9_468
    assert panel["lsoa_code"].nunique() == 263
    assert panel["month"].nunique() == 36
    assert panel.duplicated(["lsoa_code", "month"]).sum() == 0
    assert panel[FEATURES].isna().sum().sum() == 0
    assert transactions[FEATURES + ["price", "log_price"]].isna().sum().sum() == 0
    assert int(panel["total_crime"].sum()) == 190_613

    distribution = pd.concat(
        [
            describe_features(panel, "LSOA-month panel"),
            describe_features(transactions, "Category A transactions"),
        ],
        ignore_index=True,
    )
    distribution.to_csv(DISTRIBUTION_SUMMARY, index=False, float_format="%.6f")

    monthly_features = (
        panel.groupby(["month", "month_date"], as_index=False)
        .agg(
            bristol_total_crime=("total_crime", "sum"),
            mean_lsoa_crime=("total_crime", "mean"),
            median_lsoa_crime=("total_crime", "median"),
            mean_crime_per_1000_working_age_proxy=(
                "crime_per_1000_working_age_proxy",
                "mean",
            ),
            mean_claimant_rate_percent=("claimant_rate_percent", "mean"),
            median_claimant_rate_percent=("claimant_rate_percent", "median"),
        )
        .sort_values("month_date")
    )
    monthly_features.drop(columns="month_date").to_csv(
        MONTHLY_SUMMARY, index=False, float_format="%.6f"
    )

    area_features = (
        panel.groupby("lsoa_code", as_index=False)
        .agg(
            mean_monthly_crime=("total_crime", "mean"),
            median_monthly_crime=("total_crime", "median"),
            mean_log1p_monthly_crime=("log1p_total_crime", "mean"),
            mean_crime_per_1000_working_age_proxy=(
                "crime_per_1000_working_age_proxy",
                "mean",
            ),
            mean_claimant_rate_percent=("claimant_rate_percent", "mean"),
            median_claimant_rate_percent=("claimant_rate_percent", "median"),
            working_age_population_2020=("working_age_population_2020", "first"),
        )
    )
    area_prices = (
        transactions.groupby("analysis_lsoa_code", as_index=False)
        .agg(
            transaction_count=("transaction_id", "size"),
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            median_log_price=("log_price", "median"),
            flat_share=("property_type", lambda values: values.eq("F").mean()),
            terraced_share=("property_type", lambda values: values.eq("T").mean()),
            semi_detached_share=("property_type", lambda values: values.eq("S").mean()),
            detached_share=("property_type", lambda values: values.eq("D").mean()),
        )
    )
    lsoa = area_prices.merge(
        area_features,
        left_on="analysis_lsoa_code",
        right_on="lsoa_code",
        how="inner",
        validate="one_to_one",
    )
    lsoa["low_transaction_count_flag"] = lsoa["transaction_count"].lt(20)
    lsoa = lsoa.sort_values("analysis_lsoa_code").reset_index(drop=True)
    lsoa.to_csv(LSOA_SUMMARY, index=False, float_format="%.6f")

    assert len(lsoa) == 263
    assert int(lsoa["transaction_count"].sum()) == len(transactions)

    transaction_features = [
        "total_crime",
        "log1p_total_crime",
        "crime_per_1000_working_age_proxy",
        "claimant_rate_percent",
    ]
    lsoa_features = [
        "mean_monthly_crime",
        "mean_log1p_monthly_crime",
        "mean_crime_per_1000_working_age_proxy",
        "mean_claimant_rate_percent",
    ]
    correlations = correlation_rows(
        transactions,
        "Category A transactions",
        "log_price",
        transaction_features,
    )
    correlations.extend(
        correlation_rows(
            lsoa,
            "LSOA aggregate",
            "median_log_price",
            lsoa_features,
        )
    )
    for x_variable, y_variable in [
        ("total_crime", "claimant_rate_percent"),
        ("crime_per_1000_working_age_proxy", "claimant_rate_percent"),
        ("total_crime", "crime_per_1000_working_age_proxy"),
    ]:
        pair = panel[[x_variable, y_variable]].dropna()
        correlations.append(
            {
                "analysis_level": "LSOA-month feature relationship",
                "x_variable": x_variable,
                "y_variable": y_variable,
                "observations": len(pair),
                "spearman_correlation": pair[x_variable].corr(
                    pair[y_variable], method="spearman"
                ),
                "pearson_correlation": pair[x_variable].corr(
                    pair[y_variable], method="pearson"
                ),
            }
        )
    correlation_table = pd.DataFrame(correlations)
    correlation_table.to_csv(CORRELATION_SUMMARY, index=False, float_format="%.6f")

    type_rows: list[dict[str, object]] = []
    for property_type, group in transactions.groupby("property_type_label"):
        for feature in transaction_features:
            type_rows.append(
                {
                    "property_type_label": property_type,
                    "feature": feature,
                    "transactions": len(group),
                    "spearman_with_log_price": group[feature].corr(
                        group["log_price"], method="spearman"
                    ),
                    "pearson_with_log_price": group[feature].corr(
                        group["log_price"], method="pearson"
                    ),
                }
            )
    type_correlations = pd.DataFrame(type_rows).sort_values(
        ["property_type_label", "feature"]
    )
    type_correlations.to_csv(TYPE_CORRELATIONS, index=False, float_format="%.6f")

    quintile_definitions = [
        ("Mean monthly crime", "mean_monthly_crime"),
        (
            "Mean crime per 1,000 working-age residents (proxy)",
            "mean_crime_per_1000_working_age_proxy",
        ),
        ("Mean claimant rate", "mean_claimant_rate_percent"),
    ]
    quintile_labels = ["Q1 low", "Q2", "Q3", "Q4", "Q5 high"]
    quintile_rows: list[pd.DataFrame] = []
    for display_name, feature in quintile_definitions:
        working = lsoa.copy()
        working["quintile"] = pd.qcut(working[feature], 5, labels=quintile_labels)
        summary = (
            working.groupby("quintile", observed=True, as_index=False)
            .agg(
                lsoa_count=("analysis_lsoa_code", "size"),
                transaction_count=("transaction_count", "sum"),
                mean_feature_value=(feature, "mean"),
                median_feature_value=(feature, "median"),
                median_of_lsoa_median_prices=("median_price", "median"),
                mean_of_lsoa_median_prices=("median_price", "mean"),
            )
        )
        summary.insert(0, "feature", feature)
        summary.insert(0, "feature_label", display_name)
        quintile_rows.append(summary)
    quintiles = pd.concat(quintile_rows, ignore_index=True)
    quintiles.to_csv(QUINTILE_SUMMARY, index=False, float_format="%.6f")

    set_chart_style()
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.1))
    raw_limit = panel["total_crime"].quantile(0.99)
    axes[0].hist(
        panel.loc[panel["total_crime"].le(raw_limit), "total_crime"],
        bins=45,
        color=BLUE,
        edgecolor="white",
    )
    axes[0].set_title("Monthly crime count")
    axes[0].set_xlabel("Crimes per LSOA-month")
    axes[0].set_ylabel("LSOA-month observations")
    axes[0].grid(axis="y")
    axes[0].text(
        0,
        -0.22,
        f"Displayed to p99 ({raw_limit:.0f}); no observations removed.",
        transform=axes[0].transAxes,
        fontsize=8.5,
        color=GREY,
    )

    axes[1].hist(panel["log1p_total_crime"], bins=45, color=TEAL, edgecolor="white")
    axes[1].set_title("Log-transformed crime count")
    axes[1].set_xlabel("log(1 + monthly crime)")
    axes[1].set_ylabel("LSOA-month observations")
    axes[1].grid(axis="y")

    axes[2].hist(panel["claimant_rate_percent"], bins=45, color=ORANGE, edgecolor="white")
    axes[2].set_title("Claimant rate")
    axes[2].set_xlabel("Claimants as % of working-age population")
    axes[2].set_ylabel("LSOA-month observations")
    axes[2].grid(axis="y")

    fig.suptitle(
        "Distribution of neighbourhood conditions",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(DISTRIBUTION_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    lsoa_crime_rho = lsoa["mean_monthly_crime"].corr(
        lsoa["median_log_price"], method="spearman"
    )
    lsoa_claimant_rho = lsoa["mean_claimant_rate_percent"].corr(
        lsoa["median_log_price"], method="spearman"
    )
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.7))
    crime_x = np.log1p(lsoa["mean_monthly_crime"])
    axes[0].scatter(
        crime_x,
        lsoa["median_price"],
        s=np.clip(lsoa["transaction_count"] * 1.5, 18, 170),
        alpha=0.55,
        color=BLUE,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[0].set_title("Crime and area median price")
    axes[0].set_xlabel("log(1 + mean monthly crime)")
    axes[0].set_ylabel("LSOA median sale price")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[0].grid()
    axes[0].text(
        0.04,
        0.94,
        f"Spearman ρ = {lsoa_crime_rho:.2f}",
        transform=axes[0].transAxes,
        va="top",
        color=NAVY,
        fontweight="bold",
    )

    claimant_x = lsoa["mean_claimant_rate_percent"]
    axes[1].scatter(
        claimant_x,
        lsoa["median_price"],
        s=np.clip(lsoa["transaction_count"] * 1.5, 18, 170),
        alpha=0.55,
        color=ORANGE,
        edgecolor="white",
        linewidth=0.5,
    )
    axes[1].set_title("Claimant rate and area median price")
    axes[1].set_xlabel("Mean claimant rate (%)")
    axes[1].set_ylabel("LSOA median sale price")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[1].grid()
    axes[1].text(
        0.04,
        0.94,
        f"Spearman ρ = {lsoa_claimant_rho:.2f}",
        transform=axes[1].transAxes,
        va="top",
        color=NAVY,
        fontweight="bold",
    )

    fig.suptitle(
        "Neighbourhood conditions are negatively associated with sale prices",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.015,
        "Each point is one 2011 LSOA; point size reflects transaction count. Associations are descriptive, not causal.",
        ha="center",
        color=GREY,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(RELATIONSHIP_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.3), sharey=True)
    quintile_titles = ["Monthly crime", "Population-scaled crime proxy", "Claimant rate"]
    colors = [BLUE, TEAL, ORANGE]
    for axis, (display_name, feature), title, color in zip(
        axes, quintile_definitions, quintile_titles, colors
    ):
        chart_data = quintiles.loc[quintiles["feature"].eq(feature)]
        axis.plot(
            chart_data["quintile"].astype(str),
            chart_data["median_of_lsoa_median_prices"],
            color=color,
            linewidth=2.8,
            marker="o",
            markersize=7,
        )
        for x_index, value in enumerate(chart_data["median_of_lsoa_median_prices"]):
            axis.text(
                x_index,
                value + 7_500,
                f"£{value / 1_000:.0f}k",
                ha="center",
                fontsize=9,
                color=NAVY,
            )
        axis.set_title(title)
        axis.set_xlabel("Neighbourhood quintile")
        axis.grid(axis="y")
        axis.set_ylim(235_000, 515_000)
        axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[0].set_ylabel("Median of LSOA median sale prices")
    fig.suptitle(
        "Area median prices fall across higher-crime and higher-claimant neighbourhoods",
        fontsize=16,
        fontweight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.015,
        "Each quintile contains approximately one fifth of Bristol's 263 LSOAs; areas are weighted equally.",
        ha="center",
        color=GREY,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(QUINTILE_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.4), sharex=True)
    axes[0].plot(
        monthly_features["month_date"],
        monthly_features["bristol_total_crime"],
        color=BLUE,
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    axes[0].set_title("Monthly Bristol crime total after 2011-LSOA standardisation")
    axes[0].set_ylabel("Recorded crimes")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axes[0].grid(axis="y")

    axes[1].plot(
        monthly_features["month_date"],
        monthly_features["median_claimant_rate_percent"],
        color=ORANGE,
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    axes[1].set_title("Median LSOA claimant rate")
    axes[1].set_ylabel("Claimant rate (%)")
    axes[1].set_xlabel("Month")
    axes[1].grid(axis="y")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[1].set_xlim(
        monthly_features["month_date"].min() - pd.Timedelta(days=12),
        monthly_features["month_date"].max() + pd.Timedelta(days=15),
    )
    boundary_transition = pd.Timestamp("2023-06-01")
    for axis in axes:
        axis.axvline(boundary_transition, color=PURPLE, linestyle="--", linewidth=1.4)
    axes[0].text(
        boundary_transition + pd.Timedelta(days=12),
        axes[0].get_ylim()[1] * 0.985,
        "Original source boundary transition",
        color=PURPLE,
        fontsize=9,
        va="top",
    )
    fig.suptitle(
        "Neighbourhood features across the study period",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(MONTHLY_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    panel_distribution = distribution.loc[
        distribution["analysis_level"].eq("LSOA-month panel")
    ].set_index("feature")
    transaction_corr = correlation_table.loc[
        correlation_table["analysis_level"].eq("Category A transactions")
    ].set_index("x_variable")
    lsoa_corr = correlation_table.loc[
        correlation_table["analysis_level"].eq("LSOA aggregate")
    ].set_index("x_variable")
    feature_corr = correlation_table.loc[
        correlation_table["analysis_level"].eq("LSOA-month feature relationship")
    ].set_index(["x_variable", "y_variable"])

    low_crime_price = quintiles.loc[
        quintiles["feature"].eq("mean_monthly_crime")
        & quintiles["quintile"].astype(str).eq("Q1 low"),
        "median_of_lsoa_median_prices",
    ].iloc[0]
    high_crime_price = quintiles.loc[
        quintiles["feature"].eq("mean_monthly_crime")
        & quintiles["quintile"].astype(str).eq("Q5 high"),
        "median_of_lsoa_median_prices",
    ].iloc[0]
    low_claimant_price = quintiles.loc[
        quintiles["feature"].eq("mean_claimant_rate_percent")
        & quintiles["quintile"].astype(str).eq("Q1 low"),
        "median_of_lsoa_median_prices",
    ].iloc[0]
    high_claimant_price = quintiles.loc[
        quintiles["feature"].eq("mean_claimant_rate_percent")
        & quintiles["quintile"].astype(str).eq("Q5 high"),
        "median_of_lsoa_median_prices",
    ].iloc[0]

    findings = f"""# Neighbourhood-conditions exploratory analysis

## Scope

This checkpoint examines monthly crime, claimant conditions and their association with Category A residential sale prices. It uses three analytical levels:

- **18,670 property transactions** for the property-level relationship.
- **9,468 LSOA-month observations** representing all 263 Bristol 2011 LSOAs across 36 months.
- **263 LSOA summaries** so areas with many property sales do not dominate the neighbourhood comparison.

The original datasets remain unchanged.

## Feature distributions

| Feature | Median | Mean | 99th percentile | Maximum | Skewness |
|---|---:|---:|---:|---:|---:|
| Monthly crime count | {panel_distribution.loc['total_crime', 'median']:.0f} | {panel_distribution.loc['total_crime', 'mean']:.1f} | {panel_distribution.loc['total_crime', 'p99']:.0f} | {panel_distribution.loc['total_crime', 'maximum']:.0f} | {panel_distribution.loc['total_crime', 'skewness']:.2f} |
| log(1 + monthly crime) | {panel_distribution.loc['log1p_total_crime', 'median']:.2f} | {panel_distribution.loc['log1p_total_crime', 'mean']:.2f} | {panel_distribution.loc['log1p_total_crime', 'p99']:.2f} | {panel_distribution.loc['log1p_total_crime', 'maximum']:.2f} | {panel_distribution.loc['log1p_total_crime', 'skewness']:.2f} |
| Crime per 1,000 working-age residents (proxy) | {panel_distribution.loc['crime_per_1000_working_age_proxy', 'median']:.1f} | {panel_distribution.loc['crime_per_1000_working_age_proxy', 'mean']:.1f} | {panel_distribution.loc['crime_per_1000_working_age_proxy', 'p99']:.1f} | {panel_distribution.loc['crime_per_1000_working_age_proxy', 'maximum']:.1f} | {panel_distribution.loc['crime_per_1000_working_age_proxy', 'skewness']:.2f} |
| Claimant rate (%) | {panel_distribution.loc['claimant_rate_percent', 'median']:.2f} | {panel_distribution.loc['claimant_rate_percent', 'mean']:.2f} | {panel_distribution.loc['claimant_rate_percent', 'p99']:.2f} | {panel_distribution.loc['claimant_rate_percent', 'maximum']:.2f} | {panel_distribution.loc['claimant_rate_percent', 'skewness']:.2f} |

Monthly crime counts are strongly right-skewed (**skewness {panel_distribution.loc['total_crime', 'skewness']:.2f}**). Applying `log1p` reduces the skewness to **{panel_distribution.loc['log1p_total_crime', 'skewness']:.2f}**, supporting its use in linear models. Tree-based models can still be tested with the untransformed count.

## Association with property prices

Transaction-level Spearman correlations with log sale price:

- Monthly crime count: **{transaction_corr.loc['total_crime', 'spearman_correlation']:.3f}**.
- Crime per 1,000 working-age residents proxy: **{transaction_corr.loc['crime_per_1000_working_age_proxy', 'spearman_correlation']:.3f}**.
- Claimant rate: **{transaction_corr.loc['claimant_rate_percent', 'spearman_correlation']:.3f}**.

LSOA-level Spearman correlations with area median log price:

- Mean monthly crime: **{lsoa_corr.loc['mean_monthly_crime', 'spearman_correlation']:.3f}**.
- Mean population-scaled crime proxy: **{lsoa_corr.loc['mean_crime_per_1000_working_age_proxy', 'spearman_correlation']:.3f}**.
- Mean claimant rate: **{lsoa_corr.loc['mean_claimant_rate_percent', 'spearman_correlation']:.3f}**.

The neighbourhood-level relationships are stronger than the transaction-level relationships because property characteristics introduce additional variation. The correlations remain descriptive: they do not demonstrate that crime or claimant rates cause lower prices.

## Quintile comparison

- The lowest mean-crime quintile has a median area price of **{money(low_crime_price)}**, compared with **{money(high_crime_price)}** in the highest-crime quintile.
- The lowest claimant-rate quintile has a median area price of **{money(low_claimant_price)}**, compared with **{money(high_claimant_price)}** in the highest claimant-rate quintile.

The ordered pattern supports testing both neighbourhood conditions in the enriched prediction models. It may also reflect unmeasured location, accessibility, property mix and neighbourhood characteristics.

## Relationship between the neighbourhood variables

Crime and claimant conditions are related. The LSOA-month Spearman correlation between monthly crime and claimant rate is **{feature_corr.loc[('total_crime', 'claimant_rate_percent'), 'spearman_correlation']:.3f}**. Raw crime and the population-scaled crime proxy are very strongly correlated (**{feature_corr.loc[('total_crime', 'crime_per_1000_working_age_proxy'), 'spearman_correlation']:.3f}**).

Raw crime, log crime and the crime-rate proxy should therefore not all be included in the same simple linear model. They should be compared as alternative representations.

## Crime-rate limitation

The population-scaled crime measure uses the available **2020 working-age population** as its denominator. This is a useful sensitivity measure, but it is not a true crime rate for the full resident or daytime population. City-centre crime can also reflect visitors and commuters. The primary enriched experiment should therefore use `log1p(total_crime)`, with the population-scaled proxy tested separately.

## Modelling implications

1. Use `log1p(total_crime)` and `claimant_rate_percent` in the primary enriched linear model.
2. Compare raw crime and the population-scaled proxy in sensitivity experiments rather than placing all versions together.
3. Retain month or date controls because both neighbourhood features vary through time.
4. Check multicollinearity and feature importance after property and time characteristics are included.
5. Use the same 2022–2023 training data and fixed 2024 test data for both the property-only and enriched models.
6. Interpret improvement in prediction as incremental information, not evidence of causation.

## Data-quality confirmation

The neighbourhood panel contains all **263 LSOAs × 36 months = 9,468 observations**, with no duplicate LSOA-month combinations or missing feature values. The summed crime total is **190,613**, matching the standardised Bristol crime table. The monthly chart shows no artificial discontinuity at the original June 2023 boundary transition after conversion to consistent 2011 LSOAs.
"""
    FINDINGS.write_text(findings, encoding="utf-8")

    output_files = [
        DISTRIBUTION_SUMMARY,
        CORRELATION_SUMMARY,
        TYPE_CORRELATIONS,
        LSOA_SUMMARY,
        QUINTILE_SUMMARY,
        MONTHLY_SUMMARY,
        DISTRIBUTION_CHART,
        RELATIONSHIP_CHART,
        QUINTILE_CHART,
        MONTHLY_CHART,
        FINDINGS,
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {
            str(PROPERTY_DATASET.relative_to(PROJECT_ROOT)): sha256(PROPERTY_DATASET),
            str(CRIME_DATASET.relative_to(PROJECT_ROOT)): sha256(CRIME_DATASET),
            str(CLAIMANT_DATASET.relative_to(PROJECT_ROOT)): sha256(CLAIMANT_DATASET),
        },
        "analysis_cohort": "PPD Category A",
        "transaction_rows": len(transactions),
        "lsoa_month_rows": len(panel),
        "lsoa_rows": len(lsoa),
        "checks": {
            "lsoa_month_duplicates": int(panel.duplicated(["lsoa_code", "month"]).sum()),
            "panel_missing_feature_values": int(panel[FEATURES].isna().sum().sum()),
            "transaction_missing_analysis_values": int(
                transactions[FEATURES + ["price", "log_price"]].isna().sum().sum()
            ),
            "panel_months": int(panel["month"].nunique()),
            "panel_lsoas": int(panel["lsoa_code"].nunique()),
            "panel_crime_total": int(panel["total_crime"].sum()),
            "lsoa_transaction_reconciliation": int(lsoa["transaction_count"].sum()),
            "lsoas_below_20_transactions": int(lsoa["low_transaction_count_flag"].sum()),
        },
        "analysis_decisions": {
            "primary_crime_feature_for_linear_model": "log1p(total_crime)",
            "primary_socioeconomic_feature": "claimant_rate_percent",
            "crime_rate_proxy_policy": "sensitivity analysis only; denominator is 2020 working-age population",
            "causal_interpretation": "not supported by descriptive correlations",
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Category A transactions: {len(transactions):,}")
    print(f"LSOA-month observations: {len(panel):,}")
    print(f"LSOAs: {len(lsoa):,}")
    print(f"Crime total: {panel['total_crime'].sum():,}")
    print(f"Raw crime skewness: {panel['total_crime'].skew():.3f}")
    print(f"Log crime skewness: {panel['log1p_total_crime'].skew():.3f}")
    print(f"LSOA crime-price Spearman: {lsoa_crime_rho:.3f}")
    print(f"LSOA claimant-price Spearman: {lsoa_claimant_rho:.3f}")
    print(f"Saved outputs to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
