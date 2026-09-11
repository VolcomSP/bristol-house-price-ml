"""Initial EDA: dataset quality, price distribution and time coverage."""

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
DATASET = PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2022_2024.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "eda"

QUALITY_SUMMARY = OUTPUT_FOLDER / "01_dataset_quality_summary.csv"
MISSINGNESS_SUMMARY = OUTPUT_FOLDER / "01_missingness_summary.csv"
MONTHLY_SUMMARY = OUTPUT_FOLDER / "01_monthly_summary.csv"
COHORT_SUMMARY = OUTPUT_FOLDER / "01_candidate_cohort_summary.csv"
PRICE_CHART = OUTPUT_FOLDER / "01_price_distribution_raw_vs_log.png"
MONTHLY_CHART = OUTPUT_FOLDER / "01_monthly_sales_and_median_price.png"
FINDINGS = OUTPUT_FOLDER / "01_initial_eda_findings.md"
MANIFEST = OUTPUT_FOLDER / "01_initial_eda.validation.json"

CORE_COLUMNS = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "sale_year",
    "sale_month",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "analysis_lsoa_code",
    "total_crime",
    "claimant_rate_percent",
]

BLUE = "#2563EB"
NAVY = "#172554"
TEAL = "#0F766E"
ORANGE = "#D97706"
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
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "font.size": 10,
            "font.family": "DejaVu Sans",
            "xtick.color": GREY,
            "ytick.color": GREY,
            "grid.color": LIGHT_GREY,
            "grid.linewidth": 0.7,
        }
    )


def add_metric(rows: list[dict[str, object]], section: str, metric: str, value: object) -> None:
    rows.append({"section": section, "metric": metric, "value": value})


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(DATASET, low_memory=False)
    data["date_of_transfer"] = pd.to_datetime(
        data["date_of_transfer"], format="%Y-%m-%d", errors="coerce"
    )
    data["sale_month_date"] = pd.to_datetime(data["sale_month"] + "-01")
    data["log_price"] = np.log(data["price"])

    rows: list[dict[str, object]] = []
    add_metric(rows, "dataset", "rows", len(data))
    add_metric(rows, "dataset", "columns", len(data.columns) - 2)
    add_metric(rows, "dataset", "unique_transaction_ids", data["transaction_id"].nunique())
    add_metric(rows, "dataset", "duplicate_transaction_ids", data["transaction_id"].duplicated().sum())
    add_metric(rows, "dataset", "exact_duplicate_rows", data.drop(columns=["sale_month_date", "log_price"]).duplicated().sum())
    add_metric(rows, "dataset", "months", data["sale_month"].nunique())
    add_metric(rows, "dataset", "date_min", data["date_of_transfer"].min().date().isoformat())
    add_metric(rows, "dataset", "date_max", data["date_of_transfer"].max().date().isoformat())
    add_metric(rows, "dataset", "lsoa11_areas", data["analysis_lsoa_code"].nunique())
    add_metric(rows, "dataset", "unmatched_crime_rows", (~data["crime_data_matched"]).sum())
    add_metric(rows, "dataset", "unmatched_claimant_rows", (~data["claimant_data_matched"]).sum())

    quantiles = data["price"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    price_metrics = {
        "minimum": data["price"].min(),
        "p01": quantiles.loc[0.01],
        "p05": quantiles.loc[0.05],
        "p25": quantiles.loc[0.25],
        "median": quantiles.loc[0.50],
        "mean": data["price"].mean(),
        "p75": quantiles.loc[0.75],
        "p95": quantiles.loc[0.95],
        "p99": quantiles.loc[0.99],
        "maximum": data["price"].max(),
        "standard_deviation": data["price"].std(),
        "raw_price_skewness": data["price"].skew(),
        "log_price_skewness": data["log_price"].skew(),
    }
    for metric, value in price_metrics.items():
        add_metric(rows, "price", metric, round(float(value), 6))

    q1 = quantiles.loc[0.25]
    q3 = quantiles.loc[0.75]
    iqr = q3 - q1
    lower_iqr = q1 - 1.5 * iqr
    upper_iqr = q3 + 1.5 * iqr
    add_metric(rows, "outliers", "iqr_lower_bound", round(float(lower_iqr), 2))
    add_metric(rows, "outliers", "iqr_upper_bound", round(float(upper_iqr), 2))
    add_metric(rows, "outliers", "rows_below_iqr_bound", int((data["price"] < lower_iqr).sum()))
    add_metric(rows, "outliers", "rows_above_iqr_bound", int((data["price"] > upper_iqr).sum()))
    add_metric(rows, "outliers", "rows_below_25000", int((data["price"] < 25_000).sum()))
    add_metric(rows, "outliers", "rows_above_2000000", int((data["price"] > 2_000_000).sum()))

    category_counts = data["ppd_category"].value_counts(dropna=False)
    for category, count in category_counts.items():
        add_metric(rows, "ppd_category", f"category_{category}_rows", int(count))
    type_counts = data["property_type"].value_counts(dropna=False)
    for property_type, count in type_counts.items():
        add_metric(rows, "property_type", f"type_{property_type}_rows", int(count))

    quality = pd.DataFrame(rows)
    quality.to_csv(QUALITY_SUMMARY, index=False)

    missingness = pd.DataFrame(
        {
            "column": [column for column in data.columns if column not in {"sale_month_date", "log_price"}],
            "missing_count": [
                int(data[column].isna().sum())
                for column in data.columns
                if column not in {"sale_month_date", "log_price"}
            ],
        }
    )
    missingness["missing_percent"] = missingness["missing_count"] / len(data) * 100
    missingness["required_for_initial_modelling"] = missingness["column"].isin(CORE_COLUMNS)
    missingness = missingness.sort_values(
        ["missing_count", "column"], ascending=[False, True]
    ).reset_index(drop=True)
    missingness.to_csv(MISSINGNESS_SUMMARY, index=False, float_format="%.4f")

    monthly = (
        data.groupby(["sale_month", "sale_month_date"], as_index=False)
        .agg(
            transaction_count=("transaction_id", "size"),
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            p25_price=("price", lambda values: values.quantile(0.25)),
            p75_price=("price", lambda values: values.quantile(0.75)),
        )
        .sort_values("sale_month_date")
    )
    monthly.drop(columns="sale_month_date").to_csv(
        MONTHLY_SUMMARY, index=False, float_format="%.2f"
    )

    standard_type = data["property_type"].isin(["D", "S", "T", "F"])
    category_a = data["ppd_category"].eq("A")
    inside_1_99 = data["price"].between(quantiles.loc[0.01], quantiles.loc[0.99])
    cohort = pd.DataFrame(
        [
            {"cohort": "All validated sales", "rows": len(data)},
            {"cohort": "PPD category A", "rows": int(category_a.sum())},
            {"cohort": "Standard property types D/S/T/F", "rows": int(standard_type.sum())},
            {
                "cohort": "Category A and standard property types",
                "rows": int((category_a & standard_type).sum()),
            },
            {
                "cohort": "Category A, standard types and price within p01-p99",
                "rows": int((category_a & standard_type & inside_1_99).sum()),
            },
        ]
    )
    cohort["percent_of_all_sales"] = cohort["rows"] / len(data) * 100
    cohort.to_csv(COHORT_SUMMARY, index=False, float_format="%.2f")

    set_chart_style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    display_limit = quantiles.loc[0.99]
    displayed = data.loc[data["price"].le(display_limit), "price"]
    axes[0].hist(displayed, bins=55, color=BLUE, alpha=0.9, edgecolor="white")
    axes[0].axvline(data["price"].median(), color=ORANGE, linewidth=2, label="Median")
    axes[0].set_title("Sale-price distribution")
    axes[0].set_xlabel("Sale price")
    axes[0].set_ylabel("Transactions")
    axes[0].xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value/1_000:.0f}k"))
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False)
    axes[0].text(
        0,
        -0.22,
        f"Display limited to the 99th percentile ({money(display_limit)}); no rows removed.",
        transform=axes[0].transAxes,
        color=GREY,
        fontsize=9,
    )

    axes[1].hist(data["log_price"], bins=55, color=TEAL, alpha=0.9, edgecolor="white")
    axes[1].axvline(data["log_price"].median(), color=ORANGE, linewidth=2, label="Median")
    axes[1].set_title("Log-transformed sale price")
    axes[1].set_xlabel("Natural log of sale price")
    axes[1].set_ylabel("Transactions")
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Bristol residential sale prices, 2022–2024",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(PRICE_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(13.5, 7.5), sharex=True)
    axes[0].bar(monthly["sale_month_date"], monthly["transaction_count"], width=20, color=BLUE)
    axes[0].set_title("Monthly transaction volume")
    axes[0].set_ylabel("Transactions")
    axes[0].grid(axis="y")

    axes[1].plot(
        monthly["sale_month_date"],
        monthly["median_price"],
        color=TEAL,
        linewidth=2.5,
        marker="o",
        markersize=4,
    )
    axes[1].set_title("Monthly median sale price")
    axes[1].set_ylabel("Median price")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value/1_000:.0f}k"))
    axes[1].grid(axis="y")
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    axes[1].set_xlim(
        monthly["sale_month_date"].min() - pd.Timedelta(days=12),
        monthly["sale_month_date"].max() + pd.Timedelta(days=15),
    )
    axes[1].set_xlabel("Sale month")
    fig.suptitle(
        "Transaction coverage and prices across the study period",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(MONTHLY_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    core_missing = int(data[CORE_COLUMNS].isna().sum().sum())
    findings = f"""# Initial exploratory data analysis

## Scope

This first EDA checkpoint assesses dataset quality, study-period coverage and the sale-price distribution. It does not remove or modify any property transaction.

## Dataset quality

- Transactions: **{len(data):,}**
- Columns in the modelling dataset: **{len(data.columns) - 2}**
- Period: **{data['date_of_transfer'].min().date()} to {data['date_of_transfer'].max().date()}**
- Months represented: **{data['sale_month'].nunique()}**
- Bristol 2011 LSOAs represented: **{data['analysis_lsoa_code'].nunique()}**
- Duplicate transaction IDs: **{data['transaction_id'].duplicated().sum():,}**
- Exact duplicate rows: **{data.drop(columns=['sale_month_date', 'log_price']).duplicated().sum():,}**
- Missing values across the initial modelling variables: **{core_missing:,}**
- Unmatched crime rows: **{(~data['crime_data_matched']).sum():,}**
- Unmatched claimant rows: **{(~data['claimant_data_matched']).sum():,}**

Blank values remain in optional address and postcode-history fields such as secondary address, locality and postcode termination date. These are structurally optional and are not required as model inputs.

## Price distribution

| Statistic | Sale price |
|---|---:|
| Minimum | {money(data['price'].min())} |
| 1st percentile | {money(quantiles.loc[0.01])} |
| 25th percentile | {money(quantiles.loc[0.25])} |
| Median | {money(data['price'].median())} |
| Mean | {money(data['price'].mean())} |
| 75th percentile | {money(quantiles.loc[0.75])} |
| 99th percentile | {money(quantiles.loc[0.99])} |
| Maximum | {money(data['price'].max())} |

Raw-price skewness is **{data['price'].skew():.2f}**, compared with **{data['log_price'].skew():.2f}** after applying the natural logarithm. The transformation therefore reduces the extreme right skew substantially, although the transformed distribution remains moderately left-skewed.

## Records requiring modelling decisions

- PPD Category B transactions: **{int(data['ppd_category'].eq('B').sum()):,}**
- `Other` property types: **{int(data['property_type'].eq('O').sum()):,}**
- Prices below £25,000: **{int(data['price'].lt(25_000).sum()):,}**
- Prices above £2,000,000: **{int(data['price'].gt(2_000_000).sum()):,}**
- Prices above the conventional upper IQR threshold of {money(upper_iqr)}: **{int(data['price'].gt(upper_iqr).sum()):,}**

These records are flagged for investigation; they have not been deleted. All **99** transactions below £25,000 are both PPD Category B and property type `Other`. The 20 lowest-priced and 20 highest-priced transactions are also all Category B/`Other`. Their address descriptions include whole buildings and commercial-style transactions, so they are not directly comparable with ordinary individual residential sales. Category A contains no `Other` property records.

## Initial modelling recommendation

1. Use **log sale price** as the primary modelling target because it materially reduces skewness and limits the influence of extremely expensive transactions.
2. Define the main residential modelling cohort as **PPD Category A**. In this dataset that produces **{int(category_a.sum()):,} transactions ({category_a.mean() * 100:.2f}% of all sales)** and automatically restricts the data to standard D, S, T and F property types.
3. Do not automatically remove the highest and lowest prices. Compare results using the full main cohort with a sensitivity analysis restricted to the 1st–99th percentile range.
4. Retain the complete validated dataset unchanged and create modelling subsets reproducibly in later scripts.

## Outputs

- `01_dataset_quality_summary.csv`: auditable quality and price statistics.
- `01_missingness_summary.csv`: missing-value counts for every column.
- `01_monthly_summary.csv`: monthly transaction and price summaries.
- `01_candidate_cohort_summary.csv`: row counts under alternative modelling rules.
- `01_price_distribution_raw_vs_log.png`: price transformation evidence.
- `01_monthly_sales_and_median_price.png`: study-period coverage and monthly median prices.
"""
    FINDINGS.write_text(findings, encoding="utf-8")

    output_files = [
        QUALITY_SUMMARY,
        MISSINGNESS_SUMMARY,
        MONTHLY_SUMMARY,
        COHORT_SUMMARY,
        PRICE_CHART,
        MONTHLY_CHART,
        FINDINGS,
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(DATASET.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(DATASET),
        "rows": len(data),
        "columns": len(data.columns) - 2,
        "checks": {
            "duplicate_transaction_ids": int(data["transaction_id"].duplicated().sum()),
            "exact_duplicate_rows": int(data.drop(columns=["sale_month_date", "log_price"]).duplicated().sum()),
            "core_missing_values": core_missing,
            "months": int(data["sale_month"].nunique()),
            "crime_unmatched": int((~data["crime_data_matched"]).sum()),
            "claimant_unmatched": int((~data["claimant_data_matched"]).sum()),
        },
        "analysis_decisions": {
            "primary_target": "log(price)",
            "primary_cohort": "PPD Category A and property types D/S/T/F",
            "outlier_policy": "retain in source; compare full cohort with p01-p99 sensitivity analysis",
        },
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Rows: {len(data):,}")
    print(f"Columns: {len(data.columns) - 2}")
    print(f"Duplicate transaction IDs: {data['transaction_id'].duplicated().sum():,}")
    print(f"Core missing values: {core_missing:,}")
    print(f"Median price: {money(data['price'].median())}")
    print(f"Mean price: {money(data['price'].mean())}")
    print(f"Maximum price: {money(data['price'].max())}")
    print(f"Raw-price skewness: {data['price'].skew():.3f}")
    print(f"Log-price skewness: {data['log_price'].skew():.3f}")
    print(f"Saved outputs to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
