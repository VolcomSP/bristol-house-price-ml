"""EDA checkpoint 2: property characteristics and Bristol sale prices."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET = PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2022_2024.csv"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "eda"

PROPERTY_TYPE_SUMMARY = OUTPUT_FOLDER / "02_property_type_summary.csv"
DURATION_SUMMARY = OUTPUT_FOLDER / "02_duration_summary.csv"
DURATION_BY_TYPE = OUTPUT_FOLDER / "02_duration_by_property_type_summary.csv"
NEW_BUILD_SUMMARY = OUTPUT_FOLDER / "02_new_build_summary.csv"
NEW_BUILD_BY_TYPE = OUTPUT_FOLDER / "02_new_build_by_property_type_summary.csv"
YEAR_SUMMARY = OUTPUT_FOLDER / "02_year_summary.csv"
TYPE_YEAR_SUMMARY = OUTPUT_FOLDER / "02_property_type_year_summary.csv"
PROPERTY_TYPE_CHART = OUTPUT_FOLDER / "02_median_price_by_property_type.png"
CHARACTERISTICS_CHART = OUTPUT_FOLDER / "02_tenure_and_new_build_by_property_type.png"
YEAR_CHART = OUTPUT_FOLDER / "02_annual_median_price_by_property_type.png"
FINDINGS = OUTPUT_FOLDER / "02_property_characteristics_findings.md"
MANIFEST = OUTPUT_FOLDER / "02_property_characteristics_eda.validation.json"

TYPE_ORDER = ["Flat/Maisonette", "Terraced", "Semi-detached", "Detached"]
TYPE_COLORS = {
    "Flat/Maisonette": "#2563EB",
    "Terraced": "#0F766E",
    "Semi-detached": "#D97706",
    "Detached": "#7C3AED",
}
DURATION_LABELS = {"F": "Freehold", "L": "Leasehold"}
NEW_BUILD_LABELS = {"N": "Established", "Y": "New build"}

NAVY = "#172554"
BLUE = "#2563EB"
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


def percent_change(new_value: float, reference_value: float) -> float:
    return (new_value / reference_value - 1) * 100


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


def grouped_summary(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    summary = (
        data.groupby(group_columns, observed=True, dropna=False)
        .agg(
            transaction_count=("transaction_id", "size"),
            median_price=("price", "median"),
            mean_price=("price", "mean"),
            p25_price=("price", lambda values: values.quantile(0.25)),
            p75_price=("price", lambda values: values.quantile(0.75)),
            minimum_price=("price", "min"),
            maximum_price=("price", "max"),
        )
        .reset_index()
    )
    summary["share_of_cohort_percent"] = (
        summary["transaction_count"] / len(data) * 100
        if len(group_columns) == 1
        else np.nan
    )
    return summary


def add_bar_labels(axis: plt.Axes, bars, counts: list[int] | None = None) -> None:
    for index, bar in enumerate(bars):
        value = bar.get_width() if bar.get_width() > 0 else bar.get_height()
        if bar.get_width() > 0:
            axis.text(
                value + axis.get_xlim()[1] * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"£{value / 1_000:.0f}k" + (f"  n={counts[index]:,}" if counts else ""),
                va="center",
                color=NAVY,
                fontsize=10,
            )
        else:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"£{value / 1_000:.0f}k",
                ha="center",
                va="bottom",
            )


def main() -> None:
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(DATASET, low_memory=False)
    cohort = source.loc[source["ppd_category"].eq("A")].copy()
    cohort["duration_label"] = cohort["duration"].map(DURATION_LABELS)
    cohort["new_build_label"] = cohort["old_new"].map(NEW_BUILD_LABELS)

    required = [
        "transaction_id",
        "price",
        "property_type",
        "property_type_label",
        "duration_label",
        "new_build_label",
        "sale_year",
    ]
    assert len(source) == 21_968
    assert len(cohort) == 18_670
    assert cohort[required].isna().sum().sum() == 0
    assert set(cohort["property_type"]) == {"D", "S", "T", "F"}
    assert cohort["transaction_id"].duplicated().sum() == 0

    property_type = grouped_summary(cohort, ["property_type_label"])
    property_type["property_type_label"] = pd.Categorical(
        property_type["property_type_label"], TYPE_ORDER, ordered=True
    )
    property_type = property_type.sort_values("property_type_label").reset_index(drop=True)
    property_type.to_csv(PROPERTY_TYPE_SUMMARY, index=False, float_format="%.2f")

    duration = grouped_summary(cohort, ["duration_label"]).sort_values("duration_label")
    duration.to_csv(DURATION_SUMMARY, index=False, float_format="%.2f")

    duration_by_type = grouped_summary(
        cohort, ["property_type_label", "duration_label"]
    )
    duration_by_type["property_type_label"] = pd.Categorical(
        duration_by_type["property_type_label"], TYPE_ORDER, ordered=True
    )
    duration_by_type["small_group_flag"] = duration_by_type["transaction_count"].lt(50)
    duration_by_type = duration_by_type.sort_values(
        ["property_type_label", "duration_label"]
    ).reset_index(drop=True)
    duration_by_type.to_csv(DURATION_BY_TYPE, index=False, float_format="%.2f")

    new_build = grouped_summary(cohort, ["new_build_label"]).sort_values("new_build_label")
    new_build.to_csv(NEW_BUILD_SUMMARY, index=False, float_format="%.2f")

    new_build_by_type = grouped_summary(
        cohort, ["property_type_label", "new_build_label"]
    )
    new_build_by_type["property_type_label"] = pd.Categorical(
        new_build_by_type["property_type_label"], TYPE_ORDER, ordered=True
    )
    new_build_by_type["small_group_flag"] = new_build_by_type["transaction_count"].lt(50)
    new_build_by_type = new_build_by_type.sort_values(
        ["property_type_label", "new_build_label"]
    ).reset_index(drop=True)
    new_build_by_type.to_csv(NEW_BUILD_BY_TYPE, index=False, float_format="%.2f")

    year = grouped_summary(cohort, ["sale_year"]).sort_values("sale_year")
    year.to_csv(YEAR_SUMMARY, index=False, float_format="%.2f")

    type_year = grouped_summary(cohort, ["sale_year", "property_type_label"])
    type_year["property_type_label"] = pd.Categorical(
        type_year["property_type_label"], TYPE_ORDER, ordered=True
    )
    type_year = type_year.sort_values(["sale_year", "property_type_label"]).reset_index(drop=True)
    type_year.to_csv(TYPE_YEAR_SUMMARY, index=False, float_format="%.2f")

    assert property_type["transaction_count"].sum() == len(cohort)
    assert duration["transaction_count"].sum() == len(cohort)
    assert new_build["transaction_count"].sum() == len(cohort)
    assert year["transaction_count"].sum() == len(cohort)
    assert duration_by_type["transaction_count"].sum() == len(cohort)
    assert new_build_by_type["transaction_count"].sum() == len(cohort)
    assert type_year["transaction_count"].sum() == len(cohort)

    set_chart_style()
    fig, axis = plt.subplots(figsize=(10.5, 5.6))
    chart_data = property_type.sort_values("median_price")
    bars = axis.barh(
        chart_data["property_type_label"].astype(str),
        chart_data["median_price"],
        color=[TYPE_COLORS[label] for label in chart_data["property_type_label"].astype(str)],
        height=0.62,
    )
    axis.set_xlim(0, chart_data["median_price"].max() * 1.30)
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axis.set_xlabel("Median sale price")
    axis.set_ylabel("")
    axis.grid(axis="x")
    add_bar_labels(axis, bars, chart_data["transaction_count"].astype(int).tolist())
    axis.set_title("Median sale price differs substantially by property type", pad=14)
    fig.suptitle(
        "Bristol Category A residential sales, 2022–2024",
        fontsize=12,
        color=GREY,
        y=0.96,
    )
    fig.tight_layout()
    fig.savefig(PROPERTY_TYPE_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.9))
    x = np.arange(len(TYPE_ORDER))
    width = 0.36

    duration_pivot = duration_by_type.pivot(
        index="property_type_label", columns="duration_label", values="median_price"
    ).reindex(TYPE_ORDER)
    duration_count = duration_by_type.pivot(
        index="property_type_label", columns="duration_label", values="transaction_count"
    ).reindex(TYPE_ORDER)
    for offset, label, color in [(-width / 2, "Freehold", BLUE), (width / 2, "Leasehold", TEAL)]:
        values = duration_pivot[label].to_numpy()
        bars = axes[0].bar(x + offset, values, width, label=label, color=color)
        for bar, count in zip(bars, duration_count[label].to_numpy()):
            axes[0].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8_000,
                f"n={int(count):,}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color=GREY,
            )
    axes[0].set_title("Tenure comparison within property type")
    axes[0].set_xticks(x, TYPE_ORDER, rotation=18, ha="right")
    axes[0].set_ylabel("Median sale price")
    axes[0].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[0].grid(axis="y")
    axes[0].legend(frameon=False, loc="upper left")
    axes[0].set_ylim(0, np.nanmax(duration_pivot.to_numpy()) * 1.32)

    new_build_pivot = new_build_by_type.pivot(
        index="property_type_label", columns="new_build_label", values="median_price"
    ).reindex(TYPE_ORDER)
    new_build_count = new_build_by_type.pivot(
        index="property_type_label", columns="new_build_label", values="transaction_count"
    ).reindex(TYPE_ORDER)
    for offset, label, color in [(-width / 2, "Established", BLUE), (width / 2, "New build", ORANGE)]:
        values = new_build_pivot[label].to_numpy()
        bars = axes[1].bar(x + offset, values, width, label=label, color=color)
        for bar, count in zip(bars, new_build_count[label].to_numpy()):
            axes[1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 8_000,
                f"n={int(count):,}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
                color=GREY,
            )
    axes[1].set_title("New-build comparison within property type")
    axes[1].set_xticks(x, TYPE_ORDER, rotation=18, ha="right")
    axes[1].set_ylabel("Median sale price")
    axes[1].yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axes[1].grid(axis="y")
    axes[1].legend(frameon=False, loc="upper left")
    axes[1].set_ylim(0, np.nanmax(new_build_pivot.to_numpy()) * 1.32)

    fig.suptitle(
        "Property characteristics and median sale price",
        fontsize=17,
        fontweight="bold",
        color=NAVY,
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Groups with fewer than 50 transactions should be interpreted cautiously; comparisons are descriptive, not causal.",
        ha="center",
        color=GREY,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(CHARACTERISTICS_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(11.5, 6.0))
    for label in TYPE_ORDER:
        values = type_year.loc[type_year["property_type_label"].eq(label)].sort_values("sale_year")
        axis.plot(
            values["sale_year"],
            values["median_price"],
            linewidth=2.7,
            marker="o",
            markersize=6,
            label=label,
            color=TYPE_COLORS[label],
        )
    axis.set_xticks([2022, 2023, 2024])
    axis.set_xlabel("Sale year")
    axis.set_ylabel("Median sale price")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"£{value / 1_000:.0f}k"))
    axis.grid(axis="y")
    axis.legend(frameon=False, ncol=2, loc="upper right")
    axis.set_title("Annual median prices by property type", pad=14)
    fig.suptitle(
        "Bristol Category A residential sales",
        fontsize=12,
        color=GREY,
        y=0.96,
    )
    fig.tight_layout()
    fig.savefig(YEAR_CHART, dpi=220, bbox_inches="tight")
    plt.close(fig)

    property_medians = property_type.set_index("property_type_label")["median_price"]
    year_medians = year.set_index("sale_year")["median_price"]
    duration_medians = duration.set_index("duration_label")["median_price"]
    build_medians = new_build.set_index("new_build_label")["median_price"]

    build_comparisons = []
    for label in TYPE_ORDER:
        group = new_build_by_type.loc[new_build_by_type["property_type_label"].eq(label)]
        medians = group.set_index("new_build_label")["median_price"]
        counts = group.set_index("new_build_label")["transaction_count"]
        build_comparisons.append(
            f"- {label}: new-build median {money(medians['New build'])} versus "
            f"{money(medians['Established'])} for established properties "
            f"({percent_change(medians['New build'], medians['Established']):+.1f}%; "
            f"new-build n={int(counts['New build']):,})."
        )

    findings = f"""# Property-characteristics exploratory analysis

## Scope

This checkpoint examines how sale prices vary by property type, tenure, new-build status and sale year. It uses the agreed primary residential cohort: **18,670 Land Registry Category A transactions**. The complete 21,968-row source dataset remains unchanged.

## Property type

| Property type | Transactions | Median price |
|---|---:|---:|
| Flat/Maisonette | {int(property_type.loc[property_type['property_type_label'].eq('Flat/Maisonette'), 'transaction_count'].iloc[0]):,} | {money(property_medians['Flat/Maisonette'])} |
| Terraced | {int(property_type.loc[property_type['property_type_label'].eq('Terraced'), 'transaction_count'].iloc[0]):,} | {money(property_medians['Terraced'])} |
| Semi-detached | {int(property_type.loc[property_type['property_type_label'].eq('Semi-detached'), 'transaction_count'].iloc[0]):,} | {money(property_medians['Semi-detached'])} |
| Detached | {int(property_type.loc[property_type['property_type_label'].eq('Detached'), 'transaction_count'].iloc[0]):,} | {money(property_medians['Detached'])} |

Detached properties have the highest median price and flats/maisonettes the lowest. The detached median is **{percent_change(property_medians['Detached'], property_medians['Flat/Maisonette']):.1f}%** higher than the flat/maisonette median. Property type is therefore likely to be an important predictor.

## Tenure

- Freehold transactions: **{int(duration.loc[duration['duration_label'].eq('Freehold'), 'transaction_count'].iloc[0]):,}**, median {money(duration_medians['Freehold'])}.
- Leasehold transactions: **{int(duration.loc[duration['duration_label'].eq('Leasehold'), 'transaction_count'].iloc[0]):,}**, median {money(duration_medians['Leasehold'])}.

The overall freehold/leasehold difference should not be interpreted as the independent effect of tenure. **{int(duration_by_type.loc[(duration_by_type['property_type_label'].eq('Flat/Maisonette')) & (duration_by_type['duration_label'].eq('Leasehold')), 'transaction_count'].iloc[0]):,} of 6,391 flats/maisonettes are leasehold**, while most houses are freehold. Property type and tenure are therefore strongly related and both should be controlled for in modelling. The detached-leasehold group contains only 30 transactions, so its median is unstable; freehold flats are also uncommon, with 84 transactions.

## New-build status

- Established properties: **{int(new_build.loc[new_build['new_build_label'].eq('Established'), 'transaction_count'].iloc[0]):,}**, median {money(build_medians['Established'])}.
- New builds: **{int(new_build.loc[new_build['new_build_label'].eq('New build'), 'transaction_count'].iloc[0]):,}**, median {money(build_medians['New build'])}.

Within-property-type comparisons:

{chr(10).join(build_comparisons)}

The direction and size of the new-build difference varies by property type, reinforcing the need to include both variables rather than treating the overall difference as causal.

## Sale year

| Year | Transactions | Median price |
|---:|---:|---:|
| 2022 | {int(year.loc[year['sale_year'].eq(2022), 'transaction_count'].iloc[0]):,} | {money(year_medians[2022])} |
| 2023 | {int(year.loc[year['sale_year'].eq(2023), 'transaction_count'].iloc[0]):,} | {money(year_medians[2023])} |
| 2024 | {int(year.loc[year['sale_year'].eq(2024), 'transaction_count'].iloc[0]):,} | {money(year_medians[2024])} |

These are nominal sale prices and have not been adjusted for inflation. Year and month should be retained as time controls because market conditions and the mix of properties sold change over time.

## Modelling implications

1. Include property type, tenure, new-build status and time variables in the baseline model.
2. Treat all categorical variables as categories rather than numeric ranks.
3. Check interactions or subgroup errors because tenure and new-build effects may vary by property type.
4. Avoid causal wording: these summaries demonstrate associations, not the effect of changing one characteristic while holding everything else constant.
5. Once modelling begins, use 2022–2023 for fitting and tuning. Evaluate 2024 only after the model and feature decisions are fixed; this full-period EDA is descriptive and must not be used to tune against 2024 prediction errors.
"""
    FINDINGS.write_text(findings, encoding="utf-8")

    output_files = [
        PROPERTY_TYPE_SUMMARY,
        DURATION_SUMMARY,
        DURATION_BY_TYPE,
        NEW_BUILD_SUMMARY,
        NEW_BUILD_BY_TYPE,
        YEAR_SUMMARY,
        TYPE_YEAR_SUMMARY,
        PROPERTY_TYPE_CHART,
        CHARACTERISTICS_CHART,
        YEAR_CHART,
        FINDINGS,
    ]
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(DATASET.relative_to(PROJECT_ROOT)),
        "source_sha256": sha256(DATASET),
        "source_rows": len(source),
        "analysis_cohort": "PPD Category A",
        "cohort_rows": len(cohort),
        "checks": {
            "cohort_duplicate_transaction_ids": int(cohort["transaction_id"].duplicated().sum()),
            "required_missing_values": int(cohort[required].isna().sum().sum()),
            "property_type_count_reconciliation": int(property_type["transaction_count"].sum()),
            "duration_count_reconciliation": int(duration["transaction_count"].sum()),
            "new_build_count_reconciliation": int(new_build["transaction_count"].sum()),
            "year_count_reconciliation": int(year["transaction_count"].sum()),
            "small_duration_type_groups": int(duration_by_type["small_group_flag"].sum()),
            "small_new_build_type_groups": int(new_build_by_type["small_group_flag"].sum()),
        },
        "interpretation_guardrails": [
            "descriptive associations only",
            "tenure is confounded with property type",
            "groups below 50 transactions flagged",
            "prices are nominal and not inflation-adjusted",
        ],
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output_files
        },
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Source rows: {len(source):,}")
    print(f"Category A cohort rows: {len(cohort):,}")
    print("Property-type medians:")
    for label in TYPE_ORDER:
        print(f"  {label}: {money(property_medians[label])}")
    print(f"Freehold median: {money(duration_medians['Freehold'])}")
    print(f"Leasehold median: {money(duration_medians['Leasehold'])}")
    print(f"Established median: {money(build_medians['Established'])}")
    print(f"New-build median: {money(build_medians['New build'])}")
    print(f"Saved outputs to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
