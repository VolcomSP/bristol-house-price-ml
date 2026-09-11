"""Independently validate the rebuilt Bristol property and crime dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_FOLDER = PROJECT_ROOT / "verified_sources" / "land_registry"
POSTCODE_LOOKUP = (
    PROJECT_ROOT / "verified_sources" / "ONSPD_MAY_2026_UK_BS_complete.csv"
)
CRIME_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "candidate"
    / "crime_features_2022_2024_lsoa11_candidate.csv"
)
CLAIMANT_FEATURES = (
    PROJECT_ROOT
    / "outputs"
    / "candidate"
    / "claimant_features_2022_2024_lsoa11_candidate.csv"
)
CANDIDATE = (
    PROJECT_ROOT
    / "outputs"
    / "candidate"
    / "bristol_house_prices_2022_2024_candidate.csv"
)
CANDIDATE_2024 = (
    PROJECT_ROOT
    / "outputs"
    / "candidate"
    / "bristol_house_prices_2024_candidate.csv"
)
MANIFEST = CANDIDATE.with_suffix(".validation.json")
BRISTOL_LAD_CODE = "E06000023"
EXPECTED_YEAR_COUNTS = {2022: 8093, 2023: 6839, 2024: 7036}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_postcode(value: str) -> str:
    return value.upper().replace(" ", "").strip()


def expected_bristol_ids() -> tuple[set[str], dict[int, int]]:
    lookup = pd.read_csv(
        POSTCODE_LOOKUP,
        usecols=["pcds", "lad25cd"],
        dtype="string",
        keep_default_na=False,
    )
    bristol_postcodes = set(
        lookup.loc[
            lookup["lad25cd"].eq(BRISTOL_LAD_CODE), "pcds"
        ].map(clean_postcode)
    )

    ids: set[str] = set()
    counts: Counter[int] = Counter()
    for year in EXPECTED_YEAR_COUNTS:
        path = SOURCE_FOLDER / f"pp-{year}.csv"
        with path.open(newline="", encoding="utf-8-sig") as source:
            for row in csv.reader(source):
                if len(row) == 16 and clean_postcode(row[3]) in bristol_postcodes:
                    ids.add(row[0])
                    counts[year] += 1
    return ids, dict(counts)


def fail_if(condition: bool, message: str) -> None:
    if condition:
        raise ValueError(message)


def main() -> None:
    data = pd.read_csv(
        CANDIDATE,
        dtype={
            "transaction_id": "string",
            "postcode": "string",
            "lad_code": "string",
            "lsoa11cd": "string",
            "crime_lsoa_code": "string",
            "analysis_lsoa_code": "string",
            "sale_month": "string",
        },
        low_memory=False,
    )
    data_2024 = pd.read_csv(CANDIDATE_2024, low_memory=False)
    data["date_of_transfer"] = pd.to_datetime(
        data["date_of_transfer"], format="%Y-%m-%d", errors="coerce"
    )

    expected_ids, raw_year_counts = expected_bristol_ids()
    actual_ids = set(data["transaction_id"])
    year_counts = data.groupby("sale_year").size().astype(int).to_dict()
    month_counts = data.groupby("sale_month").size().astype(int).to_dict()
    expected_months = {
        f"{year}-{month:02d}"
        for year in EXPECTED_YEAR_COUNTS
        for month in range(1, 13)
    }

    fail_if(len(data) != sum(EXPECTED_YEAR_COUNTS.values()), "Unexpected row count.")
    fail_if(year_counts != EXPECTED_YEAR_COUNTS, "Unexpected annual row counts.")
    fail_if(raw_year_counts != EXPECTED_YEAR_COUNTS, "Raw-source counts changed.")
    fail_if(data["transaction_id"].duplicated().any(), "Duplicate transaction IDs.")
    fail_if(actual_ids != expected_ids, "Candidate IDs differ from raw Bristol IDs.")
    fail_if(data["date_of_transfer"].isna().any(), "Invalid transfer dates.")
    fail_if(
        not data["date_of_transfer"].dt.year.eq(data["sale_year"]).all(),
        "Sale year disagrees with transfer date.",
    )
    fail_if(set(month_counts) != expected_months, "The dataset does not cover 36 months.")
    fail_if(not data["lad_code"].eq(BRISTOL_LAD_CODE).all(), "Non-Bristol rows found.")
    fail_if(data["postcode"].str.contains(r"\s", regex=True).any(), "Unclean postcodes.")
    fail_if((data["price"] <= 0).any(), "Non-positive prices found.")
    fail_if(
        not set(data["property_type"].dropna()).issubset({"D", "S", "T", "F", "O"}),
        "Unexpected property-type code.",
    )
    fail_if(
        not data["crime_lsoa_vintage"].eq("LSOA11").all(),
        "Mixed crime geography found.",
    )
    fail_if(
        not data["crime_lsoa_code"].eq(data["lsoa11cd"]).all(),
        "Crime LSOA does not equal property LSOA11.",
    )
    fail_if(
        not data["analysis_lsoa_code"].eq(data["lsoa11cd"]).all(),
        "Analysis LSOA does not equal property LSOA11.",
    )
    fail_if(not data["crime_data_matched"].all(), "Unmatched crime rows found.")
    fail_if(not data["claimant_data_matched"].all(), "Unmatched claimant rows found.")

    key_columns = [
        "transaction_id",
        "price",
        "date_of_transfer",
        "postcode",
        "lad_code",
        "lsoa11cd",
        "msoa11cd",
        "lsoa21cd",
        "msoa21cd",
        "latitude",
        "longitude",
        "analysis_lsoa_code",
        "total_crime",
        "claimant_count",
        "working_age_population_2020",
        "claimant_rate_percent",
        "population_reference_year",
    ]
    missing_key_values = data[key_columns].isna().sum().astype(int).to_dict()
    fail_if(any(missing_key_values.values()), "Missing values in required columns.")

    crime = pd.read_csv(CRIME_FEATURES, low_memory=False)
    crime_columns = [
        column for column in crime.columns if column not in {"lsoa_code", "month"}
    ]
    fail_if(crime.duplicated(["lsoa_code", "month"]).any(), "Duplicate crime keys.")
    comparison = data.merge(
        crime,
        left_on=["analysis_lsoa_code", "sale_month"],
        right_on=["lsoa_code", "month"],
        how="left",
        suffixes=("", "_source"),
        validate="many_to_one",
    )
    for column in crime_columns:
        fail_if(
            not comparison[column].eq(comparison[f"{column}_source"]).all(),
            f"Crime value mismatch in {column}.",
        )

    claimant = pd.read_csv(CLAIMANT_FEATURES, low_memory=False).rename(
        columns={
            "lsoa11cd": "claimant_lsoa_code",
            "month": "claimant_month",
        }
    )
    fail_if(
        claimant.duplicated(["claimant_lsoa_code", "claimant_month"]).any(),
        "Duplicate claimant keys.",
    )
    claimant_columns = [
        "claimant_count",
        "working_age_population_2020",
        "claimant_rate_percent",
        "population_reference_year",
    ]
    claimant_comparison = data.merge(
        claimant[["claimant_lsoa_code", "claimant_month", *claimant_columns]],
        left_on=["analysis_lsoa_code", "sale_month"],
        right_on=["claimant_lsoa_code", "claimant_month"],
        how="left",
        suffixes=("", "_source"),
        validate="many_to_one",
    )
    for column in claimant_columns:
        fail_if(
            not claimant_comparison[column].eq(
                claimant_comparison[f"{column}_source"]
            ).all(),
            f"Claimant value mismatch in {column}.",
        )
    calculated_rate = (
        data["claimant_count"] / data["working_age_population_2020"] * 100
    )
    fail_if(
        not calculated_rate.round(6).eq(data["claimant_rate_percent"].round(6)).all(),
        "Claimant rate formula mismatch.",
    )

    expected_2024 = data.loc[data["sale_year"].eq(2024)].drop(
        columns=["lsoa_code", "month"], errors="ignore"
    )
    fail_if(len(data_2024) != EXPECTED_YEAR_COUNTS[2024], "Bad 2024 subset count.")
    fail_if(
        set(data_2024["transaction_id"]) != set(expected_2024["transaction_id"]),
        "The 2024 subset does not match the full candidate.",
    )

    category_columns = [
        column
        for column in data.columns
        if column.endswith("_count") and column != "claimant_count"
    ]
    fail_if(
        not data[category_columns].sum(axis=1).eq(data["total_crime"]).all(),
        "Crime categories do not sum to total crime.",
    )

    manifest = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "candidate_file": CANDIDATE.name,
        "candidate_sha256": sha256(CANDIDATE),
        "candidate_2024_file": CANDIDATE_2024.name,
        "candidate_2024_sha256": sha256(CANDIDATE_2024),
        "rows": len(data),
        "year_counts": {str(key): value for key, value in year_counts.items()},
        "months": len(month_counts),
        "date_min": data["date_of_transfer"].min().date().isoformat(),
        "date_max": data["date_of_transfer"].max().date().isoformat(),
        "unique_transaction_ids": data["transaction_id"].nunique(),
        "unique_lsoa11": data["lsoa11cd"].nunique(),
        "crime_matches": int(data["crime_data_matched"].sum()),
        "claimant_matches": int(data["claimant_data_matched"].sum()),
        "missing_key_values": missing_key_values,
        "cross_checks": {
            "candidate_ids_equal_raw_bristol_ids": True,
            "annual_counts_equal_source_validation": True,
            "all_36_months_present": True,
            "all_rows_bristol_lad": True,
            "all_crime_geography_lsoa11": True,
            "all_crime_values_equal_source_table": True,
            "crime_categories_sum_to_total": True,
            "all_claimant_values_equal_source_table": True,
            "claimant_rate_formula_verified": True,
            "subset_2024_equals_full_candidate_2024": True,
        },
        "source_sha256": {
            f"pp-{year}.csv": sha256(SOURCE_FOLDER / f"pp-{year}.csv")
            for year in EXPECTED_YEAR_COUNTS
        },
        "postcode_lookup_sha256": sha256(POSTCODE_LOOKUP),
        "crime_features_sha256": sha256(CRIME_FEATURES),
        "claimant_features_sha256": sha256(CLAIMANT_FEATURES),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Rows: {len(data):,}")
    print(f"Annual counts: {year_counts}")
    print(f"Months: {len(month_counts)}")
    print(f"Date range: {manifest['date_min']} to {manifest['date_max']}")
    print(f"Unique transaction IDs: {manifest['unique_transaction_ids']:,}")
    print(f"Unique LSOA11 areas represented: {manifest['unique_lsoa11']:,}")
    print(f"Crime matches: {manifest['crime_matches']:,}")
    print(f"Claimant matches: {manifest['claimant_matches']:,}")
    print("All independent validation checks passed.")
    print(f"Validation manifest: {MANIFEST}")


if __name__ == "__main__":
    main()
