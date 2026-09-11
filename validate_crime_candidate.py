"""Validate the complete 2011-LSOA Bristol crime candidate dataset."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pandas as pd

from test_june_2023_lsoa11_conversion import remove_repeated_crime_ids


PROJECT_ROOT = Path(__file__).resolve().parent
CANDIDATE_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "candidate"
    / "crime_features_2022_2024_lsoa11_candidate.csv"
)
CURRENT_CRIME_FILE = PROJECT_ROOT / "outputs" / "crime_features_2022_2024.csv"
CURRENT_PROPERTY_FILE = (
    PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2022_2024.csv"
)
BOUNDARY_FILE = (
    PROJECT_ROOT
    / "verified_sources"
    / "bristol_lsoa11_boundaries_bfc_v3.geojson"
)
INPUT_FOLDER = PROJECT_ROOT / "upload"


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_bristol_counts(months: list[str]) -> tuple[dict[str, int], int, int]:
    """Independently count raw and deduplicated source Bristol rows by month."""
    monthly_counts: dict[str, int] = {}
    total_raw_bristol = 0
    total_repeated_rows = 0

    for month in months:
        path = INPUT_FOLDER / f"{month}-avon-and-somerset-street.csv"
        with path.open(newline="", encoding="utf-8-sig") as source:
            raw_rows = list(csv.DictReader(source))
        total_raw_bristol += sum(
            row["LSOA name"].strip().startswith("Bristol ") for row in raw_rows
        )
        cleaned_rows, repeated_rows = remove_repeated_crime_ids(raw_rows)
        total_repeated_rows += repeated_rows
        monthly_counts[month] = sum(
            row["LSOA name"].strip().startswith("Bristol ") for row in cleaned_rows
        )

    return monthly_counts, total_raw_bristol, total_repeated_rows


def main() -> None:
    candidate = pd.read_csv(
        CANDIDATE_FILE,
        dtype={"lsoa_code": "string", "month": "string"},
    )
    current = pd.read_csv(
        CURRENT_CRIME_FILE,
        dtype={"lsoa_code": "string", "month": "string"},
    )
    boundary = json.loads(BOUNDARY_FILE.read_text(encoding="utf-8"))
    bristol_codes = {
        feature["properties"]["LSOA11CD"] for feature in boundary["features"]
    }
    months = [str(month) for month in pd.period_range("2022-01", "2024-12", freq="M")]
    count_columns = [column for column in candidate if column.endswith("_count")]

    duplicate_keys = int(candidate.duplicated(["lsoa_code", "month"]).sum())
    invalid_codes = sorted(set(candidate["lsoa_code"]) - bristol_codes)
    unexpected_months = sorted(set(candidate["month"]) - set(months))
    missing_values = int(candidate.isna().sum().sum())
    count_mismatches = int(
        (candidate["total_crime"] != candidate[count_columns].sum(axis=1)).sum()
    )
    negative_counts = int((candidate[["total_crime", *count_columns]] < 0).sum().sum())
    zero_crime_rows = int((candidate["total_crime"] == 0).sum())

    expected_grid = pd.MultiIndex.from_product(
        [sorted(bristol_codes), months], names=["lsoa_code", "month"]
    )
    candidate_grid = pd.MultiIndex.from_frame(candidate[["lsoa_code", "month"]])
    missing_combinations = expected_grid.difference(candidate_grid)
    missing_by_month = (
        pd.Series(missing_combinations.get_level_values("month"))
        .value_counts()
        .sort_index()
        .to_dict()
    )

    source_monthly, raw_bristol_total, all_force_repeated = source_bristol_counts(months)
    candidate_monthly = candidate.groupby("month")["total_crime"].sum().to_dict()
    source_mismatches = {
        month: (source_monthly[month], int(candidate_monthly.get(month, -1)))
        for month in months
        if source_monthly[month] != int(candidate_monthly.get(month, -1))
    }

    current_months = sorted(current["month"].dropna().unique())
    current_bristol = current[current["lsoa_code"].isin(bristol_codes)]
    current_total = int(current_bristol["total_crime"].sum())
    candidate_over_current_months = int(
        candidate[candidate["month"].isin(current_months)]["total_crime"].sum()
    )
    candidate_total = int(candidate["total_crime"].sum())

    if (
        duplicate_keys
        or invalid_codes
        or unexpected_months
        or len(candidate) != len(expected_grid)
        or len(missing_combinations)
    ):
        raise ValueError("Candidate key or geography validation failed.")
    if missing_values or count_mismatches or negative_counts:
        raise ValueError("Candidate value validation failed.")
    if source_mismatches:
        raise ValueError(f"Monthly source reconciliation failed: {source_mismatches}")

    print(f"Candidate rows: {len(candidate):,}")
    print(f"Expected complete-grid rows: {len(expected_grid):,}")
    print(f"Missing zero-crime LSOA-month combinations: {len(missing_combinations):,}")
    print(f"Missing combinations by month: {missing_by_month}")
    print(f"Unique LSOAs: {candidate['lsoa_code'].nunique():,}")
    print(f"Months: {candidate['month'].nunique():,}")
    print(f"Duplicate keys: {duplicate_keys:,}")
    print(f"Codes outside Bristol 2011 set: {len(invalid_codes):,}")
    print(f"Unexpected months: {len(unexpected_months):,}")
    print(f"Missing values: {missing_values:,}")
    print(f"Crime/count mismatches: {count_mismatches:,}")
    print(f"Negative counts: {negative_counts:,}")
    print(f"Explicit zero-crime rows: {zero_crime_rows:,}")
    print(f"Raw source Bristol rows: {raw_bristol_total:,}")
    print(f"All-force repeated nonblank ID rows removed: {all_force_repeated:,}")
    print(f"Candidate cleaned Bristol crimes: {candidate_total:,}")
    print(f"Monthly source/candidate mismatches: {len(source_mismatches):,}")
    print(
        "Current crime output coverage: "
        f"{len(current_months)} months ({min(current_months)} to {max(current_months)})"
    )
    print(
        "Current output recognised Bristol-code crimes in its available months: "
        f"{current_total:,}"
    )
    print(
        "Candidate Bristol crimes over those same months: "
        f"{candidate_over_current_months:,}"
    )
    print(f"Candidate full-period Bristol crimes: {candidate_total:,}")
    print(f"Current crime output SHA-256: {sha256(CURRENT_CRIME_FILE)}")
    print(f"Current property output SHA-256: {sha256(CURRENT_PROPERTY_FILE)}")


if __name__ == "__main__":
    main()
