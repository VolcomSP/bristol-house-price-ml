"""Prepare monthly Police.uk crime features for the Bristol house-price project.

Input files are expected in the ``upload`` folder and must be named like:
``YYYY-MM-avon-and-somerset-street.csv``.

The output has one row per 2011 Bristol LSOA and month. It contains total crime
plus one count column for every crime type found in the source files. A source
code is retained whenever it belongs to the official Bristol 2011 LSOA set.
Other usable coordinates are assigned to the full-resolution 2011 boundaries.
"""

import argparse
from pathlib import Path

import pandas as pd

from test_lsoa11_assignment import assign_lsoa, load_boundaries, wgs84_to_bng


PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = PROJECT_FOLDER / "upload"
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
OUTPUT_FILE = OUTPUT_FOLDER / "crime_features_2022_2024.csv"

START_MONTH = "2022-01"
END_MONTH = "2024-12"

REQUIRED_COLUMNS = [
    "Crime ID",
    "Month",
    "Longitude",
    "Latitude",
    "LSOA code",
    "LSOA name",
    "Crime type",
]


def expected_months() -> list[str]:
    """Return every required month in YYYY-MM format."""
    return [
        str(month)
        for month in pd.period_range(START_MONTH, END_MONTH, freq="M")
    ]


def find_input_files(months: list[str] | None = None) -> list[Path]:
    """Find the required monthly Police.uk street-crime files."""
    selected_months = months or expected_months()
    invalid_months = [month for month in selected_months if month not in expected_months()]
    if invalid_months:
        raise ValueError("Months outside the project period: " + ", ".join(invalid_months))

    files = sorted(INPUT_FOLDER.glob("*-avon-and-somerset-street.csv"))
    files_by_month = {path.name[:7]: path for path in files}

    missing = [month for month in selected_months if month not in files_by_month]
    if missing:
        raise FileNotFoundError(
            "Missing Police.uk street-crime files for: " + ", ".join(missing)
        )

    selected = [files_by_month[month] for month in selected_months]
    print(f"Found all {len(selected)} required monthly files.")
    return selected


def load_and_check_files(files: list[Path]) -> pd.DataFrame:
    """Load each file, print checks, and combine the monthly data."""
    monthly_frames = []

    for path in files:
        data = pd.read_csv(path, low_memory=False)

        missing_columns = [
            column for column in REQUIRED_COLUMNS if column not in data.columns
        ]
        if missing_columns:
            raise ValueError(
                f"{path.name} is missing columns: {', '.join(missing_columns)}"
            )

        file_month = path.name[:7]
        data_months = sorted(data["Month"].dropna().astype(str).unique())
        if data_months != [file_month]:
            raise ValueError(
                f"{path.name} contains unexpected Month values: {data_months}"
            )

        print(
            f"{path.name}: {len(data):,} rows; "
            f"missing LSOA = {data['LSOA code'].isna().sum():,}"
        )
        monthly_frames.append(data[REQUIRED_COLUMNS].copy())

    combined = pd.concat(monthly_frames, ignore_index=True)
    print(f"\nRows after combining all files: {len(combined):,}")
    print(f"Months found: {combined['Month'].nunique()}")
    return combined


def remove_repeated_crime_ids(data: pd.DataFrame) -> pd.DataFrame:
    """Count each available Crime ID once per month.

    Police.uk does not supply Crime IDs for many anti-social-behaviour rows.
    Those rows are retained because each source row represents an incident.
    For repeated non-null IDs, a row with an LSOA and coordinates is preferred.
    """
    cleaned = data.copy()
    cleaned["Crime ID"] = cleaned["Crime ID"].astype("string").str.strip()
    has_id = cleaned["Crime ID"].notna() & cleaned["Crime ID"].ne("")

    identified = cleaned.loc[has_id].copy()
    anonymous = cleaned.loc[~has_id].copy()

    repeated_rows = identified.duplicated(["Month", "Crime ID"]).sum()
    identified["_has_lsoa"] = identified["LSOA code"].notna()
    identified["_has_coordinates"] = (
        identified["Longitude"].notna() & identified["Latitude"].notna()
    )
    identified["_source_order"] = identified.index

    identified = identified.sort_values(
        ["Month", "Crime ID", "_has_lsoa", "_has_coordinates", "_source_order"],
        ascending=[True, True, False, False, True],
    )
    identified = identified.drop_duplicates(["Month", "Crime ID"], keep="first")
    identified = identified.drop(
        columns=["_has_lsoa", "_has_coordinates", "_source_order"]
    )

    result = pd.concat([identified, anonymous], ignore_index=True)
    print(f"Repeated non-null Crime ID rows removed: {repeated_rows:,}")
    print(f"Rows after Crime ID check: {len(result):,}")
    return result


def assign_analysis_lsoa11(data: pd.DataFrame) -> pd.DataFrame:
    """Add a consistent 2011 Bristol LSOA without changing source fields.

    A published source code is retained whenever it belongs to the official
    Bristol 2011 LSOA set. Rows without a recognised Bristol 2011 code are
    tested against the full-resolution boundary using their coordinates. This
    handles the small number of mixed-vintage source codes before June 2023.
    """
    standardized = data.copy()
    boundaries, boundary_crs = load_boundaries()
    if boundary_crs != "EPSG:27700":
        raise ValueError(f"Expected EPSG:27700 boundary; found {boundary_crs}.")

    bristol_lsoa11_codes = {boundary["code"] for boundary in boundaries}
    if len(bristol_lsoa11_codes) != 263:
        raise ValueError(
            "Expected 263 Bristol 2011 LSOAs; "
            f"found {len(bristol_lsoa11_codes)}."
        )

    standardized["analysis_lsoa_code"] = pd.Series(
        pd.NA, index=standardized.index, dtype="string"
    )
    standardized["analysis_lsoa_method"] = pd.Series(
        pd.NA, index=standardized.index, dtype="string"
    )
    source_codes = standardized["LSOA code"].astype("string").str.strip()
    source_is_bristol = (
        standardized["LSOA name"]
        .astype("string")
        .str.strip()
        .str.startswith("Bristol ", na=False)
    )
    valid_source_bristol = source_codes.isin(bristol_lsoa11_codes)
    standardized.loc[valid_source_bristol, "analysis_lsoa_code"] = source_codes.loc[
        valid_source_bristol
    ]
    standardized.loc[
        valid_source_bristol, "analysis_lsoa_method"
    ] = "source_2011_code"

    longitude = pd.to_numeric(standardized["Longitude"], errors="coerce")
    latitude = pd.to_numeric(standardized["Latitude"], errors="coerce")
    has_coordinates = longitude.notna() & latitude.notna()
    coordinate_rows = ~valid_source_bristol & has_coordinates

    multiple_matches = 0
    for row_index in standardized.index[coordinate_rows]:
        easting, northing = wgs84_to_bng(
            float(latitude.at[row_index]), float(longitude.at[row_index])
        )
        matches = assign_lsoa(easting, northing, boundaries)
        if len(matches) > 1:
            multiple_matches += 1
        elif matches:
            standardized.at[row_index, "analysis_lsoa_code"] = matches[0]
            standardized.at[
                row_index, "analysis_lsoa_method"
            ] = "coordinate_conversion"

    if multiple_matches:
        raise ValueError(
            f"{multiple_matches:,} coordinates matched multiple 2011 LSOAs."
        )

    assigned_is_bristol = standardized["analysis_lsoa_code"].notna()
    for month in sorted(standardized["Month"].unique()):
        month_mask = standardized["Month"].eq(month)
        month_source = month_mask & source_is_bristol
        month_assigned = month_mask & assigned_is_bristol
        source_and_assigned = month_source & month_assigned
        source_only = month_source & ~month_assigned
        assigned_only = month_assigned & ~month_source
        source_code_assignments = month_mask & standardized[
            "analysis_lsoa_method"
        ].eq("source_2011_code")
        coordinate_assignments = month_mask & standardized[
            "analysis_lsoa_method"
        ].eq("coordinate_conversion")
        missing_coordinates = month_mask & ~valid_source_bristol & ~has_coordinates

        print(f"\n{month} geography checks")
        print(f"  Source rows labelled Bristol: {month_source.sum():,}")
        print(f"  Rows assigned to 2011 Bristol LSOAs: {month_assigned.sum():,}")
        print(f"  Retained source 2011 codes: {source_code_assignments.sum():,}")
        print(f"  Coordinate conversions: {coordinate_assignments.sum():,}")
        print(f"  Source and assigned Bristol: {source_and_assigned.sum():,}")
        print(f"  Source Bristol only: {source_only.sum():,}")
        print(f"  Assigned Bristol only: {assigned_only.sum():,}")
        print(
            "  Unrecognised all-force rows missing coordinates: "
            f"{missing_coordinates.sum():,}"
        )
        print(
            "  Unassigned source Bristol rows missing coordinates: "
            f"{(month_source & ~month_assigned & ~has_coordinates).sum():,}"
        )

    print(
        "\nRows assigned to consistent 2011 Bristol LSOAs: "
        f"{assigned_is_bristol.sum():,}"
    )
    print(f"Multiple boundary matches: {multiple_matches:,}")
    return standardized


def create_monthly_features(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate street-crime records to one row per 2011 Bristol LSOA and month."""
    usable = data.copy()
    usable["analysis_lsoa_code"] = (
        usable["analysis_lsoa_code"].astype("string").str.strip()
    )
    missing_lsoa = usable["analysis_lsoa_code"].isna() | usable[
        "analysis_lsoa_code"
    ].eq("")
    print(
        "Rows excluded because they are not assigned to a 2011 Bristol LSOA: "
        f"{missing_lsoa.sum():,}"
    )
    usable = usable.loc[~missing_lsoa].copy()

    usable["crime_feature"] = (
        usable["Crime type"]
        .astype("string")
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
        .add("_count")
    )

    features = (
        usable.groupby(["analysis_lsoa_code", "Month", "crime_feature"])
        .size()
        .unstack(fill_value=0)
    )

    boundaries, boundary_crs = load_boundaries()
    if boundary_crs != "EPSG:27700":
        raise ValueError(f"Expected EPSG:27700 boundary; found {boundary_crs}.")
    bristol_lsoa11_codes = sorted({boundary["code"] for boundary in boundaries})
    months = sorted(data["Month"].dropna().astype(str).unique())
    complete_index = pd.MultiIndex.from_product(
        [bristol_lsoa11_codes, months],
        names=["analysis_lsoa_code", "Month"],
    )
    zero_rows_added = len(complete_index.difference(features.index))
    features = features.reindex(complete_index, fill_value=0).reset_index()
    features.columns.name = None
    features = features.rename(
        columns={"analysis_lsoa_code": "lsoa_code", "Month": "month"}
    )

    count_columns = sorted(
        column for column in features.columns if column.endswith("_count")
    )
    features["total_crime"] = features[count_columns].sum(axis=1)
    features = features[["lsoa_code", "month", "total_crime", *count_columns]]
    features = features.sort_values(["month", "lsoa_code"]).reset_index(drop=True)

    duplicate_keys = features.duplicated(["lsoa_code", "month"]).sum()
    count_mismatches = (
        features["total_crime"] != features[count_columns].sum(axis=1)
    ).sum()
    if duplicate_keys or count_mismatches:
        raise ValueError("Final aggregation checks failed.")

    print(f"Final LSOA-month rows: {len(features):,}")
    print(f"Explicit zero-crime LSOA-month rows added: {zero_rows_added:,}")
    print(f"Unique LSOAs: {features['lsoa_code'].nunique():,}")
    print(f"Months: {features['month'].min()} to {features['month'].max()}")
    print(f"Duplicate LSOA-month keys: {duplicate_keys}")
    print(f"Rows where crime-type counts do not equal total_crime: {count_mismatches}")
    print("Missing values in final feature columns:")
    print(features.isna().sum().to_string())
    return features


def parse_args() -> argparse.Namespace:
    """Read optional month and output overrides for controlled validation runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--months",
        nargs="+",
        help="Optional YYYY-MM months to process instead of the full project period.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output CSV path. Required for a selected-month validation run.",
    )
    return parser.parse_args()


def main(months: list[str] | None = None, output_file: Path = OUTPUT_FILE) -> None:
    """Run the preparation workflow for the selected or complete month set."""
    files = find_input_files(months)
    crime = load_and_check_files(files)
    crime = remove_repeated_crime_ids(crime)
    crime = assign_analysis_lsoa11(crime)
    features = create_monthly_features(crime)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file}")


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.months and arguments.output is None:
        raise ValueError(
            "Selected-month runs require --output so the main feature file "
            "cannot be overwritten accidentally."
        )
    main(
        months=arguments.months,
        output_file=arguments.output or OUTPUT_FILE,
    )
