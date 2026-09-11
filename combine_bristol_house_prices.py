"""Create candidate Bristol house-price datasets with consistent crime geography.

This script combines:

1. Verified HM Land Registry Price Paid Data for 2022 to 2024.
2. The ONS Postcode Directory (ONSPD) postcode geography lookup.
3. The validated 2011-LSOA monthly crime features.
4. The validated 2011-LSOA monthly claimant-rate features.

The output contains one row per Bristol property transaction. Bristol is
selected with local-authority code E06000023, rather than the Land Registry
district text. Land Registry files are processed in chunks so the script does
not need to hold every England and Wales transaction in memory at once.
"""

from collections import defaultdict
from pathlib import Path

import pandas as pd


PROJECT_FOLDER = Path(__file__).resolve().parent
INPUT_FOLDER = PROJECT_FOLDER / "upload"
VERIFIED_LAND_REGISTRY_FOLDER = (
    PROJECT_FOLDER / "verified_sources" / "land_registry"
)
OUTPUT_FOLDER = PROJECT_FOLDER / "outputs"
CANDIDATE_FOLDER = OUTPUT_FOLDER / "candidate"

POSTCODE_LOOKUP_FILE = (
    PROJECT_FOLDER / "verified_sources" / "ONSPD_MAY_2026_UK_BS_complete.csv"
)
CRIME_FEATURE_FILE = (
    CANDIDATE_FOLDER / "crime_features_2022_2024_lsoa11_candidate.csv"
)
CLAIMANT_FEATURE_FILE = (
    CANDIDATE_FOLDER / "claimant_features_2022_2024_lsoa11_candidate.csv"
)

OUTPUT_2024 = CANDIDATE_FOLDER / "bristol_house_prices_2024_candidate.csv"
OUTPUT_ALL_YEARS = (
    CANDIDATE_FOLDER / "bristol_house_prices_2022_2024_candidate.csv"
)

BRISTOL_LAD_CODE = "E06000023"
CHUNK_SIZE = 150_000

LAND_REGISTRY_FILES = [
    (2022, VERIFIED_LAND_REGISTRY_FOLDER / "pp-2022.csv"),
    (2023, VERIFIED_LAND_REGISTRY_FOLDER / "pp-2023.csv"),
    (2024, VERIFIED_LAND_REGISTRY_FOLDER / "pp-2024.csv"),
]

LAND_REGISTRY_COLUMNS = [
    "transaction_id",
    "price",
    "date_of_transfer",
    "postcode",
    "property_type",
    "old_new",
    "duration",
    "paon",
    "saon",
    "street",
    "locality",
    "town_city",
    "district",
    "county",
    "ppd_category",
    "record_status",
]

LOOKUP_COLUMNS = [
    "pcds",
    "dointr",
    "doterm",
    "lad25cd",
    "lsoa11cd",
    "msoa11cd",
    "lsoa21cd",
    "msoa21cd",
    "lat",
    "long",
]


def clean_postcode(series: pd.Series) -> pd.Series:
    """Convert postcodes to uppercase and remove all spaces."""
    return (
        series.astype("string")
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", "", regex=True)
    )


def load_postcode_lookup() -> pd.DataFrame:
    """Load and validate the postcode-to-geography mapping."""
    lookup = pd.read_csv(
        POSTCODE_LOOKUP_FILE,
        usecols=LOOKUP_COLUMNS,
        dtype={
            "pcds": "string",
            "dointr": "string",
            "doterm": "string",
            "lad25cd": "string",
            "lsoa11cd": "string",
            "msoa11cd": "string",
            "lsoa21cd": "string",
            "msoa21cd": "string",
        },
        low_memory=False,
    )
    lookup["postcode_clean"] = clean_postcode(lookup["pcds"])

    duplicate_postcodes = lookup["postcode_clean"].duplicated().sum()
    if duplicate_postcodes:
        raise ValueError(
            f"Postcode lookup has {duplicate_postcodes:,} duplicate postcodes."
        )

    bristol_lookup = lookup[lookup["lad25cd"].eq(BRISTOL_LAD_CODE)]
    print(f"Postcode lookup rows: {len(lookup):,}")
    print(f"Postcodes assigned to Bristol LAD: {len(bristol_lookup):,}")
    print(
        "Bristol lookup rows missing LSOA11: "
        f"{bristol_lookup['lsoa11cd'].isna().sum():,}"
    )
    print(
        "Bristol lookup rows missing LSOA21: "
        f"{bristol_lookup['lsoa21cd'].isna().sum():,}\n"
    )
    return lookup


def new_year_stats() -> defaultdict[str, int]:
    """Create an empty set of row-count checks for one year."""
    return defaultdict(int)


def load_bristol_transactions(lookup: pd.DataFrame) -> pd.DataFrame:
    """Load Land Registry files in chunks and retain authoritative Bristol rows."""
    bristol_chunks = []
    stats: dict[int, defaultdict[str, int]] = {
        2022: new_year_stats(),
        2023: new_year_stats(),
        2024: new_year_stats(),
    }

    for expected_year, path in LAND_REGISTRY_FILES:
        if not path.exists():
            raise FileNotFoundError(f"Missing Land Registry file: {path}")

        print(f"Reading {path.name} ...")
        reader = pd.read_csv(
            path,
            header=None,
            names=LAND_REGISTRY_COLUMNS,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk in reader:
            year_stats = stats[expected_year]
            year_stats["input_rows"] += len(chunk)
            year_stats["missing_property_postcode"] += chunk["postcode"].isna().sum()

            chunk["date_of_transfer"] = pd.to_datetime(
                chunk["date_of_transfer"],
                format="mixed",
                dayfirst=True,
                errors="coerce",
            )
            invalid_dates = chunk["date_of_transfer"].isna()
            wrong_year = chunk["date_of_transfer"].dt.year.ne(expected_year)
            year_stats["invalid_dates"] += invalid_dates.sum()
            year_stats["wrong_year"] += (wrong_year & ~invalid_dates).sum()

            if invalid_dates.any() or (wrong_year & ~invalid_dates).any():
                raise ValueError(f"Date validation failed in {path.name}.")

            chunk["postcode_clean"] = clean_postcode(chunk["postcode"])
            preliminary_bristol = chunk["district"].eq("CITY OF BRISTOL")
            year_stats["district_bristol_rows"] += preliminary_bristol.sum()

            merged = chunk.merge(
                lookup,
                on="postcode_clean",
                how="left",
                validate="many_to_one",
                indicator="postcode_lookup_result",
            )
            lookup_matched = merged["postcode_lookup_result"].eq("both")
            authoritative_bristol = merged["lad25cd"].eq(BRISTOL_LAD_CODE).fillna(False)
            preliminary_bristol_merged = merged["district"].eq("CITY OF BRISTOL")

            year_stats["postcode_lookup_matches"] += lookup_matched.sum()
            year_stats["authoritative_bristol_rows"] += authoritative_bristol.sum()
            year_stats["district_bristol_not_authoritative"] += (
                preliminary_bristol_merged & ~authoritative_bristol
            ).sum()
            year_stats["authoritative_not_district_bristol"] += (
                authoritative_bristol & ~preliminary_bristol_merged
            ).sum()

            bristol = merged.loc[authoritative_bristol].copy()
            bristol_chunks.append(bristol)

    properties = pd.concat(bristol_chunks, ignore_index=True)
    duplicate_ids = properties["transaction_id"].duplicated().sum()
    if duplicate_ids:
        raise ValueError(
            f"Authoritative Bristol data has {duplicate_ids:,} duplicate transaction IDs."
        )

    for year in sorted(stats):
        values = stats[year]
        print(f"\nLand Registry {year} checks")
        print(f"  Input rows: {values['input_rows']:,}")
        print(f"  Missing property postcode: {values['missing_property_postcode']:,}")
        print(f"  Preliminary CITY OF BRISTOL rows: {values['district_bristol_rows']:,}")
        print(f"  Authoritative Bristol LAD rows: {values['authoritative_bristol_rows']:,}")
        print(
            "  District Bristol but not authoritative Bristol: "
            f"{values['district_bristol_not_authoritative']:,}"
        )
        print(
            "  Authoritative Bristol but different district text: "
            f"{values['authoritative_not_district_bristol']:,}"
        )

    print(f"\nCombined authoritative Bristol rows: {len(properties):,}")
    print(f"Duplicate transaction IDs: {duplicate_ids}")
    return properties


def add_property_features(properties: pd.DataFrame) -> pd.DataFrame:
    """Create clean dates, labels, postcode status, and 2011 crime geography."""
    data = properties.copy()
    data["sale_year"] = data["date_of_transfer"].dt.year
    data["sale_month"] = data["date_of_transfer"].dt.strftime("%Y-%m")
    data["postcode"] = data["postcode_clean"]

    property_type_labels = {
        "D": "Detached",
        "S": "Semi-detached",
        "T": "Terraced",
        "F": "Flat/Maisonette",
        "O": "Other",
    }
    data["property_type_label"] = data["property_type"].map(property_type_labels)
    data["postcode_status"] = data["doterm"].notna().map(
        {True: "Terminated", False: "Active"}
    )

    data["crime_lsoa_vintage"] = "LSOA11"
    data["crime_lsoa_code"] = data["lsoa11cd"]
    data["analysis_lsoa_code"] = data["lsoa11cd"]

    data = data.rename(
        columns={
            "lad25cd": "lad_code",
            "lat": "latitude",
            "long": "longitude",
            "dointr": "postcode_introduction_yyyymm",
            "doterm": "postcode_termination_yyyymm",
        }
    )
    return data


def join_crime_features(properties: pd.DataFrame) -> pd.DataFrame:
    """Join monthly crime counts using LSOA code and sale month."""
    crime = pd.read_csv(
        CRIME_FEATURE_FILE,
        dtype={"lsoa_code": "string", "month": "string"},
        low_memory=False,
    )
    if crime.duplicated(["lsoa_code", "month"]).any():
        raise ValueError("Crime features contain duplicate LSOA-month keys.")

    before_rows = len(properties)
    combined = properties.merge(
        crime,
        left_on=["crime_lsoa_code", "sale_month"],
        right_on=["lsoa_code", "month"],
        how="left",
        validate="many_to_one",
    )
    if len(combined) != before_rows:
        raise ValueError("Crime join changed the number of property transactions.")

    crime_count_columns = [
        column for column in crime.columns if column.endswith("_count")
    ]
    combined["crime_data_matched"] = combined["total_crime"].notna()
    missing_matches = (~combined["crime_data_matched"]).sum()
    if missing_matches:
        raise ValueError(
            f"Crime data did not match {missing_matches:,} property transactions."
        )
    combined[["total_crime", *crime_count_columns]] = combined[
        ["total_crime", *crime_count_columns]
    ].astype("int64")

    match_summary = combined.groupby("sale_year")["crime_data_matched"].agg(
        ["sum", "count"]
    )
    match_summary["match_percent"] = (
        match_summary["sum"] / match_summary["count"] * 100
    )
    print("\nCrime join matches by year")
    print(match_summary.to_string(float_format=lambda value: f"{value:.2f}"))
    return combined


def join_claimant_features(properties: pd.DataFrame) -> pd.DataFrame:
    """Join monthly claimant counts and rates using 2011 LSOA and sale month."""
    claimant = pd.read_csv(
        CLAIMANT_FEATURE_FILE,
        dtype={"lsoa11cd": "string", "month": "string"},
        low_memory=False,
    ).rename(
        columns={
            "lsoa11cd": "claimant_lsoa_code",
            "month": "claimant_month",
        }
    )
    if claimant.duplicated(["claimant_lsoa_code", "claimant_month"]).any():
        raise ValueError("Claimant features contain duplicate LSOA-month keys.")

    before_rows = len(properties)
    combined = properties.merge(
        claimant,
        left_on=["analysis_lsoa_code", "sale_month"],
        right_on=["claimant_lsoa_code", "claimant_month"],
        how="left",
        validate="many_to_one",
    )
    if len(combined) != before_rows:
        raise ValueError("Claimant join changed the number of property transactions.")

    claimant_columns = [
        "claimant_count",
        "working_age_population_2020",
        "claimant_rate_percent",
    ]
    combined["claimant_data_matched"] = combined[claimant_columns].notna().all(axis=1)
    missing_matches = (~combined["claimant_data_matched"]).sum()
    if missing_matches:
        raise ValueError(
            f"Claimant data did not match {missing_matches:,} property transactions."
        )
    combined["claimant_count"] = combined["claimant_count"].astype("int64")
    combined["working_age_population_2020"] = combined[
        "working_age_population_2020"
    ].astype("int64")
    combined["population_reference_year"] = combined[
        "population_reference_year"
    ].astype("int64")

    match_summary = combined.groupby("sale_year")["claimant_data_matched"].agg(
        ["sum", "count"]
    )
    match_summary["match_percent"] = (
        match_summary["sum"] / match_summary["count"] * 100
    )
    print("\nClaimant join matches by year")
    print(match_summary.to_string(float_format=lambda value: f"{value:.2f}"))
    return combined


def select_output_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Keep the final, clearly named modelling and audit columns."""
    crime_count_columns = sorted(
        column
        for column in data.columns
        if column.endswith("_count") and column != "claimant_count"
    )
    output_columns = [
        "transaction_id",
        "price",
        "date_of_transfer",
        "sale_year",
        "sale_month",
        "postcode",
        "property_type",
        "property_type_label",
        "old_new",
        "duration",
        "paon",
        "saon",
        "street",
        "locality",
        "town_city",
        "district",
        "county",
        "ppd_category",
        "record_status",
        "lad_code",
        "lsoa11cd",
        "msoa11cd",
        "lsoa21cd",
        "msoa21cd",
        "crime_lsoa_vintage",
        "crime_lsoa_code",
        "analysis_lsoa_code",
        "latitude",
        "longitude",
        "postcode_status",
        "postcode_introduction_yyyymm",
        "postcode_termination_yyyymm",
        "crime_data_matched",
        "total_crime",
        *crime_count_columns,
        "claimant_data_matched",
        "claimant_count",
        "working_age_population_2020",
        "claimant_rate_percent",
        "population_reference_year",
    ]
    result = data[output_columns].copy()
    result = result.sort_values(["date_of_transfer", "transaction_id"]).reset_index(
        drop=True
    )
    return result


def validate_and_save(data: pd.DataFrame, path: Path) -> None:
    """Run final checks and save a combined CSV."""
    duplicate_ids = data["transaction_id"].duplicated().sum()
    duplicate_keys = data.duplicated(["transaction_id"]).sum()
    if duplicate_ids or duplicate_keys:
        raise ValueError(f"Duplicate transaction rows found before saving {path.name}.")

    if not data["lad_code"].eq(BRISTOL_LAD_CODE).all():
        raise ValueError(f"Non-Bristol LAD rows found before saving {path.name}.")

    print(f"\nFinal checks for {path.name}")
    print(f"  Rows: {len(data):,}")
    print(f"  Duplicate transaction IDs: {duplicate_ids}")
    print(f"  Date range: {data['date_of_transfer'].min().date()} to "
          f"{data['date_of_transfer'].max().date()}")
    print(f"  Missing crime joins: {(~data['crime_data_matched']).sum():,}")
    print(f"  Missing claimant joins: {(~data['claimant_data_matched']).sum():,}")
    print("  Missing values in key columns:")
    key_columns = [
        "price",
        "date_of_transfer",
        "postcode",
        "lsoa11cd",
        "lsoa21cd",
        "crime_lsoa_code",
        "analysis_lsoa_code",
        "total_crime",
        "claimant_count",
        "working_age_population_2020",
        "claimant_rate_percent",
    ]
    print(data[key_columns].isna().sum().to_string())

    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(path, index=False, date_format="%Y-%m-%d")
    print(f"  Saved: {path}")


def main() -> None:
    """Run the complete Bristol property and crime data pipeline."""
    lookup = load_postcode_lookup()
    properties = load_bristol_transactions(lookup)
    properties = add_property_features(properties)
    combined = join_crime_features(properties)
    combined = join_claimant_features(combined)
    combined = select_output_columns(combined)

    combined_2024 = combined[combined["sale_year"].eq(2024)].copy()
    validate_and_save(combined_2024, OUTPUT_2024)
    validate_and_save(combined, OUTPUT_ALL_YEARS)


if __name__ == "__main__":
    main()
