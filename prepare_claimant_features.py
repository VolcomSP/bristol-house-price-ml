"""Create monthly Bristol 2011-LSOA claimant-rate features from Nomis."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
POSTCODE_LOOKUP = (
    PROJECT_ROOT / "verified_sources" / "ONSPD_MAY_2026_UK_BS_complete.csv"
)
CITY_CONTROL = PROJECT_ROOT / "upload" / "Bristol wide monthly socioeconomic control.csv"
SOURCE_FOLDER = PROJECT_ROOT / "verified_sources" / "nomis"
OUTPUT_FOLDER = PROJECT_ROOT / "outputs" / "candidate"

CLAIMANT_RAW = SOURCE_FOLDER / "claimant_count_bristol_lsoa11_2022_2024_raw.csv"
POPULATION_RAW = SOURCE_FOLDER / "population_aged_16_64_bristol_lsoa11_2020_raw.csv"
FEATURE_OUTPUT = OUTPUT_FOLDER / "claimant_features_2022_2024_lsoa11_candidate.csv"
VALIDATION_OUTPUT = FEATURE_OUTPUT.with_suffix(".validation.json")

BRISTOL_LAD_CODE = "E06000023"
CLAIMANT_DATASET = "NM_162_1"
POPULATION_DATASET = "NM_2010_1"
NOMIS_API_ROOT = "https://www.nomisweb.co.uk/api/v01/dataset"
CLAIMANT_SOURCE_PAGE = "https://www.nomisweb.co.uk/datasets/ucjsa"
POPULATION_SOURCE_PAGE = "https://www.nomisweb.co.uk/datasets/pestsyoaoa"
EXPECTED_MONTHS = [
    f"{year}-{month:02d}"
    for year in range(2022, 2025)
    for month in range(1, 13)
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def api_url(dataset: str, params: dict[str, str | int]) -> str:
    return (
        f"{NOMIS_API_ROOT}/{dataset}.data.csv?"
        + urllib.parse.urlencode(params, safe=",")
    )


def download(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Bristol-house-price-research/1.0"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def csv_frame(content: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(content), dtype="string", keep_default_na=False)


def bristol_lsoa_codes() -> set[str]:
    lookup = pd.read_csv(
        POSTCODE_LOOKUP,
        usecols=["lad25cd", "lsoa11cd"],
        dtype="string",
        keep_default_na=False,
    )
    return set(
        lookup.loc[
            lookup["lad25cd"].eq(BRISTOL_LAD_CODE), "lsoa11cd"
        ].loc[lambda values: values.ne("")]
    )


def discover_nomis_geographies(bristol_codes: set[str]) -> pd.DataFrame:
    pages: list[pd.DataFrame] = []
    offset = 0
    while True:
        params = {
            "geography": "TYPE298",
            "date": "2022-01",
            "gender": 0,
            "age": 0,
            "measure": 1,
            "measures": 20100,
            "select": "geography,geography_code,geography_name",
            "recordoffset": offset,
        }
        page = csv_frame(download(api_url(CLAIMANT_DATASET, params)))
        pages.append(page)
        if len(page) < 25_000:
            break
        offset += 25_000

    geography = pd.concat(pages, ignore_index=True)
    geography = geography.loc[
        geography["GEOGRAPHY_CODE"].isin(bristol_codes),
        ["GEOGRAPHY", "GEOGRAPHY_CODE", "GEOGRAPHY_NAME"],
    ].drop_duplicates()
    if len(geography) != len(bristol_codes):
        missing = sorted(bristol_codes - set(geography["GEOGRAPHY_CODE"]))
        raise ValueError(f"Nomis geography discovery missed Bristol LSOAs: {missing}")
    if geography["GEOGRAPHY_CODE"].duplicated().any():
        raise ValueError("Nomis geography discovery returned duplicate Bristol LSOAs.")
    return geography.sort_values("GEOGRAPHY_CODE").reset_index(drop=True)


def download_claimant_counts(geography: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    params = {
        "geography": ",".join(geography["GEOGRAPHY"]),
        "date": ",".join(EXPECTED_MONTHS),
        "gender": 0,
        "age": 0,
        "measure": 1,
        "measures": 20100,
        "select": (
            "geography_code,geography_name,date_code,gender_name,age_name,"
            "measure_name,obs_value,obs_status_name"
        ),
    }
    url = api_url(CLAIMANT_DATASET, params)
    content = download(url)
    CLAIMANT_RAW.parent.mkdir(parents=True, exist_ok=True)
    CLAIMANT_RAW.write_bytes(content)
    return csv_frame(content), url


def download_population(geography: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    params = {
        "geography": ",".join(geography["GEOGRAPHY"]),
        "date": "2020",
        "gender": 0,
        "c_age": 203,
        "measures": 20100,
        "select": (
            "geography_code,geography_name,date_code,gender_name,c_age_name,"
            "obs_value,obs_status_name"
        ),
    }
    url = api_url(POPULATION_DATASET, params)
    content = download(url)
    POPULATION_RAW.parent.mkdir(parents=True, exist_ok=True)
    POPULATION_RAW.write_bytes(content)
    return csv_frame(content), url


def citywide_control() -> dict[str, int]:
    values: dict[str, int] = {}
    with CITY_CONTROL.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.reader(source):
            if len(row) < 2:
                continue
            try:
                month = datetime.strptime(row[0], "%b-%y").strftime("%Y-%m")
                count = int(row[1])
            except (ValueError, TypeError):
                continue
            values[month] = count
    return values


def main() -> None:
    bristol_codes = bristol_lsoa_codes()
    if len(bristol_codes) != 263:
        raise ValueError(f"Expected 263 Bristol LSOA11 codes, found {len(bristol_codes)}.")

    geography = discover_nomis_geographies(bristol_codes)
    claimant, claimant_query_url = download_claimant_counts(geography)
    population, population_query_url = download_population(geography)

    if len(claimant) != 263 * 36:
        raise ValueError(f"Expected 9,468 claimant rows, found {len(claimant):,}.")
    if len(population) != 263:
        raise ValueError(f"Expected 263 population rows, found {len(population):,}.")

    claimant = claimant.rename(
        columns={
            "GEOGRAPHY_CODE": "lsoa11cd",
            "GEOGRAPHY_NAME": "lsoa11_name",
            "DATE_CODE": "month",
            "OBS_VALUE": "claimant_count",
            "OBS_STATUS_NAME": "claimant_count_status",
        }
    )
    population = population.rename(
        columns={
            "GEOGRAPHY_CODE": "lsoa11cd",
            "OBS_VALUE": "working_age_population_2020",
            "OBS_STATUS_NAME": "population_status",
        }
    )

    claimant["claimant_count"] = pd.to_numeric(
        claimant["claimant_count"], errors="coerce"
    )
    population["working_age_population_2020"] = pd.to_numeric(
        population["working_age_population_2020"], errors="coerce"
    )

    features = claimant[
        [
            "lsoa11cd",
            "lsoa11_name",
            "month",
            "claimant_count",
            "claimant_count_status",
        ]
    ].merge(
        population[
            ["lsoa11cd", "working_age_population_2020", "population_status"]
        ],
        on="lsoa11cd",
        how="left",
        validate="many_to_one",
    )
    features["claimant_rate_percent"] = (
        features["claimant_count"]
        / features["working_age_population_2020"]
        * 100
    )
    features["population_reference_year"] = 2020
    features["claimant_source_url"] = CLAIMANT_SOURCE_PAGE
    features["population_source_url"] = POPULATION_SOURCE_PAGE
    features = features.sort_values(["month", "lsoa11cd"]).reset_index(drop=True)

    duplicate_keys = features.duplicated(["lsoa11cd", "month"]).sum()
    lsoas_per_month = features.groupby("month")["lsoa11cd"].nunique()
    months_per_lsoa = features.groupby("lsoa11cd")["month"].nunique()
    missing_counts = int(features["claimant_count"].isna().sum())
    missing_population = int(features["working_age_population_2020"].isna().sum())
    invalid_population = int((features["working_age_population_2020"] <= 0).sum())
    invalid_rates = int(
        (~features["claimant_rate_percent"].between(0, 100, inclusive="both")).sum()
    )
    if duplicate_keys:
        raise ValueError(f"Found {duplicate_keys:,} duplicate LSOA-month keys.")
    if set(features["month"]) != set(EXPECTED_MONTHS):
        raise ValueError("Claimant features do not cover the expected 36 months.")
    if not lsoas_per_month.eq(263).all() or not months_per_lsoa.eq(36).all():
        raise ValueError("Claimant feature grid is incomplete.")
    if missing_counts or missing_population or invalid_population or invalid_rates:
        raise ValueError(
            "Invalid claimant features: "
            f"missing counts={missing_counts}, missing population={missing_population}, "
            f"invalid population={invalid_population}, invalid rates={invalid_rates}."
        )

    city_control = citywide_control()
    feature_totals = features.groupby("month")["claimant_count"].sum().round().astype(int)
    reconciliation = pd.DataFrame(
        {
            "lsoa_sum": feature_totals,
            "citywide_control": pd.Series(city_control),
        }
    ).dropna()
    reconciliation["difference"] = (
        reconciliation["lsoa_sum"] - reconciliation["citywide_control"]
    )
    if len(reconciliation) != 36:
        raise ValueError("Citywide reconciliation does not cover all 36 months.")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    features.to_csv(FEATURE_OUTPUT, index=False, float_format="%.6f")

    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(features),
        "lsoa11_count": features["lsoa11cd"].nunique(),
        "month_count": features["month"].nunique(),
        "duplicate_lsoa_month_keys": int(duplicate_keys),
        "missing_claimant_counts": missing_counts,
        "missing_population_values": missing_population,
        "claimant_count_min": int(features["claimant_count"].min()),
        "claimant_count_max": int(features["claimant_count"].max()),
        "working_age_population_min": int(features["working_age_population_2020"].min()),
        "working_age_population_max": int(features["working_age_population_2020"].max()),
        "claimant_rate_percent_min": float(features["claimant_rate_percent"].min()),
        "claimant_rate_percent_max": float(features["claimant_rate_percent"].max()),
        "population_reference_year": 2020,
        "population_method_note": (
            "The 2020 mid-year estimate is the final official population estimate "
            "published on 2011 LSOA boundaries and is held constant for 2022-2024."
        ),
        "rounding_note": (
            "Nomis claimant counts are rounded to the nearest five, so the sum of "
            "LSOA values can differ slightly from the separately rounded city total."
        ),
        "citywide_reconciliation": {
            "maximum_absolute_monthly_difference": int(
                reconciliation["difference"].abs().max()
            ),
            "mean_absolute_monthly_difference": float(
                reconciliation["difference"].abs().mean()
            ),
            "monthly_differences": {
                month: int(value)
                for month, value in reconciliation["difference"].items()
            },
        },
        "claimant_dataset": CLAIMANT_DATASET,
        "claimant_source_page": CLAIMANT_SOURCE_PAGE,
        "claimant_query_url": claimant_query_url,
        "population_dataset": POPULATION_DATASET,
        "population_source_page": POPULATION_SOURCE_PAGE,
        "population_query_url": population_query_url,
        "files": {
            CLAIMANT_RAW.name: sha256(CLAIMANT_RAW),
            POPULATION_RAW.name: sha256(POPULATION_RAW),
            FEATURE_OUTPUT.name: sha256(FEATURE_OUTPUT),
        },
    }
    VALIDATION_OUTPUT.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Claimant rows: {len(features):,}")
    print(f"LSOA11 areas: {features['lsoa11cd'].nunique():,}")
    print(f"Months: {features['month'].nunique():,}")
    print(f"Missing claimant counts: {missing_counts:,}")
    print(f"Missing population values: {missing_population:,}")
    print(
        "Claimant rate range: "
        f"{features['claimant_rate_percent'].min():.2f}% to "
        f"{features['claimant_rate_percent'].max():.2f}%"
    )
    print(
        "Maximum monthly difference from Bristol city total: "
        f"{reconciliation['difference'].abs().max():.0f}"
    )
    print(f"Saved: {FEATURE_OUTPUT}")
    print(f"Validation: {VALIDATION_OUTPUT}")


if __name__ == "__main__":
    main()
