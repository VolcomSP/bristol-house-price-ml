"""Validate one complete HM Land Registry annual Price Paid CSV."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_FOLDER = PROJECT_ROOT / "verified_sources" / "land_registry"
POSTCODE_LOOKUP = (
    PROJECT_ROOT / "verified_sources" / "ONSPD_MAY_2026_UK_BS_complete.csv"
)
MANIFEST_FOLDER = SOURCE_FOLDER
OUTPUT_ALL_YEARS = (
    PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2022_2024.csv"
)
OUTPUT_2024 = PROJECT_ROOT / "outputs" / "bristol_house_prices_combined_2024.csv"
BRISTOL_LAD_CODE = "E06000023"

SOURCE_URLS = {
    2022: "https://price-paid-data.publicdata.landregistry.gov.uk/pp-2022.csv",
    2023: "https://price-paid-data.publicdata.landregistry.gov.uk/pp-2023.csv",
    2024: "https://price-paid-data.publicdata.landregistry.gov.uk/pp-2024.csv",
}


def sha256(path: Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def remote_metadata(url: str) -> dict[str, str]:
    """Read authoritative HTTP metadata for the current official file."""
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return {
            "content_length": response.headers.get("Content-Length", ""),
            "content_type": response.headers.get("Content-Type", ""),
            "last_modified": response.headers.get("Last-Modified", ""),
            "etag": response.headers.get("ETag", ""),
        }


def bristol_postcodes() -> set[str]:
    """Return cleaned postcodes assigned to the City of Bristol LAD."""
    lookup = pd.read_csv(
        POSTCODE_LOOKUP,
        usecols=["pcds", "lad25cd"],
        dtype="string",
        keep_default_na=False,
    )
    lookup["postcode_clean"] = (
        lookup["pcds"].str.upper().str.replace(" ", "", regex=False).str.strip()
    )
    return set(
        lookup.loc[
            lookup["lad25cd"].eq(BRISTOL_LAD_CODE), "postcode_clean"
        ]
    )


def existing_output_ids(year: int) -> set[str]:
    """Return transaction IDs from the previously produced output for one year."""
    path = OUTPUT_2024 if year == 2024 else OUTPUT_ALL_YEARS
    if not path.is_file():
        return set()
    output = pd.read_csv(
        path,
        usecols=["transaction_id", "sale_year"],
        dtype="string",
        keep_default_na=False,
    )
    return set(output.loc[output["sale_year"].eq(str(year)), "transaction_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int, choices=sorted(SOURCE_URLS))
    args = parser.parse_args()

    year = args.year
    path = SOURCE_FOLDER / f"pp-{year}.csv"
    if not path.is_file():
        raise FileNotFoundError(path)

    source_metadata = remote_metadata(SOURCE_URLS[year])
    expected_bytes = int(source_metadata["content_length"])
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"File size mismatch: local {actual_bytes:,}; remote {expected_bytes:,}."
        )

    bristol_codes = bristol_postcodes()
    transaction_ids: set[str] = set()
    bristol_ids: set[str] = set()
    monthly_counts: Counter[str] = Counter()
    total_rows = 0
    malformed_rows = 0
    duplicate_ids = 0
    wrong_year_rows = 0
    missing_postcodes = 0
    invalid_prices = 0
    district_bristol_rows = 0

    with path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.reader(source):
            total_rows += 1
            if len(row) != 16:
                malformed_rows += 1
                continue

            transaction_id, price, date, postcode = row[:4]
            if transaction_id in transaction_ids:
                duplicate_ids += 1
            transaction_ids.add(transaction_id)
            if not price.isdigit():
                invalid_prices += 1
            if not date.startswith(f"{year}-"):
                wrong_year_rows += 1
            else:
                monthly_counts[date[:7]] += 1
            if not postcode:
                missing_postcodes += 1

            postcode_clean = postcode.upper().replace(" ", "").strip()
            if postcode_clean in bristol_codes:
                bristol_ids.add(transaction_id)
            if row[13] == "CITY OF BRISTOL":
                district_bristol_rows += 1

    with path.open("rb") as source:
        source.seek(-1, 2)
        ends_with_newline = source.read(1) in {b"\n", b"\r"}
    month_keys = sorted(monthly_counts)
    expected_months = [f"{year}-{month:02d}" for month in range(1, 13)]

    failures = {
        "malformed_rows": malformed_rows,
        "duplicate_transaction_ids": duplicate_ids,
        "wrong_year_rows": wrong_year_rows,
        "invalid_prices": invalid_prices,
        "missing_months": sorted(set(expected_months) - set(month_keys)),
        "ends_with_newline": ends_with_newline,
    }
    if (
        malformed_rows
        or duplicate_ids
        or wrong_year_rows
        or invalid_prices
        or failures["missing_months"]
        or not ends_with_newline
    ):
        raise ValueError(f"Annual Price Paid validation failed: {failures}")

    file_hash = sha256(path)
    previous_ids = existing_output_ids(year)
    source_not_previous = sorted(bristol_ids - previous_ids)
    previous_not_source = sorted(previous_ids - bristol_ids)
    manifest = {
        "source": "HM Land Registry Price Paid Data yearly file",
        "source_url": SOURCE_URLS[year],
        "retrieved_and_validated_utc": datetime.now(timezone.utc).isoformat(),
        "year": year,
        "file_name": path.name,
        "bytes": actual_bytes,
        "sha256": file_hash,
        "remote_metadata": source_metadata,
        "rows": total_rows,
        "unique_transaction_ids": len(transaction_ids),
        "malformed_rows": malformed_rows,
        "duplicate_transaction_ids": duplicate_ids,
        "wrong_year_rows": wrong_year_rows,
        "invalid_prices": invalid_prices,
        "missing_postcodes": missing_postcodes,
        "months": month_keys,
        "monthly_row_counts": dict(sorted(monthly_counts.items())),
        "bristol_lad_rows_by_postcode_lookup": len(bristol_ids),
        "city_of_bristol_district_rows": district_bristol_rows,
        "previous_output_rows": len(previous_ids),
        "source_bristol_ids_absent_from_previous_output": source_not_previous,
        "previous_output_ids_absent_from_source": previous_not_source,
        "ends_with_newline": ends_with_newline,
    }
    manifest_path = MANIFEST_FOLDER / f"pp-{year}.validation.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"Year: {year}")
    print(f"Bytes: {actual_bytes:,}")
    print(f"Rows: {total_rows:,}")
    print(f"Unique transaction IDs: {len(transaction_ids):,}")
    print(f"Malformed rows: {malformed_rows:,}")
    print(f"Duplicate transaction IDs: {duplicate_ids:,}")
    print(f"Wrong-year rows: {wrong_year_rows:,}")
    print(f"Missing postcodes: {missing_postcodes:,}")
    print(f"Months: {month_keys[0]} to {month_keys[-1]}")
    print(f"Bristol LAD rows from postcode lookup: {len(bristol_ids):,}")
    print(f"CITY OF BRISTOL district rows: {district_bristol_rows:,}")
    print(f"Previous output rows for year: {len(previous_ids):,}")
    print(
        "Source Bristol IDs absent from previous output: "
        f"{len(source_not_previous):,}"
    )
    print(
        "Previous output IDs absent from current source: "
        f"{len(previous_not_source):,}"
    )
    print(f"Ends with newline: {ends_with_newline}")
    print(f"SHA-256: {file_hash}")
    print(f"Validation manifest: {manifest_path}")


if __name__ == "__main__":
    main()
