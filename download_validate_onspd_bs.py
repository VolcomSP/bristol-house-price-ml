"""Download and validate the complete May 2026 ONSPD BS-postcode subset."""

from __future__ import annotations

import csv
import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CURRENT_FILE = PROJECT_ROOT / "upload" / "ONSPD_MAY_2026_UK_BS.csv"
OUTPUT_FILE = (
    PROJECT_ROOT
    / "verified_sources"
    / "ONSPD_MAY_2026_UK_BS_complete.csv"
)
BOUNDARY_FILE = (
    PROJECT_ROOT
    / "verified_sources"
    / "bristol_lsoa11_boundaries_bfc_v3.geojson"
)
SERVICE_URL = (
    "https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
    "ONS_Postcode_Directory_%28May_2026%29_for_the_United_Kingdom_"
    "%28Hosted_Table%29/FeatureServer/0/query"
)
WHERE = "pcd7 LIKE 'BS%'"
PAGE_SIZE = 1_000


def request_json(parameters: dict[str, str], attempts: int = 4) -> dict:
    """Submit an ArcGIS query and return its decoded JSON response."""
    payload = urllib.parse.urlencode({**parameters, "f": "json"}).encode("utf-8")
    request = urllib.request.Request(SERVICE_URL, data=payload, method="POST")

    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                result = json.load(response)
            if "error" in result:
                raise RuntimeError(result["error"])
            return result
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(attempt * 2)

    raise RuntimeError("ArcGIS query did not return a response.")


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read a CSV while preserving postcode codes as text."""
    with path.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def normalise(value: object) -> str:
    """Convert API nulls and CSV blanks to the same text representation."""
    return "" if value is None else str(value)


def comparison_value(value: object) -> str:
    """Normalise harmless integer padding used by the hosted-table importer."""
    text = normalise(value).strip()
    if re.fullmatch(r"-?\d+", text):
        return str(int(text))
    return text


def download_batch(
    object_ids: list[int], current_fields: list[str]
) -> list[dict[str, str]]:
    """Download one independent object-ID batch."""
    result = request_json(
        {
            "objectIds": ",".join(map(str, object_ids)),
            "outFields": "*",
            "returnGeometry": "false",
        }
    )
    return [
        {
            field: normalise(feature["attributes"].get(field))
            for field in current_fields
        }
        for feature in result.get("features", [])
    ]


def main() -> None:
    current_fields, current_rows = read_csv(CURRENT_FILE)
    if not current_fields:
        raise ValueError("The local ONSPD file has no header.")

    count_result = request_json(
        {"where": WHERE, "returnCountOnly": "true"}
    )
    expected_count = int(count_result["count"])

    id_result = request_json(
        {"where": WHERE, "returnIdsOnly": "true"}
    )
    object_id_field = id_result["objectIdFieldName"]
    object_ids = sorted(int(value) for value in id_result["objectIds"])
    if len(object_ids) != expected_count or len(set(object_ids)) != expected_count:
        raise ValueError("The official record count and object-ID list do not agree.")

    batches = [
        object_ids[start : start + PAGE_SIZE]
        for start in range(0, len(object_ids), PAGE_SIZE)
    ]
    downloaded: list[dict[str, str]] = []
    completed_records = 0
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(download_batch, batch, current_fields): len(batch)
            for batch in batches
        }
        for future in as_completed(futures):
            downloaded.extend(future.result())
            completed_records += futures[future]
            print(
                f"Downloaded {completed_records:,} of {expected_count:,} records"
            )

    if len(downloaded) != expected_count:
        raise ValueError(
            f"Downloaded {len(downloaded):,} rows; expected {expected_count:,}."
        )

    downloaded.sort(key=lambda row: row["pcd7"])
    postcodes = [row["pcds"].strip() for row in downloaded]
    if any(not postcode.startswith("BS") for postcode in postcodes):
        raise ValueError("The official extract contains a non-BS postcode.")
    if len(set(postcodes)) != expected_count:
        raise ValueError("The official extract contains duplicate postcodes.")

    bristol_rows = [
        row for row in downloaded if row["lad25cd"] == "E06000023"
    ]
    bristol_required_fields = [
        "pcds",
        "dointr",
        "lad25cd",
        "lsoa11cd",
        "msoa11cd",
        "lsoa21cd",
        "msoa21cd",
        "lat",
        "long",
    ]
    bristol_missing_values = sum(
        not row[field]
        for row in bristol_rows
        for field in bristol_required_fields
    )
    boundary = json.loads(BOUNDARY_FILE.read_text(encoding="utf-8"))
    boundary_codes = {
        feature["properties"]["LSOA11CD"] for feature in boundary["features"]
    }
    lookup_codes = {row["lsoa11cd"] for row in bristol_rows}
    if bristol_missing_values:
        raise ValueError(
            f"Bristol rows contain {bristol_missing_values:,} missing required values."
        )
    if lookup_codes != boundary_codes:
        raise ValueError("Bristol 2011 LSOA codes do not match the boundary file.")

    complete_by_postcode = {row["pcds"].strip(): row for row in downloaded}
    missing_current = [
        row["pcds"].strip()
        for row in current_rows
        if row["pcds"].strip() not in complete_by_postcode
    ]
    raw_differences_by_field = {field: 0 for field in current_fields}
    semantic_differences_by_field = {field: 0 for field in current_fields}
    geography_fields = [
        "lad25cd",
        "lsoa11cd",
        "msoa11cd",
        "lsoa21cd",
        "msoa21cd",
    ]
    geography_difference_postcodes: set[str] = set()
    latitude_differences: list[float] = []
    longitude_differences: list[float] = []
    for current in current_rows:
        postcode = current["pcds"].strip()
        complete = complete_by_postcode.get(postcode)
        if complete is None:
            continue
        for field in current_fields:
            if normalise(current[field]) != normalise(complete[field]):
                raw_differences_by_field[field] += 1
            if comparison_value(current[field]) != comparison_value(complete[field]):
                semantic_differences_by_field[field] += 1
        if any(
            comparison_value(current[field]) != comparison_value(complete[field])
            for field in geography_fields
        ):
            geography_difference_postcodes.add(postcode)
        if current["lat"] and complete["lat"]:
            latitude_differences.append(
                abs(float(current["lat"]) - float(complete["lat"]))
            )
        if current["long"] and complete["long"]:
            longitude_differences.append(
                abs(float(current["long"]) - float(complete["long"]))
            )

    raw_differences_by_field = {
        field: count for field, count in raw_differences_by_field.items() if count
    }
    semantic_differences_by_field = {
        field: count
        for field, count in semantic_differences_by_field.items()
        if count
    }

    if missing_current:
        raise ValueError(
            f"The official table is missing {len(missing_current):,} local postcodes."
        )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=current_fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(downloaded)

    print(f"Object-ID field used for retrieval: {object_id_field}")
    print(f"Current local rows: {len(current_rows):,}")
    print(f"Complete official BS rows: {len(downloaded):,}")
    print(f"Bristol LAD rows: {len(bristol_rows):,}")
    print(f"Bristol 2011 LSOAs: {len(lookup_codes):,}")
    print(f"Bristol required-field blanks: {bristol_missing_values:,}")
    print(f"Rows absent from the local copy: {len(downloaded) - len(current_rows):,}")
    print(f"Current rows absent from official extract: {len(missing_current):,}")
    print(f"Raw formatting differences by field: {raw_differences_by_field}")
    print(
        "Semantic overlapping-field differences: "
        f"{semantic_differences_by_field}"
    )
    print(
        "Postcodes with changed core geography: "
        f"{sorted(geography_difference_postcodes)}"
    )
    print(
        "Maximum latitude difference: "
        f"{max(latitude_differences, default=0):.9f} degrees"
    )
    print(
        "Maximum longitude difference: "
        f"{max(longitude_differences, default=0):.9f} degrees"
    )
    print(f"First postcode: {downloaded[0]['pcds']}")
    print(f"Last postcode: {downloaded[-1]['pcds']}")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
