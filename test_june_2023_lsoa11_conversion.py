"""Validate the June 2023 crime conversion to 2011 Bristol LSOAs.

June 2023 is the first project month in which the Police.uk source contains
2021 LSOA codes. This checkpoint assigns every usable coordinate to the
full-resolution 2011 Bristol boundary, reconciles the source and spatial
Bristol classifications, and verifies the resulting totals. It does not write
or modify any crime data.
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from test_lsoa11_assignment import assign_lsoa, load_boundaries, wgs84_to_bng


PROJECT_ROOT = Path(__file__).resolve().parent
CRIME_FILE = PROJECT_ROOT / "upload" / "2023-06-avon-and-somerset-street.csv"
EXPECTED_MONTH = "2023-06"
EXPECTED_FORCE = "Avon and Somerset Constabulary"

REQUIRED_COLUMNS = {
    "Crime ID",
    "Month",
    "Reported by",
    "Longitude",
    "Latitude",
    "LSOA code",
    "LSOA name",
    "Crime type",
}


def valid_coordinates(row: dict[str, str]) -> bool:
    """Return whether a source row contains numeric longitude and latitude."""
    try:
        float(row["Longitude"])
        float(row["Latitude"])
    except ValueError:
        return False
    return True


def load_and_validate() -> list[dict[str, str]]:
    """Load the June file and validate its schema, month, and police force."""
    with CRIME_FILE.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        rows = list(reader)

    months = {row["Month"] for row in rows}
    forces = {row["Reported by"] for row in rows}
    if months != {EXPECTED_MONTH}:
        raise ValueError(f"Unexpected months: {sorted(months)}")
    if forces != {EXPECTED_FORCE}:
        raise ValueError(f"Unexpected forces: {sorted(forces)}")
    return rows


def remove_repeated_crime_ids(
    rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    """Apply the same monthly Crime ID rule used by the existing pipeline."""
    anonymous_rows: list[tuple[int, dict[str, str]]] = []
    best_identified_rows: dict[str, tuple[tuple[bool, bool, int], dict[str, str]]] = {}
    nonblank_id_rows = 0

    for source_order, row in enumerate(rows):
        crime_id = row["Crime ID"].strip()
        if not crime_id:
            anonymous_rows.append((source_order, row))
            continue

        nonblank_id_rows += 1
        score = (
            bool(row["LSOA code"].strip()),
            valid_coordinates(row),
            -source_order,
        )
        current = best_identified_rows.get(crime_id)
        if current is None or score > current[0]:
            best_identified_rows[crime_id] = (score, row)

    identified_rows = [
        (-score[2], row) for score, row in best_identified_rows.values()
    ]
    retained = [
        row for _, row in sorted([*anonymous_rows, *identified_rows], key=lambda item: item[0])
    ]
    repeated_rows = nonblank_id_rows - len(best_identified_rows)
    return retained, repeated_rows


def main() -> None:
    raw_rows = load_and_validate()
    rows, repeated_rows = remove_repeated_crime_ids(raw_rows)
    boundaries, boundary_crs = load_boundaries()
    if boundary_crs != "EPSG:27700":
        raise ValueError(f"Expected EPSG:27700 boundary; found {boundary_crs}")

    counts: Counter[str] = Counter()
    assigned_lsoas: Counter[str] = Counter()
    assigned_crime_types: Counter[str] = Counter()

    for row in rows:
        counts["cleaned_rows"] += 1
        source_is_bristol = row["LSOA name"].strip().startswith("Bristol ")
        if source_is_bristol:
            counts["source_bristol"] += 1

        if not valid_coordinates(row):
            counts["missing_coordinates"] += 1
            if source_is_bristol:
                counts["source_bristol_missing_coordinates"] += 1
            continue

        counts["valid_coordinates"] += 1
        longitude = float(row["Longitude"])
        latitude = float(row["Latitude"])
        easting, northing = wgs84_to_bng(latitude, longitude)
        matches = assign_lsoa(easting, northing, boundaries)

        if len(matches) > 1:
            counts["multiple_matches"] += 1
            continue

        spatial_is_bristol = len(matches) == 1
        if spatial_is_bristol:
            counts["spatial_bristol"] += 1
            assigned_code = matches[0]
            assigned_lsoas[assigned_code] += 1
            assigned_crime_types[row["Crime type"].strip()] += 1

        if source_is_bristol and spatial_is_bristol:
            counts["source_and_spatial_bristol"] += 1
        elif source_is_bristol:
            counts["source_bristol_only"] += 1
        elif spatial_is_bristol:
            counts["spatial_bristol_only"] += 1

    if sum(assigned_lsoas.values()) != counts["spatial_bristol"]:
        raise ValueError("LSOA aggregation does not preserve the assigned row total.")
    if sum(assigned_crime_types.values()) != counts["spatial_bristol"]:
        raise ValueError("Crime-type aggregation does not preserve the assigned row total.")
    if counts["multiple_matches"]:
        raise ValueError("At least one crime coordinate matched multiple 2011 LSOAs.")

    source_spatial_difference = counts["spatial_bristol"] - counts["source_bristol"]
    coordinate_coverage = counts["valid_coordinates"] / counts["cleaned_rows"]
    bristol_overlap_rate = (
        counts["source_and_spatial_bristol"] / counts["source_bristol"]
        if counts["source_bristol"]
        else 0
    )

    print(f"Test file: {CRIME_FILE.name}")
    print(f"Raw crime rows: {len(raw_rows):,}")
    print(f"Repeated nonblank Crime ID rows removed: {repeated_rows:,}")
    print(f"Rows after Crime ID check: {counts['cleaned_rows']:,}")
    print(f"Rows with valid coordinates: {counts['valid_coordinates']:,}")
    print(f"Rows missing coordinates: {counts['missing_coordinates']:,}")
    print(f"Coordinate coverage: {coordinate_coverage:.2%}")
    print(f"Source rows labelled Bristol: {counts['source_bristol']:,}")
    print(f"Rows assigned to a 2011 Bristol LSOA: {counts['spatial_bristol']:,}")
    print(f"Source and spatial Bristol: {counts['source_and_spatial_bristol']:,}")
    print(f"Source Bristol only: {counts['source_bristol_only']:,}")
    print(f"Spatial Bristol only: {counts['spatial_bristol_only']:,}")
    print(f"Source Bristol rows missing coordinates: {counts['source_bristol_missing_coordinates']:,}")
    print(f"Net spatial/source Bristol difference: {source_spatial_difference:+,}")
    print(f"Source Bristol overlap retained: {bristol_overlap_rate:.2%}")
    print(f"Multiple boundary matches: {counts['multiple_matches']:,}")
    print(f"2011 Bristol LSOAs receiving at least one crime: {len(assigned_lsoas):,}")
    print(f"Aggregated LSOA total: {sum(assigned_lsoas.values()):,}")
    print(f"Aggregated crime-type total: {sum(assigned_crime_types.values()):,}")


if __name__ == "__main__":
    main()
