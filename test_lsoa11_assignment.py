"""Test crime-coordinate assignment to 2011 Bristol LSOAs for one month.

This is a validation checkpoint only. It reads the January 2022 Police.uk CSV
and the official full-resolution Bristol 2011 LSOA boundary file, performs the
spatial match, and prints reconciliation statistics. It does not write or
modify data files.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
CRIME_FILE = PROJECT_ROOT / "upload" / "2022-01-avon-and-somerset-street.csv"
BOUNDARY_FILE = (
    PROJECT_ROOT
    / "verified_sources"
    / "bristol_lsoa11_boundaries_bfc_v3.geojson"
)


def geodetic_to_cartesian(
    latitude: float,
    longitude: float,
    height: float,
    major_axis: float,
    minor_axis: float,
) -> tuple[float, float, float]:
    """Convert ellipsoidal latitude/longitude to Cartesian coordinates."""
    eccentricity_squared = 1 - (minor_axis * minor_axis) / (major_axis * major_axis)
    sin_latitude = math.sin(latitude)
    nu = major_axis / math.sqrt(1 - eccentricity_squared * sin_latitude**2)
    x = (nu + height) * math.cos(latitude) * math.cos(longitude)
    y = (nu + height) * math.cos(latitude) * math.sin(longitude)
    z = ((1 - eccentricity_squared) * nu + height) * sin_latitude
    return x, y, z


def cartesian_to_geodetic(
    x: float,
    y: float,
    z: float,
    major_axis: float,
    minor_axis: float,
) -> tuple[float, float]:
    """Convert Cartesian coordinates to ellipsoidal latitude/longitude."""
    eccentricity_squared = 1 - (minor_axis * minor_axis) / (major_axis * major_axis)
    horizontal_distance = math.hypot(x, y)
    latitude = math.atan2(z, horizontal_distance * (1 - eccentricity_squared))

    for _ in range(12):
        nu = major_axis / math.sqrt(
            1 - eccentricity_squared * math.sin(latitude) ** 2
        )
        updated = math.atan2(
            z + eccentricity_squared * nu * math.sin(latitude),
            horizontal_distance,
        )
        if abs(updated - latitude) < 1e-12:
            latitude = updated
            break
        latitude = updated

    longitude = math.atan2(y, x)
    return latitude, longitude


def wgs84_to_osgb36(latitude_degrees: float, longitude_degrees: float) -> tuple[float, float]:
    """Transform WGS84 coordinates to OSGB36 with the standard Helmert transform."""
    wgs84_a = 6_378_137.0
    wgs84_b = 6_356_752.3141
    airy_a = 6_377_563.396
    airy_b = 6_356_256.909

    latitude = math.radians(latitude_degrees)
    longitude = math.radians(longitude_degrees)
    x, y, z = geodetic_to_cartesian(
        latitude, longitude, 0.0, wgs84_a, wgs84_b
    )

    # WGS84 to OSGB36 transform parameters (metres, arcseconds, ppm).
    tx, ty, tz = -446.448, 125.157, -542.060
    rx = math.radians(-0.1502 / 3600)
    ry = math.radians(-0.2470 / 3600)
    rz = math.radians(-0.8421 / 3600)
    scale = 1 + 20.4894e-6

    transformed_x = tx + x * scale - y * rz + z * ry
    transformed_y = ty + x * rz + y * scale - z * rx
    transformed_z = tz - x * ry + y * rx + z * scale
    return cartesian_to_geodetic(
        transformed_x, transformed_y, transformed_z, airy_a, airy_b
    )


def osgb36_to_bng(latitude: float, longitude: float) -> tuple[float, float]:
    """Project OSGB36 latitude/longitude to British National Grid metres."""
    major_axis = 6_377_563.396
    minor_axis = 6_356_256.909
    scale = 0.9996012717
    latitude_origin = math.radians(49)
    longitude_origin = math.radians(-2)
    northing_origin = -100_000.0
    easting_origin = 400_000.0

    eccentricity_squared = 1 - (minor_axis * minor_axis) / (major_axis * major_axis)
    n = (major_axis - minor_axis) / (major_axis + minor_axis)
    sin_latitude = math.sin(latitude)
    cos_latitude = math.cos(latitude)
    tan_latitude = math.tan(latitude)
    nu = major_axis * scale / math.sqrt(
        1 - eccentricity_squared * sin_latitude**2
    )
    rho = (
        major_axis
        * scale
        * (1 - eccentricity_squared)
        / (1 - eccentricity_squared * sin_latitude**2) ** 1.5
    )
    eta_squared = nu / rho - 1

    latitude_difference = latitude - latitude_origin
    latitude_sum = latitude + latitude_origin
    meridional_arc = minor_axis * scale * (
        (1 + n + 5 / 4 * n**2 + 5 / 4 * n**3) * latitude_difference
        - (3 * n + 3 * n**2 + 21 / 8 * n**3)
        * math.sin(latitude_difference)
        * math.cos(latitude_sum)
        + (15 / 8 * n**2 + 15 / 8 * n**3)
        * math.sin(2 * latitude_difference)
        * math.cos(2 * latitude_sum)
        - 35 / 24
        * n**3
        * math.sin(3 * latitude_difference)
        * math.cos(3 * latitude_sum)
    )

    i = meridional_arc + northing_origin
    ii = nu / 2 * sin_latitude * cos_latitude
    iii = (
        nu
        / 24
        * sin_latitude
        * cos_latitude**3
        * (5 - tan_latitude**2 + 9 * eta_squared)
    )
    iiia = (
        nu
        / 720
        * sin_latitude
        * cos_latitude**5
        * (61 - 58 * tan_latitude**2 + tan_latitude**4)
    )
    iv = nu * cos_latitude
    v = nu / 6 * cos_latitude**3 * (nu / rho - tan_latitude**2)
    vi = (
        nu
        / 120
        * cos_latitude**5
        * (
            5
            - 18 * tan_latitude**2
            + tan_latitude**4
            + 14 * eta_squared
            - 58 * tan_latitude**2 * eta_squared
        )
    )

    longitude_difference = longitude - longitude_origin
    northing = (
        i
        + ii * longitude_difference**2
        + iii * longitude_difference**4
        + iiia * longitude_difference**6
    )
    easting = (
        easting_origin
        + iv * longitude_difference
        + v * longitude_difference**3
        + vi * longitude_difference**5
    )
    return easting, northing


def wgs84_to_bng(latitude: float, longitude: float) -> tuple[float, float]:
    """Convert a WGS84 crime coordinate to British National Grid metres."""
    osgb36_latitude, osgb36_longitude = wgs84_to_osgb36(latitude, longitude)
    return osgb36_to_bng(osgb36_latitude, osgb36_longitude)


def point_on_segment(
    x: float,
    y: float,
    start: list[float],
    end: list[float],
    tolerance: float = 1e-7,
) -> bool:
    """Return True if a point lies on a polygon segment."""
    x1, y1 = start[:2]
    x2, y2 = end[:2]
    cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
    if abs(cross_product) > tolerance:
        return False
    return (
        min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
        and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
    )


def point_in_ring(x: float, y: float, ring: list[list[float]]) -> bool:
    """Test a point against one closed linear ring using ray casting."""
    inside = False
    for start, end in zip(ring, ring[1:]):
        if point_on_segment(x, y, start, end):
            return True
        x1, y1 = start[:2]
        x2, y2 = end[:2]
        if (y1 > y) != (y2 > y):
            intersection_x = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < intersection_x:
                inside = not inside
    return inside


def point_in_polygon(x: float, y: float, rings: list[list[list[float]]]) -> bool:
    """Test a point against a polygon, respecting any interior holes."""
    return point_in_ring(x, y, rings[0]) and not any(
        point_in_ring(x, y, hole) for hole in rings[1:]
    )


def geometry_parts(geometry: dict) -> list[list[list[list[float]]]]:
    """Normalise Polygon and MultiPolygon coordinates to polygon parts."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError(f"Unsupported geometry type: {geometry['type']}")


def polygon_bounds(rings: list[list[list[float]]]) -> tuple[float, float, float, float]:
    """Return min/max easting and northing for one polygon part."""
    outer_ring = rings[0]
    eastings = [point[0] for point in outer_ring]
    northings = [point[1] for point in outer_ring]
    return min(eastings), min(northings), max(eastings), max(northings)


def load_boundaries() -> tuple[list[dict], str]:
    """Load Bristol polygon parts with their codes and bounding boxes."""
    source = json.loads(BOUNDARY_FILE.read_text(encoding="utf-8"))
    crs_name = source["crs"]["properties"]["name"]
    if crs_name not in {"EPSG:27700", "EPSG:4326"}:
        raise ValueError(f"Unsupported boundary coordinate system: {crs_name}")
    boundaries = []
    for feature in source["features"]:
        code = feature["properties"]["LSOA11CD"]
        for rings in geometry_parts(feature["geometry"]):
            boundaries.append(
                {
                    "code": code,
                    "rings": rings,
                    "bounds": polygon_bounds(rings),
                }
            )
    return boundaries, crs_name


def assign_lsoa(easting: float, northing: float, boundaries: list[dict]) -> list[str]:
    """Return every Bristol LSOA polygon containing the point."""
    matches = []
    for boundary in boundaries:
        min_e, min_n, max_e, max_n = boundary["bounds"]
        if min_e <= easting <= max_e and min_n <= northing <= max_n:
            if point_in_polygon(easting, northing, boundary["rings"]):
                matches.append(boundary["code"])
    return sorted(set(matches))


def main() -> None:
    boundaries, boundary_crs = load_boundaries()
    counts: Counter[str] = Counter()
    mismatches = []
    source_bristol_coordinates: set[tuple[float, float]] = set()
    exact_coordinates: set[tuple[float, float]] = set()
    mismatch_coordinates: set[tuple[float, float]] = set()

    with CRIME_FILE.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        for row_number, row in enumerate(reader, start=2):
            counts["rows"] += 1
            source_code = row["LSOA code"].strip()
            source_name = row["LSOA name"].strip()
            source_is_bristol = source_name.startswith("Bristol ")
            if source_is_bristol:
                counts["source_bristol"] += 1

            try:
                longitude = float(row["Longitude"])
                latitude = float(row["Latitude"])
            except ValueError:
                counts["missing_coordinates"] += 1
                if source_is_bristol:
                    counts["source_bristol_missing_coordinates"] += 1
                continue

            counts["valid_coordinates"] += 1
            if boundary_crs == "EPSG:4326":
                x_coordinate, y_coordinate = longitude, latitude
            else:
                x_coordinate, y_coordinate = wgs84_to_bng(latitude, longitude)
            matches = assign_lsoa(x_coordinate, y_coordinate, boundaries)
            if len(matches) > 1:
                counts["multiple_boundary_matches"] += 1
            assigned_code = matches[0] if matches else ""

            if assigned_code:
                counts["spatial_bristol"] += 1
            if source_is_bristol:
                counts["source_bristol_valid_coordinates"] += 1
                coordinate = (longitude, latitude)
                source_bristol_coordinates.add(coordinate)
                if source_code in matches:
                    counts["source_code_contained"] += 1
                    if len(matches) > 1:
                        counts["source_code_contained_ambiguous"] += 1
                if assigned_code == source_code:
                    counts["exact_agreement"] += 1
                    exact_coordinates.add(coordinate)
                else:
                    counts["source_bristol_not_exact"] += 1
                    mismatch_coordinates.add(coordinate)
                    if len(mismatches) < 10:
                        mismatches.append(
                            {
                                "row": row_number,
                                "longitude": longitude,
                                "latitude": latitude,
                                "source_code": source_code,
                                "assigned_code": assigned_code or "NO MATCH",
                            }
                        )
                if not assigned_code:
                    counts["source_bristol_outside_boundary"] += 1
            elif assigned_code:
                counts["spatial_bristol_source_not_bristol"] += 1
                if not source_code:
                    counts["spatial_bristol_source_code_missing"] += 1

    comparable = counts["source_bristol_valid_coordinates"]
    agreement_rate = counts["exact_agreement"] / comparable if comparable else 0
    containment_rate = counts["source_code_contained"] / comparable if comparable else 0
    unique_comparable = len(source_bristol_coordinates)
    unique_agreement_rate = (
        len(exact_coordinates) / unique_comparable if unique_comparable else 0
    )

    print(f"Test file: {CRIME_FILE.name}")
    print(f"Boundary file: {BOUNDARY_FILE.name}")
    print(f"Boundary coordinate system: {boundary_crs}")
    print(f"Crime rows: {counts['rows']:,}")
    print(f"Rows with valid coordinates: {counts['valid_coordinates']:,}")
    print(f"Rows missing coordinates: {counts['missing_coordinates']:,}")
    print(f"Rows labelled Bristol by Police.uk: {counts['source_bristol']:,}")
    print(
        "Police.uk Bristol rows with valid coordinates: "
        f"{counts['source_bristol_valid_coordinates']:,}"
    )
    print(f"Rows spatially assigned to Bristol: {counts['spatial_bristol']:,}")
    print(f"Exact 2011 LSOA agreements: {counts['exact_agreement']:,}")
    print(f"Exact agreement rate: {agreement_rate:.2%}")
    print(
        "Rows where the source code is among all containing polygons: "
        f"{counts['source_code_contained']:,} ({containment_rate:.2%})"
    )
    print(
        "Those rows with an ambiguous shared-boundary match: "
        f"{counts['source_code_contained_ambiguous']:,}"
    )
    print(f"Unique Police.uk Bristol coordinates: {unique_comparable:,}")
    print(f"Unique coordinates agreeing exactly: {len(exact_coordinates):,}")
    print(f"Unique coordinates disagreeing: {len(mismatch_coordinates):,}")
    print(f"Unique-coordinate agreement rate: {unique_agreement_rate:.2%}")
    print(
        "Police.uk Bristol coordinates outside the boundary: "
        f"{counts['source_bristol_outside_boundary']:,}"
    )
    print(
        "Spatial Bristol rows not labelled Bristol by Police.uk: "
        f"{counts['spatial_bristol_source_not_bristol']:,}"
    )
    print(f"Points matching multiple polygons: {counts['multiple_boundary_matches']:,}")

    if mismatches:
        print("\nFirst mismatches:")
        for mismatch in mismatches:
            print(mismatch)


if __name__ == "__main__":
    main()
