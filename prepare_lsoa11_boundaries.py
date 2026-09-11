"""Validate the official 2011 LSOA boundary file and extract Bristol.

Source dataset:
Lower layer Super Output Areas (December 2011) Boundaries EW BGC (V3)
Office for National Statistics / Open Geography Portal

The national GeoJSON uses British National Grid (EPSG:27700). This script
checks the expected national and Bristol feature counts, validates the basic
polygon structure, and writes a Bristol-only GeoJSON for later spatial joins.
It does not modify any crime records.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_FILE = (
    PROJECT_ROOT
    / "verified_sources"
    / "LSOA_Dec_2011_Boundaries_Generalised_Clipped_BGC_EW_V3.geojson"
)
OUTPUT_FILE = (
    PROJECT_ROOT
    / "verified_sources"
    / "bristol_lsoa11_boundaries_bgc_v3.geojson"
)

EXPECTED_NATIONAL_LSOAS = 34_753
EXPECTED_BRISTOL_LSOAS = 263
LSOA_CODE_PATTERN = re.compile(r"^[EW]\d{8}$")


def geometry_rings(geometry: dict) -> list[list[list[float]]]:
    """Return every linear ring from a Polygon or MultiPolygon geometry."""
    geometry_type = geometry.get("type")
    coordinates = geometry.get("coordinates", [])
    if geometry_type == "Polygon":
        return coordinates
    if geometry_type == "MultiPolygon":
        return [ring for polygon in coordinates for ring in polygon]
    raise ValueError(f"Unexpected geometry type: {geometry_type!r}")


def validate_feature(feature: dict) -> None:
    """Validate the fields and basic ring structure required for the join."""
    properties = feature.get("properties") or {}
    code = properties.get("LSOA11CD")
    name = properties.get("LSOA11NM")
    if not isinstance(code, str) or not LSOA_CODE_PATTERN.fullmatch(code):
        raise ValueError(f"Invalid LSOA11CD: {code!r}")
    if not isinstance(name, str) or not name:
        raise ValueError(f"Missing LSOA11NM for {code}")

    geometry = feature.get("geometry")
    if not isinstance(geometry, dict):
        raise ValueError(f"Missing geometry for {code}")

    for ring in geometry_rings(geometry):
        if len(ring) < 4 or ring[0] != ring[-1]:
            raise ValueError(f"Invalid or unclosed polygon ring for {code}")
        for point in ring:
            if len(point) < 2 or not all(math.isfinite(value) for value in point[:2]):
                raise ValueError(f"Invalid coordinate for {code}")


def main() -> None:
    raw_bytes = SOURCE_FILE.read_bytes()
    source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
    source = json.loads(raw_bytes)

    if source.get("type") != "FeatureCollection":
        raise ValueError("The source is not a GeoJSON FeatureCollection.")
    crs_name = (source.get("crs") or {}).get("properties", {}).get("name")
    if crs_name != "EPSG:27700":
        raise ValueError(f"Unexpected coordinate reference system: {crs_name!r}")

    features = source.get("features")
    if not isinstance(features, list) or len(features) != EXPECTED_NATIONAL_LSOAS:
        raise ValueError(
            f"Expected {EXPECTED_NATIONAL_LSOAS:,} national LSOAs; "
            f"found {len(features) if isinstance(features, list) else 'invalid data'}."
        )

    codes: set[str] = set()
    bristol_features: list[dict] = []
    for feature in features:
        validate_feature(feature)
        properties = feature["properties"]
        code = properties["LSOA11CD"]
        if code in codes:
            raise ValueError(f"Duplicate LSOA11CD: {code}")
        codes.add(code)
        if properties["LSOA11NM"].startswith("Bristol "):
            bristol_features.append(feature)

    if len(bristol_features) != EXPECTED_BRISTOL_LSOAS:
        raise ValueError(
            f"Expected {EXPECTED_BRISTOL_LSOAS} Bristol LSOAs; "
            f"found {len(bristol_features)}."
        )

    bristol = {
        "type": "FeatureCollection",
        "name": "Bristol 2011 LSOA boundaries (ONS BGC V3)",
        "crs": source["crs"],
        "features": sorted(
            bristol_features,
            key=lambda feature: feature["properties"]["LSOA11CD"],
        ),
    }
    OUTPUT_FILE.write_text(
        json.dumps(bristol, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )

    print(f"National LSOAs validated: {len(features):,}")
    print(f"Unique national codes: {len(codes):,}")
    print(f"Bristol LSOAs written: {len(bristol_features):,}")
    print(f"Coordinate reference system: {crs_name}")
    print(f"Source SHA-256: {source_sha256}")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
