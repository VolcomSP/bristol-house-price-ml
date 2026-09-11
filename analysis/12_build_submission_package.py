"""Build a clean, marker-facing repository and compressed artefact package."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CLEAN_ROOT = PROJECT_ROOT / "submission_artefact"
ARCHIVE = OUTPUT_DIR / "bristol_house_price_complete_artefact.zip"

ROOT_FILES = [
    "README.md",
    "SUBMISSION_CHECKLIST.md",
    "requirements.txt",
    "run_project.py",
    ".gitignore",
    ".streamlit/config.toml",
]
PIPELINE_FILES = [
    "prepare_lsoa11_boundaries.py",
    "test_lsoa11_assignment.py",
    "test_june_2023_lsoa11_conversion.py",
    "prepare_crime_features.py",
    "validate_crime_candidate.py",
    "prepare_claimant_features.py",
    "combine_bristol_house_prices.py",
    "validate_property_candidate.py",
    "validate_land_registry_year.py",
    "download_validate_onspd_bs.py",
]
PROCESSED_FILES = [
    "outputs/bristol_house_prices_combined_2022_2024.csv",
    "outputs/bristol_house_prices_combined_2022_2024.validation.json",
    "outputs/candidate/crime_features_2022_2024_lsoa11_candidate.csv",
    "outputs/candidate/claimant_features_2022_2024_lsoa11_candidate.csv",
    "outputs/candidate/claimant_features_2022_2024_lsoa11_candidate.validation.json",
    "outputs/candidate/bristol_house_prices_2022_2024_candidate.validation.json",
]
VERIFIED_SOURCE_FILES = [
    "verified_sources/ONSPD_MAY_2026_UK_BS_complete.csv",
    "verified_sources/bristol_lsoa11_boundaries_bfc_v3.geojson",
    "verified_sources/nomis/claimant_count_bristol_lsoa11_2022_2024_raw.csv",
    "verified_sources/nomis/population_aged_16_64_bristol_lsoa11_2020_raw.csv",
    "verified_sources/land_registry/RAW_DATA_REQUIRED.md",
    "verified_sources/land_registry/pp-2022.validation.json",
    "verified_sources/land_registry/pp-2023.validation.json",
    "verified_sources/land_registry/pp-2024.validation.json",
    "verified_sources/land_registry/land_registry_source_validation_2022_2024.md",
    "upload/Bristol wide monthly socioeconomic control.csv",
    "upload/RAW_DATA_REQUIRED.md",
]
MODEL_FILES = ["models/09_lagged_enriched_random_forest.joblib"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def included_paths() -> list[Path]:
    relative_files = [*ROOT_FILES, *PIPELINE_FILES, *PROCESSED_FILES, *VERIFIED_SOURCE_FILES, *MODEL_FILES]
    relative_files.extend(
        str(path.relative_to(PROJECT_ROOT))
        for folder in [PROJECT_ROOT / "analysis", PROJECT_ROOT / "dashboard", PROJECT_ROOT / "docs", PROJECT_ROOT / "tests"]
        for path in sorted(folder.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts
    )
    relative_files.extend(
        str(path.relative_to(PROJECT_ROOT))
        for folder in [PROJECT_ROOT / "outputs" / "eda", PROJECT_ROOT / "outputs" / "models"]
        for path in sorted(folder.iterdir())
        if path.is_file()
        and not path.name.endswith("_predictions.csv")
        and path.name != "09_lagged_neighbourhood_feature_alignment.csv"
    )
    unique = sorted(set(relative_files))
    paths = [PROJECT_ROOT / relative for relative in unique]
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required artefact files: " + ", ".join(missing))
    return paths


def main() -> None:
    if CLEAN_ROOT.exists():
        raise FileExistsError(
            f"{CLEAN_ROOT} already exists. Preserve or rename it before rebuilding."
        )

    source_paths = included_paths()
    staging_parent = OUTPUT_DIR
    with tempfile.TemporaryDirectory(prefix="submission_staging_", dir=staging_parent) as temporary:
        staging = Path(temporary) / CLEAN_ROOT.name
        for source in source_paths:
            relative = source.relative_to(PROJECT_ROOT)
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)

        copied_paths = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest = {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "project": "Predicting house prices in Bristol using machine learning and socio-economic data",
            "purpose": "Clean submission artefact repository",
            "raw_files_excluded": [
                "national HM Land Registry annual CSV files",
                "36 Avon and Somerset Police.uk street-crime CSV files",
                "national ONS 2011 boundary file",
                "backup and temporary files",
                "row-level model prediction exports",
            ],
            "file_count_excluding_manifest": len(copied_paths),
            "total_bytes_excluding_manifest": sum(path.stat().st_size for path in copied_paths),
            "files": {
                str(path.relative_to(staging)): {
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
                for path in copied_paths
            },
        }
        (staging / "SUBMISSION_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(CLEAN_ROOT)

    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(CLEAN_ROOT.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(CLEAN_ROOT))

    with zipfile.ZipFile(ARCHIVE) as archive:
        failed = archive.testzip()
        if failed is not None:
            raise ValueError(f"Archive integrity check failed at {failed}")

    print(f"Clean repository: {CLEAN_ROOT}")
    print(f"Files: {sum(path.is_file() for path in CLEAN_ROOT.rglob('*')):,}")
    print(f"Archive: {ARCHIVE}")
    print(f"Archive bytes: {ARCHIVE.stat().st_size:,}")
    print(f"Archive SHA-256: {sha256(ARCHIVE)}")


if __name__ == "__main__":
    main()
