"""Run the reproducible stages of the Bristol house-price artefact."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

EDA_SCRIPTS = [
    "analysis/01_dataset_quality_and_price.py",
    "analysis/02_property_characteristics_eda.py",
    "analysis/03_neighbourhood_conditions_eda.py",
]
MODEL_SCRIPTS = [
    "analysis/04_property_only_linear_baseline.py",
    "analysis/05_enriched_linear_model.py",
    "analysis/06_property_only_random_forest.py",
    "analysis/07_enriched_random_forest.py",
    "analysis/08_gradient_boosting_comparison.py",
    "analysis/09_lagged_neighbourhood_sensitivity.py",
]
DASHBOARD_DATA_SCRIPT = "analysis/10_prepare_dashboard_artefact.py"
VALIDATION_SCRIPT = "analysis/11_validate_submission_artefact.py"


def run_python(relative_path: str) -> None:
    print(f"\nRunning {relative_path}", flush=True)
    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / relative_path)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def run_tests() -> None:
    print("\nRunning dashboard tests", flush=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            "tests",
            "-p",
            "test_*.py",
            "-v",
        ],
        cwd=PROJECT_ROOT,
        check=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=["validate", "eda", "models", "dashboard-data", "dashboard", "all"],
        help="Artefact stage to run.",
    )
    return parser.parse_args()


def main() -> None:
    stage = parse_args().stage
    if stage in {"eda", "all"}:
        for script in EDA_SCRIPTS:
            run_python(script)
    if stage in {"models", "all"}:
        for script in MODEL_SCRIPTS:
            run_python(script)
    if stage in {"dashboard-data", "all"}:
        run_python(DASHBOARD_DATA_SCRIPT)
    if stage in {"validate", "all"}:
        run_python(VALIDATION_SCRIPT)
        run_tests()
    if stage == "dashboard":
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "dashboard/app.py"],
            cwd=PROJECT_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
