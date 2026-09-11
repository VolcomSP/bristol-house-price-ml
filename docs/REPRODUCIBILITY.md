# Reproducibility guide

## Reproduction levels

The repository supports two levels of reproduction.

### Included processed-data reproduction

This is the recommended route for a marker. The package contains the validated Bristol dataset, neighbourhood feature tables and selected dashboard model. It can recreate every EDA, modelling and dashboard output without downloading the national Land Registry files or 36 Police.uk files.

### Raw-source reconstruction

The raw preparation scripts are also included for audit. Reconstructing from source requires roughly 650 MB of additional Land Registry and crime downloads and internet access for Nomis. The exact inputs and filenames are listed below.

## Environment

Python 3.12 and the fixed `requirements.txt` versions were used for the validated run.

```bash
python -m venv .venv
```

Activate the environment, then run:

```bash
python -m pip install -r requirements.txt
python run_project.py --stage validate
```

## Recreate included analytical outputs

```bash
python run_project.py --stage eda
python run_project.py --stage models
python run_project.py --stage dashboard-data
python run_project.py --stage validate
```

Or run the complete sequence:

```bash
python run_project.py --stage all
```

The modelling stage refits multiple models and can take several minutes. It writes results into `outputs/eda`, `outputs/models` and `models`.

## Start the user interface

```bash
python run_project.py --stage dashboard
```

The tool supports January–December 2024. Its default example should return approximately £344,496 with the packaged model and data.

## Raw-source directory requirements

Place the annual Land Registry files here:

```text
verified_sources/land_registry/pp-2022.csv
verified_sources/land_registry/pp-2023.csv
verified_sources/land_registry/pp-2024.csv
```

Place the 36 Police.uk files here, one for every month from `2022-01` to `2024-12`:

```text
upload/YYYY-MM-avon-and-somerset-street.csv
```

The submission already contains:

- `verified_sources/ONSPD_MAY_2026_UK_BS_complete.csv`;
- `verified_sources/bristol_lsoa11_boundaries_bfc_v3.geojson`;
- compact Nomis source extracts;
- the citywide Nomis reconciliation file.

## Raw preparation order

Validate the annual Land Registry downloads:

```bash
python validate_land_registry_year.py 2022
python validate_land_registry_year.py 2023
python validate_land_registry_year.py 2024
```

Build and validate the crime panel:

```bash
python prepare_crime_features.py --output outputs/candidate/crime_features_2022_2024_lsoa11_candidate.csv
python validate_crime_candidate.py
```

Build the claimant panel, combine the property data and validate it:

```bash
python prepare_claimant_features.py
python combine_bristol_house_prices.py
python validate_property_candidate.py
```

The validated candidate has the same bytes as the included final modelling dataset. Promotion to a final version should occur only after the independent validation completes successfully.

## Integrity evidence

Validation JSON files record source/output SHA-256 hashes, row counts, duplicate checks, missingness and cross-source reconciliation. `analysis/11_validate_submission_artefact.py` verifies the included final data, feature panels, selected model, dashboard bundle and required documentation without requiring raw downloads.

Stochastic models use random seed `42`. Hyperparameters are chosen with chronological validation folds ending before the 2024 holdout.

## Expected headline checks

| Check | Expected value |
|---|---:|
| Final transactions | 21,968 |
| Category A transactions | 18,670 |
| Study months | 36 |
| Bristol 2011 LSOAs | 263 |
| LSOA-month feature rows | 9,468 |
| Selected-model 2024 test rows | 5,885 |
| Selected-model 2024 MAE | approximately £99,655 |
| Improvement over common-cohort property-only RF | approximately 20.0% |
