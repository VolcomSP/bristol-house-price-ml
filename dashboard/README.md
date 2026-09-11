# Bristol House Price Explorer

This Streamlit interface is the user-facing component of the MSc project artefact. It uses the validated one-month-lagged enriched random forest to explore historical Bristol property-price estimates for January to December 2024.

## What a user enters

- Bristol postcode
- Property type
- Freehold or leasehold
- Existing property or new build
- Intended 2024 sale month

The postcode is converted to a 2011 LSOA. The app automatically obtains total recorded crime and claimant rate from the month before the selected sale month.

## Run the dashboard

From the project root:

```bash
python -m pip install -r dashboard/requirements.txt
python -m streamlit run dashboard/app.py
```

Streamlit will display a local address, normally `http://localhost:8501`.

## Rebuild the dashboard data

The selected model is not retrained by this command:

```bash
python analysis/10_prepare_dashboard_artefact.py
```

It rebuilds and validates:

- `dashboard/data/bristol_postcode_lsoa11.csv`
- `dashboard/data/lagged_neighbourhood_features_2024.csv`
- `dashboard/data/model_performance_2024.csv`
- `dashboard/data/dashboard_validation_cases.csv`
- `dashboard/model_card.json`
- `dashboard/dashboard_validation.json`

## Run the checks

The tests use Python's built-in test runner:

```bash
python -m unittest discover -s tests -p "test_dashboard*.py" -v
```

## Interpretation

The headline estimate is the exponential of the model's predicted log price. This version is used because it achieved the lowest 2024 mean absolute error in the validated comparison. The model-wide 2024 MAE was approximately £99,655; this is not a confidence interval for an individual estimate.

The dashboard is an academic research artefact, not a professional or current-market valuation service. It omits important property-level variables that were unavailable in the Land Registry Price Paid Data, including bedrooms, floor area and condition.
