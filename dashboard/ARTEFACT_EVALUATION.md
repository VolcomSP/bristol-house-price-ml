# Dashboard artefact evidence

## Purpose

The Bristol House Price Explorer is the user-facing component promised in the project proposal. It allows a researcher or housing-data analyst to explore a historical 2024 Bristol sale-price estimate without manually supplying technical neighbourhood variables.

## User journey

1. The user enters a Bristol postcode.
2. The user selects property type, tenure, build status and a 2024 sale month.
3. The application converts the postcode to its 2011 LSOA.
4. It retrieves total recorded crime and claimant rate from the preceding calendar month.
5. It constructs the eight model features in the exact order expected by the saved pipeline.
6. The validated lagged enriched random forest returns an estimated sale price.
7. The application shows the neighbourhood inputs, held-out performance evidence and limitations.

## Why the model is suitable for the interface

The one-month-lagged version is used because it reduces the risk of relying on neighbourhood information that would not yet have been available at prediction time. On the common 2024 test cohort, it achieved:

| Measure | Result |
|---|---:|
| Test transactions | 5,885 |
| Mean absolute error | £99,655 |
| Root mean squared error | £172,680 |
| Price-scale R² | 0.431 |
| Median absolute percentage error | 18.3% |
| MAE improvement over property-only random forest | 20.0% |

The headline dashboard result uses `exp(predicted log price)` without the smearing adjustment because that version achieved the lower held-out MAE, the project's primary prediction-error measure.

## Validation completed

- 17,844 Bristol postcode records map uniquely to 263 valid 2011 LSOAs.
- The dashboard feature table contains 3,156 rows: 263 LSOAs for each of 12 sale months.
- Every feature row uses the calendar month immediately before the requested sale month.
- No lagged crime or claimant-rate values are missing.
- Six fixed 2024 regression predictions are reproduced to the exported precision.
- Unsupported dates, malformed postcodes and postcodes outside the Bristol lookup are rejected.
- The interface loads without exceptions, contains all three intended tabs and produces a prediction from the default form.
- The local Streamlit health endpoint responds successfully.

## Scope and limitations

The tool is restricted to January–December 2024 because this is the complete, independently tested holdout period. It is a historical analytical demonstration rather than a current-market valuation service.

The Price Paid Data does not include bedrooms, floor area, property condition or detailed amenities. Crime and claimant measures describe an LSOA rather than an individual property. The estimated relationship is predictive and associational; it does not establish that changing crime or claimant rates would cause a particular house price to change.

The displayed test MAE is a model-wide evaluation statistic and must not be presented as an individual confidence interval.

## Reproduction

Run the data-preparation script, the tests and then the application from the project root:

```bash
python analysis/10_prepare_dashboard_artefact.py
python -m unittest discover -s tests -p "test_dashboard*.py" -v
python -m streamlit run dashboard/app.py
```
