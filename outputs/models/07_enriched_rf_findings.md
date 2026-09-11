# Enriched random forest

## Purpose

This checkpoint tests the research question under a nonlinear specification. It adds monthly 2011-LSOA crime and claimant rate to the property-only random forest while keeping the cohort, target, chronological split, validation folds and tuning procedure unchanged.

## Comparable sample and leakage control

| Dataset | Period | Transactions |
|---|---|---:|
| Development | 2022–2023 | 12,785 |
| Test | 2024 | 5,885 |

The 2024 records were not used for hyperparameter selection. Crime and claimant variables are complete for every modelling transaction, so enrichment does not change the sample.

## Hyperparameter selection

Twelve settings were tested using mean validation MAE. The selected settings were 240 trees, maximum depth **12**, minimum leaf size **5**, and maximum feature proportion **1.0**.

## Time-based validation

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only random forest | £120,998 | £200,378 | 0.140 |
| Enriched random forest | £98,721 | £164,447 | 0.420 |

Adding the neighbourhood variables reduces mean validation MAE by **18.4%** and mean validation RMSE by **17.9%**.

## Final 2024 comparison

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only random forest | £124,558 | £212,700 | 0.137 | 0.234 |
| Enriched linear regression | £115,054 | £199,155 | 0.244 | 0.354 |
| Enriched random forest | £99,124 | £171,204 | 0.441 | 0.502 |
| Enriched random forest with smearing | £99,257 | £168,284 | 0.460 | 0.502 |

Against the property-only random forest, the direct enriched model reduces 2024 MAE by **20.4%** and RMSE by **19.5%**. Its price R² changes by **+0.304**.

## Feature importance

The five largest impurity importances are `claimant_rate_percent` (50.8%), `property_type_F` (26.6%), `log_total_crime` (8.8%), `month_index` (4.4%), `month_sin` (2.3%). These values show how the fitted forest used variables for prediction; they do not demonstrate causal effects.

## Interpretation

This experiment supplies a nonlinear test of whether crime and claimant conditions contain predictive information beyond basic property and time variables. The fair comparison is the change from the property-only random forest under the same data split and tuning procedure.

## Limitations

- The Land Registry data lacks bedrooms, floor area, condition and detailed amenities.
- Monthly crime can include incidents occurring after an early-month sale, so a lagged sensitivity test is still advisable for a strictly prospective interpretation.
- The claimant-rate denominator is the fixed 2020 working-age population.
- Neighbourhood conditions may proxy for omitted aspects of location and housing stock.
- Impurity importance is descriptive and can favour variables with more potential split points.

## Next modelling experiment

Fit property-only and enriched gradient-boosting models using the same chronological design. This will provide a second nonlinear comparison before selecting the final model for the dashboard.
