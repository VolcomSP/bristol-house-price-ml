# Property-only random forest

## Purpose

This checkpoint tests whether a nonlinear random forest improves house-price predictions when it receives exactly the same property and time features as the property-only linear model:

- Property type.
- Tenure.
- New-build status.
- Continuous month index and cyclical month-of-year terms.

No postcode, LSOA identifier, crime or claimant feature is included. The model predicts `log(price)` and uses the same Category A cohort, 2022–2023 development data, expanding-window validation and unseen 2024 holdout.

## Hyperparameter selection

Twelve candidate settings were evaluated across all three development-period folds. The selection criterion was mean validation MAE using direct exponentiation. The selected settings were:

- Trees: **240**.
- Maximum depth: **8**.
- Minimum observations per leaf: **20**.
- Candidate feature proportion per split: **0.7**.
- Random seed: **42**.

The 2024 test data was not used during parameter selection.

## Time-based validation

| Prediction method | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only linear regression | £120,533 | £199,962 | 0.143 |
| Random forest, direct exponentiation | £120,998 | £200,378 | 0.140 |
| Random forest, smearing | £130,046 | £199,086 | 0.151 |

The direct random forest changes mean validation MAE by **-0.4%** relative to the linear model. It therefore does not provide a meaningful validation improvement with this feature set.

## Final 2024 comparison

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only linear regression | £124,518 | £212,638 | 0.138 | 0.237 |
| Property-only random forest | £124,558 | £212,700 | 0.137 | 0.234 |
| Property-only random forest with smearing | £129,974 | £207,923 | 0.176 | 0.234 |

Relative to the property-only linear model, the direct random forest changes MAE by **+40 pounds** (-0.0%) and changes RMSE by **+63 pounds** (-0.0%). Its price R² changes by **-0.001**. These differences are negligible, with the random forest marginally worse on the primary metrics.

## Feature importance

The three most important transformed inputs were `property_type_F` (54.2%), `duration_L` (19.0%), `property_type_S` (11.0%). Random-forest impurity importance measures how often and how effectively a variable splits the training observations. It does not show causality and can favour continuous variables or features with more possible split points.

## Interpretation

This experiment isolates the effect of changing the algorithm. Because the feature set contains no detailed location or structural property information, a more flexible algorithm may still have limited information from which to learn. The result should therefore be assessed against the linear property-only model, not against the later enriched models.

## Limitations

- Land Registry data does not include bedrooms, floor area, property condition or detailed amenities.
- No postcode or neighbourhood identifier is included.
- Random forests cannot extrapolate beyond patterns observed in their training data.
- Impurity-based feature importance is descriptive and may be biased.
- Direct exponentiation and smearing optimise different error characteristics.

## Next modelling experiment

Fit an enriched random forest using the selected modelling procedure and add `log1p(total_crime)` and `claimant_rate_percent`. Comparing property-only and enriched random forests will test whether the neighbourhood variables provide predictive value under a nonlinear specification.
