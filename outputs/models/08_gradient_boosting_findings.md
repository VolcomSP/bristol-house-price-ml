# Gradient-boosting comparison

## Purpose

This checkpoint compares property-only and enriched histogram gradient-boosting models. Both predict `log(price)` using the same Category A transactions, chronological folds and 2024 holdout. The enriched model adds `log1p(total_crime)` and `claimant_rate_percent`.

## Model design

Eight predefined parameter settings were assessed separately for each feature set. All models used 300 boosting iterations, L2 regularisation of 1.0, early stopping disabled and random seed 42. Parameter selection used mean validation MAE; 2024 was not used for tuning.

Selected property-only parameters: learning rate **0.03**, maximum leaves **15**, minimum leaf size **50**.

Selected enriched parameters: learning rate **0.03**, maximum leaves **31**, minimum leaf size **20**.

## Time-based validation

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only gradient boosting | £121,104 | £200,428 | 0.139 |
| Enriched gradient boosting | £99,071 | £165,428 | 0.413 |

Enrichment reduces mean validation MAE by **18.2%** and mean validation RMSE by **17.5%**.

## Final 2024 comparison

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only gradient boosting | £124,524 | £213,169 | 0.133 | 0.232 |
| Enriched linear regression | £115,054 | £199,155 | 0.244 | 0.354 |
| Enriched random forest | £99,124 | £171,204 | 0.441 | 0.502 |
| Enriched gradient boosting | £101,832 | £176,776 | 0.404 | 0.482 |

Compared with property-only gradient boosting, enrichment reduces 2024 MAE by **18.2%** and RMSE by **17.1%**. Price R² changes by **+0.271**.

## Model selection

The planned final-model candidate is **Enriched random forest**, selected using the lowest mean validation MAE across the three enriched algorithms. This rule avoids choosing solely from the 2024 holdout result. A leakage-sensitivity check should still be completed before the dashboard is finalised.

## Permutation importance

The largest enriched-model permutation effects were `claimant_rate_percent` (0.1041), `property_type` (0.0814), `duration` (0.0130), `log_total_crime` (0.0102), `old_new` (0.0029). The values are increases in log-price MAE after shuffling a feature. They measure predictive reliance, not causation.

## Limitations

- The Land Registry data lacks bedrooms, floor area, condition and detailed amenities.
- Contemporaneous monthly crime can include incidents occurring after an early-month sale.
- The claimant-rate denominator is the fixed 2020 working-age population.
- Neighbourhood variables may proxy for omitted location and housing-stock characteristics.
- The same 2024 holdout is reported for several pre-planned model families; final selection therefore uses validation performance.

## Next checkpoint

Run a lagged-crime sensitivity experiment for the selected enriched model. This will show whether the predictive improvement remains when each transaction uses crime information available before the sale month.
