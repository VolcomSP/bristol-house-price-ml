# Lagged-neighbourhood sensitivity analysis

## Purpose

The main models associate each sale with crime and claimant conditions from the same calendar month. For an early-month sale, some of that monthly information may not yet have been observed. This checkpoint therefore replaces both neighbourhood variables with values from the previous calendar month.

## Common comparison cohort

| Dataset | Period | Transactions |
|---|---|---:|
| Development | February 2022–December 2023 | 12,343 |
| Test | January–December 2024 | 5,885 |

The full dataset contains 505 January 2022 transactions without December 2021 source data. Of these, 442 belong to the Category A modelling cohort and are excluded only from this sensitivity exercise. Every 2024 test transaction has correctly dated previous-month crime and claimant information.

## Controlled design

The three models use identical transactions and time folds. Hyperparameters are frozen from the earlier property-only and enriched random-forest checkpoints. The contemporaneous and lagged enriched models use the same enriched parameters, so the only difference between them is neighbourhood timing.

## Time-based validation

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only random forest | £121,067 | £200,340 | 0.140 |
| Current-month enriched random forest | £99,215 | £165,521 | 0.412 |
| Previous-month enriched random forest | £99,560 | £167,279 | 0.400 |

The lagged model reduces mean validation MAE by **17.8%** relative to the property-only model.

## Final 2024 results

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only random forest | £124,634 | £212,586 | 0.138 | 0.234 |
| Current-month enriched random forest | £99,685 | £171,953 | 0.436 | 0.497 |
| Previous-month enriched random forest | £99,655 | £172,680 | 0.431 | 0.496 |

The lagged model reduces 2024 MAE by **20.0%** relative to the property-only model. Its MAE changes by **-0.0%** relative to the otherwise identical current-month model.

## Interpretation and decision

The predictive value of neighbourhood conditions remains after lagging. This reduces concern that the main improvement is explained by using information from later in the sale month. The recommended dashboard model is **Lagged enriched random forest** because previous-month inputs have a clearer real-world availability interpretation.

This remains an associational prediction exercise. Lagging the variables addresses timing leakage but does not make crime or claimant conditions causal determinants of price.

## Limitations

- A one-month lag is a practical availability rule rather than a guarantee of each source's exact publication date.
- The claimant-rate denominator remains the fixed 2020 working-age population.
- Neighbourhood variables may proxy for unmeasured amenities, housing stock and location quality.
- The data lacks bedrooms, floor area, condition and detailed property amenities.

## Next checkpoint

Create the final dashboard-ready model bundle and prediction input table, then implement the small visual prediction tool promised in the proposal.
