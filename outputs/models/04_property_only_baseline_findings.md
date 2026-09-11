# Property-only linear baseline

## Purpose

This checkpoint establishes the baseline prediction performance before crime and claimant variables are introduced. It uses only basic property and time information:

- Property type: detached, semi-detached, terraced or flat/maisonette.
- Tenure: freehold or leasehold.
- New-build status: new build or established property.
- A continuous month index and cyclical month-of-year terms.

The model predicts `log(price)`. Two back-transformations are evaluated: direct exponentiation, which estimates a conditional median, and a training-derived smearing correction, which shifts predictions towards an expected pound value. No postcode, LSOA, crime or claimant feature is included.

## Chronological split

| Dataset | Period | Transactions | Median price |
|---|---|---:|---:|
| Development | 2022–2023 | 12,785 | £335,000 |
| Test | 2024 | 5,885 | £340,000 |

There is no transaction overlap, and every development transaction occurs before every test transaction. Model fitting and the optional smearing correction use development data only.

## Time-based validation

Three expanding-window validation folds were used within 2022–2023. Every fold begins with at least 12 months of training data so that all seasons are represented. The average results were:

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Training-median benchmark | £130,186 | £221,735 | -0.054 |
| Property-only linear regression | £120,533 | £199,962 | 0.143 |
| Property-only linear regression with smearing | £129,955 | £198,626 | 0.155 |

## Final 2024 holdout result

| Model | MAE | RMSE | Price R² | Log-price R² | Median absolute percentage error |
|---|---:|---:|---:|---:|---:|
| Training-median benchmark | £138,952 | £236,096 | -0.063 | -0.007 | 25.6% |
| Property-only linear regression | £124,518 | £212,638 | 0.138 | 0.237 | 24.5% |
| Property-only linear regression with smearing | £129,962 | £207,708 | 0.177 | 0.237 | 27.9% |

Relative to predicting the development-set median for every property, direct exponentiation reduces MAE by **10.4%**. Smearing produces the lower RMSE, reducing it by **12.0%**. Because MAE is less dominated by unusually expensive properties, direct exponentiation is retained as the primary baseline prediction; the smeared result remains a reported sensitivity measure.

## Coefficient interpretation

The categorical reference property is detached, freehold and established. Holding the other included features constant:

- Flat/maisonette coefficient: **-0.824**, approximately **-56.1%** relative to detached.
- Semi-detached coefficient: **-0.378**, approximately **-31.5%** relative to detached.
- Terraced coefficient: **-0.423**, approximately **-34.5%** relative to detached.
- Leasehold coefficient: **0.052**, approximately **5.3%** relative to freehold.
- New-build coefficient: **-0.031**, approximately **-3.1%** relative to established property.

These coefficients describe conditional associations within this additive model. They should not be interpreted causally, especially because property type and tenure are related and the model does not yet include floor area, bedrooms, exact location or other important housing characteristics.

## Limitations

- The Land Registry data does not provide bedrooms, floor area, condition or detailed amenities.
- An additive linear model cannot automatically learn interactions or nonlinear effects.
- Predictions for unusually expensive properties remain difficult, so RMSE is substantially influenced by the upper price tail.
- Direct exponentiation and smearing optimise different aspects of performance: the former gives lower MAE, while the latter gives lower RMSE.
- This baseline contains no explicit neighbourhood location or neighbourhood-condition feature.

## Next modelling experiment

Fit the same linear model on the same 12,785 development transactions and evaluate the same 5,885 test transactions after adding:

- `log1p(total_crime)`
- `claimant_rate_percent`

The difference in 2024 MAE, RMSE and R² will provide the first direct answer to whether the neighbourhood variables improve predictions beyond property and time characteristics.
