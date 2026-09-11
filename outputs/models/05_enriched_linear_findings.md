# Enriched linear model: crime and claimant conditions

## Purpose

This checkpoint tests whether two neighbourhood-condition variables improve 2024 house-price predictions beyond the property-only linear baseline:

- `log1p(total_crime)`: the natural logarithm of one plus the number of police-recorded crimes in the property's 2011 LSOA during the sale month.
- `claimant_rate_percent`: claimant count as a percentage of the fixed 2020 working-age population in the property's 2011 LSOA.

Everything else is held constant: Category A transactions, `log(price)` target, property and time features, linear regression, expanding-window validation, and the 2024 holdout.

## Comparable sample

| Dataset | Period | Transactions |
|---|---|---:|
| Development | 2022–2023 | 12,785 |
| Test | 2024 | 5,885 |

Both neighbourhood variables are present for every transaction. Adding them therefore does not alter the modelling sample.

## Time-based validation

| Model | Mean MAE | Mean RMSE | Mean price R² |
|---|---:|---:|---:|
| Property-only linear regression | £120,533 | £199,962 | 0.143 |
| Enriched linear regression | £111,136 | £188,427 | 0.240 |

Across the three validation periods, enrichment reduces mean MAE by **7.8%** and mean RMSE by **5.8%**. The enriched direct transformation has lower MAE than its smeared alternative in every validation fold, so direct exponentiation remains the primary prediction method.

## Final 2024 comparison

| Model | MAE | RMSE | Price R² | Log-price R² |
|---|---:|---:|---:|---:|
| Property-only linear regression | £124,518 | £212,638 | 0.138 | 0.237 |
| Enriched linear regression | £115,054 | £199,155 | 0.244 | 0.354 |
| Enriched linear regression with smearing | £117,759 | £193,665 | 0.285 | 0.354 |

Relative to the property-only model, the direct enriched model reduces MAE by **7.6%** (9,464 pounds) and reduces RMSE by **6.3%** (13,483 pounds). Its price R² increases by **+0.106**. The smeared enriched result reduces RMSE by **8.9%**, but has higher MAE than direct exponentiation.

## Neighbourhood coefficients

Holding the included property and time features constant:

- A one-standard-deviation increase in `log1p(total_crime)` is associated with an approximate **-4.0%** change in predicted price.
- A one-standard-deviation increase in `claimant_rate_percent` is associated with an approximate **-13.0%** change in predicted price.

These are conditional associations, not causal effects. Crime, claimant rate and omitted aspects of location are related, so the coefficients should not be interpreted as the effect of changing crime or claimant levels while everything else in the real world remains fixed.

## Interpretation

This is the first direct test of the research question. The comparison shows whether the two neighbourhood-condition variables add predictive information under the same linear specification. It does not yet show whether they improve nonlinear machine-learning models, and it does not include an LSOA identifier or detailed structural property attributes.

## Limitations

- Crime is measured for the entire sale month; for a strictly prospective prediction system, a lagged crime measure would avoid using incidents recorded after an early-month sale.
- Claimant rates use a fixed 2020 working-age population denominator.
- Neighbourhood variables may proxy for unobserved location quality, housing stock and amenities.
- Land Registry data still lacks bedrooms, floor area, condition and detailed amenities.
- Linear regression assumes additive, constant effects and cannot automatically learn interactions or thresholds.

## Next modelling experiment

Apply the same property-only versus enriched feature comparison to nonlinear tree-based models. A random forest and a gradient-boosting model can test whether crime and claimant conditions become more useful when interactions and nonlinear relationships are allowed.
