# Results summary

## Data quality

- 21,968 validated Bristol transactions covering all 36 months from January 2022 to December 2024.
- 18,670 standard Category A residential transactions in the primary cohort.
- 263 Bristol 2011 LSOAs.
- No duplicated transaction IDs.
- Complete crime and claimant matches for every transaction.
- 9,468 complete LSOA-month neighbourhood observations.

## Descriptive findings

| Property type | Transactions | Median price |
|---|---:|---:|
| Flat/maisonette | 6,391 | £255,500 |
| Terraced | 7,584 | £367,850 |
| Semi-detached | 3,814 | £360,000 |
| Detached | 881 | £550,000 |

At LSOA level, mean claimant rate had a Spearman correlation of -0.672 with median log price. Mean monthly crime had a correlation of -0.562. These are descriptive associations and can reflect omitted neighbourhood and housing-stock differences.

## Controlled model comparisons

All results below use direct exponentiation of predicted log price and an unseen 2024 holdout.

| Model | Feature set | 2024 MAE | 2024 RMSE | Price R² |
|---|---|---:|---:|---:|
| Linear regression | Property and time | £124,518 | £212,638 | 0.138 |
| Linear regression | Enriched | £115,054 | £199,155 | 0.244 |
| Random forest | Property and time | £124,558 | £212,700 | 0.137 |
| Random forest | Enriched, current month | £99,124 | £171,204 | 0.441 |
| Gradient boosting | Property and time | £124,524 | £213,169 | 0.133 |
| Gradient boosting | Enriched, current month | £101,832 | £176,776 | 0.404 |

The enriched random forest achieved the best validation MAE among the planned model families and was taken forward for the timing-sensitivity check.

## Lagged sensitivity and final model

The controlled lagged comparison uses a common cohort and frozen random-forest settings.

| Model | 2024 MAE | 2024 RMSE | Price R² |
|---|---:|---:|---:|
| Property-only random forest | £124,634 | £212,586 | 0.138 |
| Current-month enriched random forest | £99,685 | £171,953 | 0.436 |
| Previous-month enriched random forest | £99,655 | £172,680 | 0.431 |

The previous-month model reduced MAE by 20.0% relative to the property-only random forest and performed almost identically to the otherwise equivalent current-month model. This reduces concern that the improvement depends on later information from the sale month.

## Answer to the research question

Within the available data and evaluation design, adding LSOA-level crime and claimant conditions materially improved Bristol house-price prediction accuracy beyond basic Land Registry property and time features. The strongest evidence is the consistent validation improvement and the approximately 20% reduction in 2024 MAE for the selected lagged enriched random forest.

The result should be described as incremental predictive information, not a causal effect. The neighbourhood measures may also represent unobserved location quality, amenities, accessibility and housing stock.

## Selected artefact model

- Algorithm: random forest regressor.
- Target: log sale price.
- Training period: February 2022–December 2023.
- Test period: January–December 2024.
- Trees: 240.
- Maximum depth: 12.
- Minimum observations per leaf: 5.
- Feature proportion per split: 1.0.
- Neighbourhood timing: one calendar month before sale month.
- Dashboard headline method: direct exponentiation.
