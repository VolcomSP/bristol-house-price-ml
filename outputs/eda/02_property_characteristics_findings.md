# Property-characteristics exploratory analysis

## Scope

This checkpoint examines how sale prices vary by property type, tenure, new-build status and sale year. It uses the agreed primary residential cohort: **18,670 Land Registry Category A transactions**. The complete 21,968-row source dataset remains unchanged.

## Property type

| Property type | Transactions | Median price |
|---|---:|---:|
| Flat/Maisonette | 6,391 | £255,500 |
| Terraced | 7,584 | £367,850 |
| Semi-detached | 3,814 | £360,000 |
| Detached | 881 | £550,000 |

Detached properties have the highest median price and flats/maisonettes the lowest. The detached median is **115.3%** higher than the flat/maisonette median. Property type is therefore likely to be an important predictor.

## Tenure

- Freehold transactions: **11,661**, median £370,000.
- Leasehold transactions: **7,009**, median £265,000.

The overall freehold/leasehold difference should not be interpreted as the independent effect of tenure. **6,307 of 6,391 flats/maisonettes are leasehold**, while most houses are freehold. Property type and tenure are therefore strongly related and both should be controlled for in modelling. The detached-leasehold group contains only 30 transactions, so its median is unstable; freehold flats are also uncommon, with 84 transactions.

## New-build status

- Established properties: **17,508**, median £340,000.
- New builds: **1,162**, median £295,950.

Within-property-type comparisons:

- Flat/Maisonette: new-build median £262,250 versus £255,000 for established properties (+2.8%; new-build n=708).
- Terraced: new-build median £320,000 versus £370,000 for established properties (-13.5%; new-build n=307).
- Semi-detached: new-build median £349,000 versus £360,000 for established properties (-3.1%; new-build n=124).
- Detached: new-build median £360,000 versus £571,000 for established properties (-37.0%; new-build n=23).

The direction and size of the new-build difference varies by property type, reinforcing the need to include both variables rather than treating the overall difference as causal.

## Sale year

| Year | Transactions | Median price |
|---:|---:|---:|
| 2022 | 7,018 | £335,000 |
| 2023 | 5,767 | £335,500 |
| 2024 | 5,885 | £340,000 |

These are nominal sale prices and have not been adjusted for inflation. Year and month should be retained as time controls because market conditions and the mix of properties sold change over time.

## Modelling implications

1. Include property type, tenure, new-build status and time variables in the baseline model.
2. Treat all categorical variables as categories rather than numeric ranks.
3. Check interactions or subgroup errors because tenure and new-build effects may vary by property type.
4. Avoid causal wording: these summaries demonstrate associations, not the effect of changing one characteristic while holding everything else constant.
5. Once modelling begins, use 2022–2023 for fitting and tuning. Evaluate 2024 only after the model and feature decisions are fixed; this full-period EDA is descriptive and must not be used to tune against 2024 prediction errors.
