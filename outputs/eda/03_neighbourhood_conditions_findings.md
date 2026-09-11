# Neighbourhood-conditions exploratory analysis

## Scope

This checkpoint examines monthly crime, claimant conditions and their association with Category A residential sale prices. It uses three analytical levels:

- **18,670 property transactions** for the property-level relationship.
- **9,468 LSOA-month observations** representing all 263 Bristol 2011 LSOAs across 36 months.
- **263 LSOA summaries** so areas with many property sales do not dominate the neighbourhood comparison.

The original datasets remain unchanged.

## Feature distributions

| Feature | Median | Mean | 99th percentile | Maximum | Skewness |
|---|---:|---:|---:|---:|---:|
| Monthly crime count | 14 | 20.1 | 122 | 407 | 7.21 |
| log(1 + monthly crime) | 2.71 | 2.68 | 4.81 | 6.01 | 0.15 |
| Crime per 1,000 working-age residents (proxy) | 12.1 | 15.5 | 60.9 | 145.2 | 2.00 |
| Claimant rate (%) | 3.12 | 3.61 | 10.86 | 17.66 | 1.27 |

Monthly crime counts are strongly right-skewed (**skewness 7.21**). Applying `log1p` reduces the skewness to **0.15**, supporting its use in linear models. Tree-based models can still be tested with the untransformed count.

## Association with property prices

Transaction-level Spearman correlations with log sale price:

- Monthly crime count: **-0.301**.
- Crime per 1,000 working-age residents proxy: **-0.315**.
- Claimant rate: **-0.367**.

LSOA-level Spearman correlations with area median log price:

- Mean monthly crime: **-0.562**.
- Mean population-scaled crime proxy: **-0.606**.
- Mean claimant rate: **-0.672**.

The neighbourhood-level relationships are stronger than the transaction-level relationships because property characteristics introduce additional variation. The correlations remain descriptive: they do not demonstrate that crime or claimant rates cause lower prices.

## Quintile comparison

- The lowest mean-crime quintile has a median area price of **£461,500**, compared with **£285,000** in the highest-crime quintile.
- The lowest claimant-rate quintile has a median area price of **£476,000**, compared with **£276,250** in the highest claimant-rate quintile.

The ordered pattern supports testing both neighbourhood conditions in the enriched prediction models. It may also reflect unmeasured location, accessibility, property mix and neighbourhood characteristics.

## Relationship between the neighbourhood variables

Crime and claimant conditions are related. The LSOA-month Spearman correlation between monthly crime and claimant rate is **0.508**. Raw crime and the population-scaled crime proxy are very strongly correlated (**0.949**).

Raw crime, log crime and the crime-rate proxy should therefore not all be included in the same simple linear model. They should be compared as alternative representations.

## Crime-rate limitation

The population-scaled crime measure uses the available **2020 working-age population** as its denominator. This is a useful sensitivity measure, but it is not a true crime rate for the full resident or daytime population. City-centre crime can also reflect visitors and commuters. The primary enriched experiment should therefore use `log1p(total_crime)`, with the population-scaled proxy tested separately.

## Modelling implications

1. Use `log1p(total_crime)` and `claimant_rate_percent` in the primary enriched linear model.
2. Compare raw crime and the population-scaled proxy in sensitivity experiments rather than placing all versions together.
3. Retain month or date controls because both neighbourhood features vary through time.
4. Check multicollinearity and feature importance after property and time characteristics are included.
5. Use the same 2022–2023 training data and fixed 2024 test data for both the property-only and enriched models.
6. Interpret improvement in prediction as incremental information, not evidence of causation.

## Data-quality confirmation

The neighbourhood panel contains all **263 LSOAs × 36 months = 9,468 observations**, with no duplicate LSOA-month combinations or missing feature values. The summed crime total is **190,613**, matching the standardised Bristol crime table. The monthly chart shows no artificial discontinuity at the original June 2023 boundary transition after conversion to consistent 2011 LSOAs.
