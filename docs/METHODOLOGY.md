# Methodology

## 1. Data preparation

Annual Land Registry Price Paid files for 2022–2024 were validated and joined to the ONS Postcode Directory. Transactions were retained when their postcode mapped to City of Bristol LAD code `E06000023`.

Police.uk street records were deduplicated by nonblank crime ID within month. Records with a valid Bristol 2011 LSOA code retained that code; other usable coordinates were spatially assigned to the full-resolution 2011 boundary. Crime records were then aggregated to one row per LSOA and month, with explicit zero-count combinations added where necessary.

Nomis claimant counts and the 2020 population aged 16–64 were obtained for all 263 Bristol 2011 LSOAs. Claimant counts were converted to percentages using the fixed population denominator.

The property, crime and claimant tables were joined by 2011 LSOA and sale month. The resulting data has one row per transaction.

## 2. Geographic design

Police data changed from 2011 to 2021 statistical geography during the study period. Mixing those boundaries could create an artificial break in a time comparison. All crime observations were therefore placed on 2011 LSOAs, which are also the geography available for the selected population denominator.

The full neighbourhood panel contains:

\[
263\text{ LSOAs} \times 36\text{ months} = 9{,}468\text{ observations}
\]

## 3. Analytical cohort

The validated source dataset contains 21,968 transactions. The primary model cohort is restricted reproducibly to 18,670 Price Paid Data Category A transactions. This removes non-standard Category B records and leaves the four normal residential property types: detached, semi-detached, terraced and flat/maisonette.

No high or low price was manually deleted from the Category A cohort. The natural logarithm of sale price was used as the target because raw price was extremely right-skewed.

## 4. Feature sets

The property-only feature set contains:

- property type;
- freehold or leasehold tenure;
- new-build status;
- continuous month index;
- sine and cosine terms for month of year.

The enriched feature set adds:

- `log1p(total_crime)`;
- claimant rate as a percentage of the 2020 working-age population.

Categorical variables are one-hot encoded inside each model pipeline. All transformations are defined in code and are fitted using development data only.

## 5. Experimental design

The study compares the property-only and enriched feature sets under common conditions. The same transactions, time split and performance measures are used so that differences can be attributed to the additional information rather than a changed sample.

- Development data: 2022–2023.
- Final holdout: 2024.
- Tuning: three expanding-window folds within the development period.
- Primary selection measure: mean validation MAE after direct exponentiation.
- Reported measures: MAE, RMSE, price-scale R², log-price R² and median absolute percentage error.

Models tested:

1. linear regression;
2. random forest;
3. histogram gradient boosting.

The 2024 holdout was not used to select hyperparameters. Final algorithm selection used validation performance rather than choosing the lowest reported 2024 error after inspection.

## 6. Log-price transformation

Each model predicts:

\[
y = \log(\text{price})
\]

The primary pound prediction is:

\[
\widehat{\text{price}} = \exp(\hat{y})
\]

A training-derived smearing correction was also evaluated because directly reversing a log transform can underestimate an arithmetic mean. Direct exponentiation produced the lower MAE, while smearing sometimes reduced RMSE. Both are reported, and direct exponentiation is used for the dashboard headline because MAE is the primary error measure.

## 7. Leakage sensitivity

Same-month crime can include incidents recorded after an early-month sale. The selected random-forest specification was therefore refitted using crime and claimant values from the preceding calendar month.

January 2022 has no December 2021 feature data. The sensitivity development cohort consequently begins in February 2022, while every 2024 test transaction remains available. Current-month and lagged enriched models used identical transactions and frozen hyperparameters, isolating the timing change.

## 8. Dashboard

The dashboard accepts a postcode, property type, tenure, build status and 2024 sale month. It maps the postcode to a 2011 LSOA, obtains the preceding month's neighbourhood variables and constructs the eight model inputs in their saved order.

The dashboard is limited to the independently tested 2024 period. It displays the estimate, neighbourhood inputs, held-out performance and limitations. It is a research demonstration rather than a live production valuation service.

## 9. Reproducibility controls

- Fixed random seed (`42`) for stochastic models.
- Fixed dependency versions.
- Chronological rather than random train/test split.
- Source and output SHA-256 hashes.
- Row, duplicate, missingness and join checks.
- Saved model-feature order.
- Automated reproduction of representative predictions.
- Separate immutable source table and derived modelling subsets.
