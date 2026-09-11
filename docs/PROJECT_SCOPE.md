# Project scope

## Title

Predicting house prices in Bristol using machine learning and socio-economic data.

## Research question

> Can adding local crime, socioeconomic and property-related features improve the accuracy of machine-learning house-price predictions in Bristol compared with using basic Land Registry property data alone?

## Aim

To develop and evaluate a reproducible analytical artefact that integrates official property and neighbourhood data and tests whether the added neighbourhood information improves Bristol house-price predictions.

## Objectives

1. Collect and validate Bristol residential transactions from 2022–2024.
2. Harmonise crime, claimant and property records to consistent 2011 LSOA geography.
3. Create a documented transaction-level modelling dataset and retain validation evidence.
4. Explore price, property and neighbourhood patterns without changing the validated source table.
5. Compare property-only and enriched feature sets using linear regression, random forest and gradient boosting.
6. Use chronological validation within 2022–2023 and reserve 2024 as an unseen test period.
7. Test whether the result survives using neighbourhood values from the month before each sale.
8. Deliver a small visual tool through which a user can generate and interpret a historical estimate.

## Intended user

The primary user is a researcher or housing-data analyst who needs a transparent way to examine whether property and neighbourhood conditions contain useful predictive information. The artefact also gives an academic marker a reproducible path from processed data to findings and a working interface.

## Included scope

- City of Bristol local-authority area, identified using LAD code `E06000023`.
- Transactions dated January 2022 to December 2024.
- Standard Category A residential transactions for model fitting and evaluation.
- Detached, semi-detached, terraced and flat/maisonette property types.
- Property type, tenure, new-build status and time controls.
- Monthly total crime and claimant rate at 2011 LSOA level.
- Explanatory and predictive model comparisons.
- Historical 2024 dashboard estimates.

## Excluded scope

- Current or future property-market forecasts.
- Professional valuations or mortgage decisions.
- Causal estimates of the effect of crime or unemployment on prices.
- Properties outside the City of Bristol LAD.
- Bedrooms, floor area, condition, energy performance and amenities, which were not available in the selected source data.
- A production deployment with continuously updated source data.

## Artefact definition

The artefact is the complete reproducible analytical system rather than the CSV or dashboard alone. It comprises the preparation pipeline, validated data, analysis code, evaluated models, selected model bundle, user interface, automated checks and documentation.
