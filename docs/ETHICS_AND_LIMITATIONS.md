# Ethics and limitations

## Data and privacy

The project uses publicly available administrative and statistical datasets. Land Registry records contain property addresses and transaction identifiers, but the model does not use owner names or other direct personal identifiers. Crime and claimant variables are aggregated to LSOAs and do not identify individuals.

Only the information needed for analysis should be published. The dashboard validation file replaces raw transaction identifiers with irreversible short case labels. The interface displays neighbourhood aggregates rather than individual crime or claimant records.

Public availability does not remove the need for responsible use. Exact address fields are retained in the validated research data for auditability but are not dashboard inputs and should not be used to infer information about residents.

## Responsible interpretation

The artefact estimates statistical associations. It must not be used to claim that crime or claimant conditions cause property values to rise or fall. Both measures may stand in for unobserved location, housing stock, accessibility, services and amenities.

Neighbourhood labels can contribute to stigma. Results should be reported at an aggregate level, with uncertainty and limitations, and should not be used to rank residents or make decisions about individual eligibility, lending, insurance or policing.

## Bias and representativeness

- Price Paid Data records completed transactions rather than all housing or unsuccessful listings.
- Category A modelling excludes non-standard Category B transactions.
- Police-recorded crime depends on reporting, recording practice, policing activity and anonymisation.
- Crime counts may be elevated in commercial or city-centre areas because of visitors and daytime populations.
- Claimant Count does not measure every form of unemployment or economic hardship.
- Changes to Universal Credit administrative earnings thresholds created discontinuities in September 2022, January 2023 and May 2024.
- The 2020 working-age population denominator is held fixed and does not capture later population change.
- The data covers one city and three years, so results should not be generalised automatically to other places or periods.

## Model limitations

Land Registry data omits bedrooms, floor area, condition, energy efficiency, garden, parking, school access and detailed amenities. These omissions constrain accuracy and mean that neighbourhood features may absorb some unmeasured property and location information.

Random forests interpolate from observed patterns and do not extrapolate reliably to new market regimes. The dashboard is therefore restricted to historical 2024 estimates rather than presenting a current 2026 valuation.

The 2024 test MAE of approximately £99,655 is a model-wide average absolute error. It is not a confidence interval and does not mean each estimate lies within that amount of the true price.

## Geographic limitations

All neighbourhood features use 2011 LSOAs to avoid mixing boundary systems. This improves time consistency but retains an older statistical geography. The May 2026 postcode directory is used to map 2022–2024 transactions; terminated-postcode fields help, but directory revisions may not perfectly reproduce the lookup available at the time of sale.

Police coordinates are anonymised and some source rows lack usable coordinates. Only records that can be assigned reliably to Bristol 2011 LSOAs contribute to the neighbourhood counts.

## Security and deployment

The local Streamlit tool performs no login, payment or automated decision. It should remain a demonstration unless source-update processes, security controls, monitoring and model-governance procedures are designed and approved.

## Academic integrity

All source data and external literature must be referenced using the required UWE Harvard approach. Any use of generative AI must be checked against the current assessment rules, acknowledged where required and recorded accurately. The student remains responsible for understanding the code, verifying the evidence and writing the submitted report in their own words.
