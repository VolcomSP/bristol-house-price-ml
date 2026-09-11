# Data dictionary

## Dataset level

File: `outputs/bristol_house_prices_combined_2022_2024.csv`

Each row represents one HM Land Registry property transaction whose postcode maps to the City of Bristol local-authority area. The table contains 21,968 rows and 53 columns covering January 2022 to December 2024.

## Columns

| Column | Type or format | Meaning |
|---|---|---|
| `transaction_id` | Text | Unique Land Registry transaction reference. |
| `price` | Integer, GBP | Recorded sale price in pounds. The primary model target is `log(price)`. |
| `date_of_transfer` | `YYYY-MM-DD` | Date ownership was transferred. |
| `sale_year` | Integer | Calendar year extracted from the transfer date. |
| `sale_month` | `YYYY-MM` | Calendar month used for monthly feature matching and time splits. |
| `postcode` | Text | Uppercase property postcode stored without spaces in the modelling table. |
| `property_type` | Category code | `D` detached, `S` semi-detached, `T` terraced, `F` flat/maisonette, `O` other. |
| `property_type_label` | Text | Written label corresponding to `property_type`. |
| `old_new` | Category code | `Y` newly built; `N` established property. |
| `duration` | Category code | `F` freehold; `L` leasehold; `U` unknown where present in source data. |
| `paon` | Text | Primary addressable object name or number, such as a house number or building name. |
| `saon` | Text, optional | Secondary addressable object, such as a flat number. |
| `street` | Text | Street name recorded by Land Registry. |
| `locality` | Text, optional | Locality component of the Land Registry address. |
| `town_city` | Text | Town or city recorded by Land Registry. |
| `district` | Text | District text recorded in Price Paid Data. Bristol selection instead uses `lad_code`. |
| `county` | Text | County text recorded in Price Paid Data. |
| `ppd_category` | Category code | `A` standard residential transaction; `B` additional transaction not meeting every Category A condition. |
| `record_status` | Category code | Land Registry update status: `A` added, `C` changed or `D` deleted. |
| `lad_code` | Text code | Local Authority District code from the postcode directory. All included rows use Bristol code `E06000023`. |
| `lsoa11cd` | Text code | Property's Lower Layer Super Output Area under 2011 boundaries. |
| `msoa11cd` | Text code | Property's Middle Layer Super Output Area under 2011 boundaries. |
| `lsoa21cd` | Text code | Property's Lower Layer Super Output Area under 2021 boundaries. |
| `msoa21cd` | Text code | Property's Middle Layer Super Output Area under 2021 boundaries. |
| `crime_lsoa_vintage` | Text | Statistical geography used for the standardised crime join. It is `LSOA11` throughout the final data. |
| `crime_lsoa_code` | Text code | LSOA code used to join monthly crime. It equals the property's 2011 LSOA. |
| `analysis_lsoa_code` | Text code | Common 2011 LSOA used by the main analysis and claimant join. |
| `latitude` | Decimal degrees | Postcode latitude supplied by the postcode directory. |
| `longitude` | Decimal degrees | Postcode longitude supplied by the postcode directory. |
| `postcode_status` | Category | `Active` or `Terminated` according to the May 2026 postcode directory. |
| `postcode_introduction_yyyymm` | `YYYYMM` | Month in which the postcode was introduced. |
| `postcode_termination_yyyymm` | `YYYYMM`, optional | Month in which the postcode was terminated; blank for non-terminated postcodes. |
| `crime_data_matched` | Boolean | Whether a complete LSOA-month crime feature row matched the transaction. |
| `total_crime` | Integer count | Total standardised Police.uk street-crime records in the property's 2011 LSOA during the sale month. |
| `anti_social_behaviour_count` | Integer count | Anti-social-behaviour records in the LSOA and month. |
| `bicycle_theft_count` | Integer count | Bicycle-theft records in the LSOA and month. |
| `burglary_count` | Integer count | Burglary records in the LSOA and month. |
| `criminal_damage_and_arson_count` | Integer count | Criminal-damage and arson records in the LSOA and month. |
| `drugs_count` | Integer count | Drug-offence records in the LSOA and month. |
| `other_crime_count` | Integer count | Records classified as other crime in the LSOA and month. |
| `other_theft_count` | Integer count | Records classified as other theft in the LSOA and month. |
| `possession_of_weapons_count` | Integer count | Possession-of-weapons records in the LSOA and month. |
| `public_order_count` | Integer count | Public-order records in the LSOA and month. |
| `robbery_count` | Integer count | Robbery records in the LSOA and month. |
| `shoplifting_count` | Integer count | Shoplifting records in the LSOA and month. |
| `theft_from_the_person_count` | Integer count | Theft-from-the-person records in the LSOA and month. |
| `vehicle_crime_count` | Integer count | Vehicle-crime records in the LSOA and month. |
| `violence_and_sexual_offences_count` | Integer count | Violence and sexual-offence records in the LSOA and month. |
| `claimant_data_matched` | Boolean | Whether a complete LSOA-month claimant feature row matched the transaction. |
| `claimant_count` | Integer count | Nomis Claimant Count for the property's 2011 LSOA and sale month, rounded by the source to the nearest five. |
| `working_age_population_2020` | Integer count | 2020 estimated population aged 16–64 in the 2011 LSOA. Held fixed across the study. |
| `claimant_rate_percent` | Percentage | `claimant_count / working_age_population_2020 × 100`. |
| `population_reference_year` | Integer year | Reference year for the population denominator; `2020`. |

## Model use

| Role | Variables |
|---|---|
| Outcome | `price`, transformed to `log(price)` during modelling |
| Property inputs | `property_type`, `duration`, `old_new` |
| Time inputs | Derived month index plus sine and cosine month-of-year terms |
| Enriched inputs | `log1p(total_crime)` and `claimant_rate_percent` |
| Lagged final-model inputs | Previous calendar month's `log1p(total_crime)` and `claimant_rate_percent` |
| Matching and audit fields | Postcode, LAD, LSOA/MSOA, coordinates and match flags |
| Not used as predictors | Transaction ID, price address components, record status, latitude, longitude and explicit area identifiers |

## Important distinctions

The 2011 and 2021 LSOA/MSOA columns describe the same property location under two boundary systems. They do not represent four separate places. The main analysis uses `analysis_lsoa_code`, which is consistently a 2011 LSOA.

The crime counts are neighbourhood-month totals, not crimes attributed to the sold property. Claimant rate is also a neighbourhood statistic and must not be interpreted as information about the buyer, seller or residents of the address.
