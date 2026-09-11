# Data sources

## Source inventory

| Source | Project use | Period/version | Included in submission |
|---|---|---|---|
| HM Land Registry Price Paid Data | Transaction price, date, property type, tenure, new-build status and address fields | 2022–2024 annual files | Bristol processed extract included; national annual raw files excluded because of size |
| Police.uk open street-level crime data | Monthly crime counts and crime-type counts | January 2022–December 2024 | Aggregated Bristol LSOA-month table included; 36 raw force files excluded because of size |
| ONS Postcode Directory | Postcode-to-LAD, LSOA and MSOA mapping and coordinates | May 2026 | Validated BS-postcode extract included |
| ONS 2011 LSOA boundaries | Coordinate assignment to consistent 2011 LSOAs | BFC V3 | Bristol-only boundary extract included |
| Nomis Claimant Count | Monthly claimant counts | January 2022–December 2024 | Raw Bristol LSOA extract and processed features included |
| Nomis population estimates | Working-age population denominator | 2020 | Raw Bristol LSOA extract and processed values included |

## Official access points

- HM Land Registry Price Paid Data downloads: <https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads>
- Annual files used by the validation scripts: `https://price-paid-data.publicdata.landregistry.gov.uk/pp-YYYY.csv`
- Police.uk data downloads: <https://data.police.uk/data/>
- ONS Postcode Directory: ONS Open Geography Portal, May 2026 hosted table
- 2011 LSOA boundary catalogue record: <https://www.data.gov.uk/dataset/573bb7e5-9a05-42ac-a404-4d29dab11675/lower-layer-super-output-areas-december-2011-boundaries-ew-bgc-v31>
- Nomis Claimant Count dataset: <https://www.nomisweb.co.uk/datasets/ucjsa>
- Nomis small-area population estimates: <https://www.nomisweb.co.uk/datasets/pestsyoaoa>

## Source-specific treatment

### Land Registry

The annual England and Wales files were validated for file completeness, year, dates, transaction identifiers and malformed rows. Bristol was selected using the postcode directory's LAD code rather than relying only on the Land Registry district text.

The complete project data contains 21,968 Bristol transactions: 8,093 in 2022, 6,839 in 2023 and 7,036 in 2024.

### Police.uk

Thirty-six Avon and Somerset street-crime files were inspected. Fourteen originally supplied files were truncated and were replaced with complete official copies before feature construction. Available nonblank crime IDs were deduplicated within month. Anonymous anti-social-behaviour records were retained because each source row represents an incident.

All usable crime coordinates and recognised source codes were assigned to the same 2011 Bristol LSOA geography. The processed panel contains 263 LSOAs for each of 36 months.

### ONS postcode and boundary data

The May 2026 BS-postcode extract contains active and terminated postcodes so historical transactions can be mapped. The final mapping represents 263 Bristol 2011 LSOAs. The boundary used for coordinate conversion is the Bristol-only full-resolution, clipped 2011 LSOA layer in British National Grid coordinates.

Using a 2026 directory for 2022–2024 transactions is a limitation: it provides historical termination fields, but later postcode revisions may not perfectly reproduce the directory as it existed on each sale date.

### Nomis claimant and population data

Claimant counts are rounded to the nearest five. The rate is calculated as:

\[
\text{Claimant rate (\%)} = \frac{\text{Claimant count}}{\text{2020 population aged 16--64}} \times 100
\]

The 2020 estimate is the final official population estimate available on 2011 LSOA boundaries and is held fixed across 2022–2024.

The Claimant Count source records discontinuities caused by changes to the Universal Credit administrative earnings threshold in September 2022, January 2023 and May 2024. Changes in the series must therefore not be interpreted entirely as changes in underlying economic conditions.

## Attribution and reuse

The repository retains source names and URLs so the original licences and attribution requirements can be checked at submission time. Official data remains subject to the terms stated by each publisher. The raw national or force-level downloads are not redistributed in the submission package.
