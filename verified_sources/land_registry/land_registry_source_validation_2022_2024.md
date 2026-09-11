# HM Land Registry source validation: 2022–2024

Validation completed on 31 August 2026 for the Bristol house-price project.

## Purpose

The annual HM Land Registry Price Paid Data files were checked before rebuilding the modelling dataset. The checks confirm that the raw sources are complete, structurally valid, cover the intended period and can be matched consistently to Bristol using the validated ONS postcode lookup.

## Results

| Check | 2022 | 2023 | 2024 |
|---|---:|---:|---:|
| File size (bytes) | 187,868,906 | 150,324,812 | 162,267,126 |
| National rows | 1,077,001 | 861,263 | 930,559 |
| Unique transaction IDs | 1,077,001 | 861,263 | 930,559 |
| Malformed rows | 0 | 0 | 0 |
| Duplicate IDs within year | 0 | 0 | 0 |
| Rows dated outside expected year | 0 | 0 | 0 |
| Invalid prices | 0 | 0 | 0 |
| Months present | 12 | 12 | 12 |
| Bristol LAD rows identified by postcode | 8,093 | 6,839 | 7,036 |
| Rows in previous project output | 8,091 | 5,731 | 7,011 |
| New-source Bristol IDs absent from previous output | 3 | 1,108 | 27 |
| Previous-output IDs absent from current source | 1 | 0 | 2 |
| Complete final row/newline | Yes | Yes | Yes |

## Reconciliation

- Verified national rows across all three years: **2,868,823**.
- Verified Bristol transactions across all three years: **21,968**.
- Duplicate transaction IDs across different annual files: **0**.
- The existing project outputs were not modified during validation.
- The 2023 comparison is unusually large because the existing combined output stops in October 2023; the verified source includes November and December plus later source updates.
- The current verified total is 31 records higher than the earlier expected total of 21,937. This reflects later HM Land Registry updates: net +2 in 2022, +4 in 2023 compared with the earlier inferred full-year total, and +25 in 2024.

## Decision

Use the verified annual sources and the ONS postcode-to-LAD mapping when rebuilding the Bristol property dataset. Retain the older outputs only as comparison evidence; do not combine their transaction lists with the current official sources.

## Source

HM Land Registry, Price Paid Data yearly files: https://www.gov.uk/government/statistical-data-sets/price-paid-data-yearly-file
