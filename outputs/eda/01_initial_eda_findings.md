# Initial exploratory data analysis

## Scope

This first EDA checkpoint assesses dataset quality, study-period coverage and the sale-price distribution. It does not remove or modify any property transaction.

## Dataset quality

- Transactions: **21,968**
- Columns in the modelling dataset: **53**
- Period: **2022-01-04 to 2024-12-31**
- Months represented: **36**
- Bristol 2011 LSOAs represented: **263**
- Duplicate transaction IDs: **0**
- Exact duplicate rows: **0**
- Missing values across the initial modelling variables: **0**
- Unmatched crime rows: **0**
- Unmatched claimant rows: **0**

Blank values remain in optional address and postcode-history fields such as secondary address, locality and postcode termination date. These are structurally optional and are not required as model inputs.

## Price distribution

| Statistic | Sale price |
|---|---:|
| Minimum | £100 |
| 1st percentile | £81,335 |
| 25th percentile | £255,000 |
| Median | £330,500 |
| Mean | £449,610 |
| 75th percentile | £436,306 |
| 99th percentile | £1,695,000 |
| Maximum | £78,296,822 |

Raw-price skewness is **32.53**, compared with **-1.29** after applying the natural logarithm. The transformation therefore reduces the extreme right skew substantially, although the transformed distribution remains moderately left-skewed.

## Records requiring modelling decisions

- PPD Category B transactions: **3,298**
- `Other` property types: **1,221**
- Prices below £25,000: **99**
- Prices above £2,000,000: **172**
- Prices above the conventional upper IQR threshold of £708,266: **1,763**

These records are flagged for investigation; they have not been deleted. All **99** transactions below £25,000 are both PPD Category B and property type `Other`. The 20 lowest-priced and 20 highest-priced transactions are also all Category B/`Other`. Their address descriptions include whole buildings and commercial-style transactions, so they are not directly comparable with ordinary individual residential sales. Category A contains no `Other` property records.

## Initial modelling recommendation

1. Use **log sale price** as the primary modelling target because it materially reduces skewness and limits the influence of extremely expensive transactions.
2. Define the main residential modelling cohort as **PPD Category A**. In this dataset that produces **18,670 transactions (84.99% of all sales)** and automatically restricts the data to standard D, S, T and F property types.
3. Do not automatically remove the highest and lowest prices. Compare results using the full main cohort with a sensitivity analysis restricted to the 1st–99th percentile range.
4. Retain the complete validated dataset unchanged and create modelling subsets reproducibly in later scripts.

## Outputs

- `01_dataset_quality_summary.csv`: auditable quality and price statistics.
- `01_missingness_summary.csv`: missing-value counts for every column.
- `01_monthly_summary.csv`: monthly transaction and price summaries.
- `01_candidate_cohort_summary.csv`: row counts under alternative modelling rules.
- `01_price_distribution_raw_vs_log.png`: price transformation evidence.
- `01_monthly_sales_and_median_price.png`: study-period coverage and monthly median prices.
