# Data dictionary

## Source file
- Name: `hmda_2024.parquet` (published at [huggingface.co/datasets/retrogradespace/hmda_2024](https://huggingface.co/datasets/retrogradespace/hmda_2024))
- Format: columnar parquet

## Key columns
- `derived_sex`: applicant sex category as reported in HMDA
- `derived_ethnicity`: applicant ethnicity category as reported in HMDA
- `derived_race`: applicant race category as reported in HMDA
- `loan_to_value_ratio`: loan-to-value ratio
- `debt_to_income_ratio`: debt-to-income ratio
- `rate_spread`: rate spread value
- `interest_rate`: interest rate value
- `loan_amount`: loan amount
- `occupancy_type`: occupancy type of the property
- `loan_type`: loan type
- `loan_purpose`: loan purpose
- `action_taken`: action taken on the application

## Notes
- Values may include categories such as `not available` or `joint` and may need filtering before analysis.
- Review the official HMDA documentation for full definitions and any changes across reporting years. https://www.consumerfinance.gov/data-research/hmda/
