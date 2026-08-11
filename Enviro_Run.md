# HMDA Loan Characteristics by Race & Gender
This repository contains exploratory analysis of the 2024 Home Mortgage Disclosure Act (HMDA) loan application data. The analysis focuses on visualizing rate spread, debt-to-income ratio, loan amount, and loan-to-value ratio by derived race and derived sex.

## Overview
The workflow processes a large HMDA source file and filters for a relevant subset of applicants and property types before generating figures and summary statistics.

## Setup
A high-memory environment is recommended for the full dataset because some steps, such as KDE plots, can be computationally intensive.

## Installation
Create a local virtual environment in the repository root and install the required packages:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For local script execution, use a writable Matplotlib cache directory. On a headless or remote environment (no display attached), also set `MPLBACKEND=Agg` so the script's `plt.show()` calls don't block waiting for a window to close:

```bash
MPLCONFIGDIR="$PWD/.mplcache" MPLBACKEND=Agg python HMDA_loan_equity_gender_race_artifact2.py
```

## Data Input
The full raw HMDA file isn't in this repo — it's published separately on Hugging Face at https://huggingface.co/datasets/retrogradespace/hmda_2024. Fetch it and export it to a local CSV:

```bash
python3 load_hf_dataset.py
```

This saves the file to `~/.cache/hmda_2024/hmda_2024_lar.csv` by default (override with `HMDA_EXPORT_PATH`). Point the workflow at it — or at your own HMDA source file — by setting `HMDA_DATA_PATH` or by updating the file path in the notebook or script:

```bash
export HMDA_DATA_PATH="$HOME/.cache/hmda_2024/hmda_2024_lar.csv"
```

## Data Notes
The analysis uses a subset of HMDA fields such as derived sex, derived ethnicity, derived race, loan-to-value ratio, debt-to-income ratio, rate spread, loan amount, occupancy type, loan type, loan purpose, and action taken.

Data cleaning typically removes categories such as sex not available, joint, ethnicity not available, free-form text only, and race not available. Outliers in numeric variables are also filtered using the 1st and 99th percentiles.

## Analysis Highlights
- Comparison of loan approval rates across racial and gender groups
- Visualization of rate spread distributions by race and gender
- Analysis of loan amounts, debt-to-income ratios, and loan-to-value ratios across demographic groups
- 2D KDE plots for joint distributions such as rate spread versus debt-to-income ratio




