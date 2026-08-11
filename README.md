# 2024 HMDA Loan Equity Analysis: Race & Gender

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Library-Pandas-orange.svg)](https://pandas.pydata.org/)

## Overview
This repository contains an exploratory analysis of the 2024 Home Mortgage Disclosure Act (HMDA) dataset. The project focuses on race- and gender-based differences in mortgage approval rates and loan pricing for single-party applicants.

## What this project does
- Compares loan approval rates across derived race and gender groups.
- Visualizes rate spread, debt-to-income ratio, loan amount, and loan-to-value ratio distributions by race and gender.
- Produces visualizations and summary tables from a curated HMDA subset.

## Tech stack
- Data processing: Pandas, NumPy
- Visualization: Seaborn, Matplotlib

## Repository structure
```text
.
├── README.md
├── Enviro_Run.md
├── HMDA_Loan_Equity_Gender_Race_Artifact2.ipynb
├── HMDA_loan_equity_gender_race_artifact2.py
├── load_hf_dataset.py
├── data_dictionary.md
├── requirements.txt
├── .gitignore
├── LICENSE
└── .github/workflows/python-ci.yml
```

## Getting started

### 1. Create a local environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

### 2. Get the HMDA data
The full raw HMDA file isn't in this repo — it's published separately on Hugging Face at https://huggingface.co/datasets/retrogradespace/hmda_2024. Fetch it and export it to a local CSV:

```bash
python3 -m pip install datasets
python3 load_hf_dataset.py
```

This downloads the dataset and saves it to `~/.cache/hmda_2024/hmda_2024_lar.csv` by default (override with `HMDA_EXPORT_PATH`). Point `HMDA_DATA_PATH` at that file — or at your own HMDA source CSV instead:

```bash
export HMDA_DATA_PATH="$HOME/.cache/hmda_2024/hmda_2024_lar.csv"
```

Keep in mind this is a large 4.8GB file. Running this in Colab Notebook and pointing to Google Drive folder worked best for me. 

### 3. Run the analysis
The script calls `plt.show()` for each figure. On a machine with a display this pops up a window per plot; running headlessly (CI, a remote server, or just to avoid babysitting ~15 windows) set `MPLBACKEND=Agg` so it renders without blocking:
```bash
MPLBACKEND=Agg python3 HMDA_loan_equity_gender_race_artifact2.py
```

You can also open the notebook in Jupyter for a step-by-step walkthrough (`pip install jupyter` first — it isn't in `requirements.txt`).

## Data
- The full raw HMDA file is published as a separate data release on Hugging Face at https://huggingface.co/datasets/retrogradespace/hmda_2024/tree/main. See "Get the HMDA data" above for how to fetch it. You can also find it from the source here: https://www.consumerfinance.gov/data-research/hmda/. 
- Retrograde Space. (2024). HMDA 2024. Hugging Face Datasets. https://huggingface.co/datasets/retrogradespace/hmda_2024
- A companion data dictionary is available in [data_dictionary.md](data_dictionary.md).

## Notes
- The analysis uses a curated subset of HMDA fields such as derived sex, derived ethnicity, derived race, loan-to-value ratio, debt-to-income ratio, rate spread, loan amount, occupancy type, loan type, loan purpose, and action taken.

## License
This project is distributed under the terms of the included license.

