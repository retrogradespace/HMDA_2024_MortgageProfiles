# -*- coding: utf-8 -*-
"""HMDA Loan Equity Analysis - Artifact 2

This script reproduces the exploratory analysis workflow for the 2024 HMDA dataset,
including visualizations of rate spread, debt-to-income ratio, loan amount, and
loan-to-value ratio by derived race and derived sex.

The original notebook workflow has been generalized here so the repository can be
shared without relying on a specific local or Colab path.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Load dataset from a configurable local path.
# Set HMDA_DATA_PATH to your source file location before running.
import os
from pathlib import Path

DATA_FILE = os.environ.get('HMDA_DATA_PATH', 'year_2024.csv')
print(f"Using data file: {DATA_FILE}")

OUT_FIG_DIR = 'output/figures'
OUT_TAB_DIR = 'output/tables'
OUT_CKP_DIR = 'output/checkpoints'

Path(OUT_FIG_DIR).mkdir(parents=True, exist_ok=True)


for d in (OUT_FIG_DIR, OUT_TAB_DIR, OUT_CKP_DIR):
    Path(d).mkdir(parents=True, exist_ok=True)

def safe_save_csv(df, path, **kwargs):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, **kwargs)
    print(f"Saved: {p}")

def safe_save_fig(path, dpi=300):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib.pyplot as plt
    plt.savefig(p, dpi=dpi)
    print(f"Saved: {p}")

# Validate the configured path.

hdma2024 = DATA_FILE
assert os.path.exists(hdma2024), f"File not found: {hdma2024}. Please set HMDA_DATA_PATH to the correct source file."

print("Paths OK.")

# Preview the raw file without loading the full (potentially multi-GB,
# ~100-column) file into memory - only the columns used below are read in full.

preview_df = pd.read_csv(hdma2024, dtype=str, nrows=5000, low_memory=False)

# This is necessary if using matplotlib or seaborn where '-' is an operator and its presence in the name of a variable will throw issues.
# Also generally good data practive to use underscores and lowercase for table naming conventions

preview_df.columns = [col.replace('-', '_') for col in preview_df.columns]

with open(hdma2024) as f:
    row_count = sum(1 for _ in f) - 1  # exclude header

print("Dataset Shape:", (row_count, len(preview_df.columns)))
print("Columns:", preview_df.columns.tolist())

pd.unique(preview_df.columns)

preview_df.describe()

preview_df.info()
rawdatahead = preview_df.head()

# Write sample to CSV for easy review
safe_save_csv(rawdatahead, f'{OUT_TAB_DIR}/rawdatahead.csv')

# Columns to load for efficiency
use_cols = [
    'derived_sex', 'derived_ethnicity', 'derived_race',
    'loan_to_value_ratio', 'debt_to_income_ratio', 'rate_spread',
    'interest_rate', 'loan_amount', 'occupancy_type', 'loan_type',
    'loan_purpose', 'action_taken'
]

# Load in chunks to avoid memory overload
chunks = []
chunk_size = 1_000_000  # Adjust if needed
for chunk in pd.read_csv(hdma2024, usecols=use_cols, chunksize=chunk_size, low_memory=False):
    chunks.append(chunk)

derived_df = pd.concat(chunks, ignore_index=True)

# Rename columns: replace '-' with '_'
derived_df.columns = [col.replace('-', '_') for col in derived_df.columns]

print("Data Loaded:", derived_df.shape)
print("Columns:", derived_df.columns.tolist())

df_noNA = derived_df.dropna(subset=['derived_sex', 'derived_ethnicity', 'derived_race', 'action_taken']).copy()

# Normalize categorical values
categorical_cols = ['derived_sex', 'derived_ethnicity', 'derived_race',
                    'occupancy_type', 'loan_type', 'loan_purpose']
for col in categorical_cols:
    df_noNA[col] = df_noNA[col].astype(str).str.lower()

# Define categories to exclude after lowercasing
exclude_derived_sex = ['sex not available', 'joint']
exclude_derived_ethnicity = ['ethnicity not available', 'joint', 'free form text only']
exclude_derived_race = ['race not available', 'joint', '2 or more minority races', 'free form text only']

# Apply exclusions
df_noNA = df_noNA[~df_noNA['derived_sex'].isin(exclude_derived_sex)]
df_noNA = df_noNA[~df_noNA['derived_ethnicity'].isin(exclude_derived_ethnicity)]
df_noNA = df_noNA[~df_noNA['derived_race'].isin(exclude_derived_race)]

# Convert numeric columns on a copy of df_noNA for sampledf
sampledf = df_noNA.copy()
numeric_cols = ['loan_to_value_ratio', 'debt_to_income_ratio', 'rate_spread',
                'interest_rate', 'loan_amount']
for col in numeric_cols:
    sampledf[col] = pd.to_numeric(sampledf[col], errors='coerce')

# Drop rows with missing numeric values
sampledf = sampledf.dropna(subset=numeric_cols)

print("Cleaned Data Shape:", sampledf.shape)

print("===== Describe Data =====")
print(sampledf.describe(include='all'))

print(sampledf.derived_sex.value_counts())
print(sampledf.derived_ethnicity.value_counts())
print(sampledf.derived_race.value_counts())

# COME BACK to create a Race/Ethnicity variable using only race masks experience of Hispanic and Latino



# Only considering single applicants
# clear limitation --> think about how to approach
#

import matplotlib.pyplot as plt
import seaborn as sns


# Set a style
sns.set(style="darkgrid")

# Create the scatter plot
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=sampledf,
    x='debt_to_income_ratio',
    y='rate_spread',
    hue='derived_race',  # Color by race
    style='derived_sex',          # Shape by gender
    alpha=0.6,
    s=50                     # Size of markers
)

plt.title('Rate Spread vs Debt-to-Income Ratio by Race and Gender')
plt.xlabel('Debt-to-Income Ratio')
plt.ylabel('Rate Spread')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
plt.grid(True)
plt.tight_layout()
plt.show()


# Later examine this outlier when have time

# Look at data again more closely at 1% & 99% CI -->
# The default confidence internals sampledf.describe() is 25% 50% 75% need more precision
# Define percentiles as desired

from numpy.random import sample
# Inspect unique race categories and their counts
print("Unique derived_race categories and their counts:")
print(sampledf['derived_race'].value_counts())
print(f"Number of unique race categories: {sampledf['derived_race'].nunique()}")

# Inspect descriptive statistics for the numeric variables
print("\nDescriptive statistics for debt_to_income_ratio:")
print(sampledf['debt_to_income_ratio'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

print("\nDescriptive statistics for rate_spread:")
print(sampledf['rate_spread'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

print("\nDescriptive statistics for loan_amount")
print(sampledf['loan_amount'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

print("\nDescriptive statistics for loan_to_value_ratio")
print(sampledf['loan_to_value_ratio'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

safe_save_csv(sampledf.derived_race.value_counts(), f'{OUT_TAB_DIR}/derived_race_counts.csv')
safe_save_csv(sampledf.debt_to_income_ratio.describe(percentiles=[.01, .05, .25, .50, .75, .99]), f'{OUT_TAB_DIR}/debt_to_income_ratio_stats.csv')

"""## 2) Data Clean-Up

Creating a new DataFrame `df_filtered` by filtering out extreme outliers from `sampledf` using the 1st and 99th percentiles for `debt_to_income_ratio`, `loan_to_value_ratio`, `loan_amount` and `rate_spread`.
"""

dti_lower_bound = sampledf['debt_to_income_ratio'].quantile(0.01)
dti_upper_bound = sampledf['debt_to_income_ratio'].quantile(0.99)
rate_spread_lower_bound = sampledf['rate_spread'].quantile(0.01)
rate_spread_upper_bound = sampledf['rate_spread'].quantile(0.99)
loan_amount_lower_bound = sampledf['loan_amount'].quantile(0.01)
loan_amount_upper_bound = sampledf['loan_amount'].quantile(0.99)
loan_to_value_lower_bound = sampledf['loan_to_value_ratio'].quantile(0.01)
loan_to_value_upper_bound = sampledf['loan_to_value_ratio'].quantile(0.99)

df_filtered = sampledf[
    (sampledf['debt_to_income_ratio'] >= dti_lower_bound) &
    (sampledf['debt_to_income_ratio'] <= dti_upper_bound) &
    (sampledf['rate_spread'] >= rate_spread_lower_bound) &
    (sampledf['rate_spread'] <= rate_spread_upper_bound) &
    (sampledf['loan_amount'] >= loan_amount_lower_bound) &
    (sampledf['loan_amount'] <= loan_amount_upper_bound) &
    (sampledf['loan_to_value_ratio'] >= loan_to_value_lower_bound) &
    (sampledf['loan_to_value_ratio'] <= loan_to_value_upper_bound)
].copy()

# Ensure 'derived_sex' column is correctly defined (using derived_sex as it exists in sampledf)
df_filtered['derived_sex'] = df_filtered['derived_sex'].astype('category')

# Ensure 'derived_race' is included and set as a categorical type
df_filtered['derived_race'] = df_filtered['derived_race'].astype('category')

print(f"Shape of df_filtered after outlier removal: {df_filtered.shape}")
print("\nDescriptive statistics for debt_to_income_ratio in df_filtered:")
print(df_filtered['debt_to_income_ratio'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

print("\nDescriptive statistics for rate_spread in df_filtered:")
print(df_filtered['rate_spread'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

print("\nDescriptive statistics for loan_amount in df_filtered:")
print(df_filtered['loan_amount'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

print("\nDescriptive statistics for loan_to_value_ratio in df_filtered:")
print(df_filtered['loan_to_value_ratio'].describe(percentiles=[.01, .05, .25, .50, .75, .95, .99]))

"""Now that I have excluded outliers re-plotting using the `df_filtered` DataFrame to visualize the relationship between `rate_spread` and `debt_to_income_ratio`, colored by `derived_race` and styled by `derived_sex`.


"""

import matplotlib.pyplot as plt
import seaborn as sns

# Set a style
sns.set(style="darkgrid")

# Create the scatter plot using df_filtered
plt.figure(figsize=(12, 8))
sns.scatterplot(
    data=df_filtered,
    x='debt_to_income_ratio',
    y='rate_spread',
    hue='derived_race',  # Color by race
    style='derived_sex',          # Shape by gender
    alpha=0.6,
    s=50                     # Size of markers
)

plt.title('Rate Spread vs Debt-to-Income Ratio by Race and Gender (Outliers Removed)')
plt.xlabel('Debt-to-Income Ratio')
plt.ylabel('Rate Spread')
plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left') # Place legend outside the plot
plt.grid(True)
plt.tight_layout()
plt.show()

"""Opps, that really didn't tell me much given that `debt_to_income_ratio` is discrete so I will attempt a different approach using a density plot."""

import matplotlib.pyplot as plt
import seaborn as sns # https://seaborn.pydata.org/tutorial/distributions.html

# This was my first attempt at using Seaborn which is built on matplotlib and pandas.
# I am still working to update legends and formats.
# Check back for updates to the GitHub repo where they will be published.

sns.set(style="darkgrid")

g = sns.FacetGrid(df_filtered, col="derived_sex", hue="derived_race", height=6, aspect=1.2, col_wrap=2, palette="tab20")
g.map(sns.kdeplot, "debt_to_income_ratio", "rate_spread", fill=True, levels=5)
g.add_legend(title='Applicant Race', bbox_to_anchor=(1.05, 1), loc='upper left') #the legend color is not rendering on the legend adjecent labels FIX THIS

for ax in g.axes.flat:
    ax.set_title(f'Rate Spread vs Debt-to-Income Ratio for {ax.get_title().split("=")[1].strip()}')
    ax.set_xlabel('Debt-to-Income Ratio')
    ax.set_ylabel('Rate Spread')

plt.suptitle('2D KDE of Rate Spread vs Debt-to-Income Ratio by Race and Gender', y=1.02, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
safe_save_fig(f'{OUT_FIG_DIR}/kde_rate_spread_dti_race_gender.png')

plt.show()

"""Even though I have not figured out the legends just yet this is getting abit closer to what I want to visualized. I can see there is a clear difference between groups though not in a way that provides me insights just yet. Also on save these are not rendering. Will need to come back to address later"""

# Replot using FacetGrid.set_titles calling race variable as 'row'
# Legend is still an issue why?

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

g = sns.FacetGrid(df_filtered, row="derived_race", col="derived_sex", height=4, aspect=1.5, hue="derived_race", palette="tab20", sharex=False, sharey=False)
g.map(sns.kdeplot, "debt_to_income_ratio", "rate_spread", fill=True, levels=5)
g.add_legend(title='Applicant Race', bbox_to_anchor=(1.05, 1), loc='upper left')

# Set titles for each facet correctly using row_name and col_name
# https://seaborn.pydata.org/generated/seaborn.FacetGrid.set_titles.html
g.set_titles(row_template='Race: {row_name}', col_template='Gender: {col_name}')

# Set common x and y labels if needed, otherwise FacetGrid handles it
g.set_xlabels('Debt-to-Income Ratio')
g.set_ylabels('Rate Spread')

plt.suptitle('2D KDE of Rate Spread vs Debt-to-Income Ratio by Race and Gender', y=1.02, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
safe_save_fig(f'{OUT_FIG_DIR}/kde_rate_spread_dti_race_gender_grid.png')
plt.show()

from numpy.random import sample
# Inspect unique race categories and their counts
print("Unique derived_race categories and their counts:")
print(df_filtered['derived_race'].value_counts())
print(f"Number of unique race categories: {sampledf['derived_race'].nunique()}")

print("Unique derived_gender categories and their counts:")
print(df_filtered['derived_sex'].value_counts())
print(f"Number of unique gender categories: {sampledf['derived_race'].nunique()}")

"""## 3) Loan Approval Rates by Rate and Sex in 2024

Calculate approval rates for each group and then provide side-by-side comparison.

### 3.a. Loan Approval Rates by Race for Women
"""

df_women = df_filtered[df_filtered['derived_sex'] == 'female'].copy()
df_women['approved'] = (df_women['action_taken'] == 1).astype(int)

print("df_women shape:", df_women.shape)
print("Approved column created and converted to integer type.")

approval_rates_women_by_race = df_women.groupby('derived_race', observed=False)['approved'].mean().reset_index()
approval_rates_women_by_race.rename(columns={'approved': 'approval_rate'}, inplace=True)

print("Loan Approval Rates for Women by Race:")
print(approval_rates_women_by_race)

safe_save_csv(approval_rates_women_by_race, f'{OUT_TAB_DIR}/approvalrates_women_x_race.csv')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.barplot(x='derived_race', y='approval_rate', hue='derived_race', data=approval_rates_women_by_race, palette='viridis', legend=False)
plt.title('Loan Approval Rates for Women by Race')
plt.xlabel('Applicant Race')
plt.ylabel('Approval Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/loan_approval_rates_women_by_race.png')

plt.show()

"""### 3.b. Loan Approval Rates for Males by Race



"""

df_men = df_filtered[df_filtered['derived_sex'] == 'male'].copy()
df_men['approved'] = (df_men['action_taken'] == 1).astype(int)

print("df_men shape:", df_men.shape)
print("Approved column created and converted to integer type.")

approval_rates_men_by_race = df_men.groupby('derived_race', observed=False)['approved'].mean().reset_index()
approval_rates_men_by_race.rename(columns={'approved': 'approval_rate'}, inplace=True)

print("Loan Approval Rates for Men by Race:")
print(approval_rates_men_by_race)

safe_save_csv(approval_rates_men_by_race, f'{OUT_TAB_DIR}/approvalrates_men_x_race.csv')

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.barplot(x='derived_race', y='approval_rate', hue='derived_race', data=approval_rates_men_by_race, palette='viridis', legend=False)
plt.title('Loan Approval Rates for Men by Race')
plt.xlabel('Applicant Race')
plt.ylabel('Approval Rate')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/loan_approval_rates_men_by_race.png')

plt.show()

"""### 3.c. Loan Approval Rates Gender & Race Comparison"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add a 'gender' column to each DataFrame for differentiation
approval_rates_women_by_race['gender'] = 'Female'
approval_rates_men_by_race['gender'] = 'Male'

# Concatenate the two DataFrames
combined_approval_rates = pd.concat([approval_rates_women_by_race, approval_rates_men_by_race], ignore_index=True)

print("Combined Loan Approval Rates by Race and Gender:")
print(combined_approval_rates)

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

plt.figure(figsize=(14, 8))
sns.barplot(x='gender', y='approval_rate', hue='derived_race', data=combined_approval_rates, palette='viridis')
plt.title('Loan Approval Rates by Race and Gender')
plt.xlabel('Applicant Race')
plt.ylabel('Approval Rate')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Applicant Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/loan_approval_rates_by_race_and_gender.png')

plt.show()

"""## 4) Mean Rate Spreads and Loan Metrics

### 4.a. Rate Distributions
"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(df_filtered['rate_spread'], bins=50, kde=True, color='skyblue')
plt.title('Distribution of Rate Spread (Outliers Excluded)')
plt.xlabel('Rate Spread')
plt.ylabel('Frequency')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 8))
sns.kdeplot(
    data=df_filtered,
    x='rate_spread',
    hue='derived_race', # Color by race
    fill=True,              # Fill the area under the density curve
    alpha=0.3,              # Transparency of the fill
    palette='tab20',        # Use a palette with more distinct colors
    common_norm=False       # Each density curve is normalized independently
)
plt.title('Rate Spread Distribution by Race (Outliers Excluded - KDE)')
plt.xlabel('Rate Spread')
plt.ylabel('Density')
plt.legend(title='Applicant Race', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

"""### 4.b. Calculate Metrics"""

rate_spread_stats_women_by_race = df_women.groupby('derived_race', observed=False).agg(
    mean_rate_spread=('rate_spread', 'mean'),
    std_rate_spread=('rate_spread', 'std'),
    mean_loan_amount=('loan_amount', 'mean'),
    std_loan_amount=('loan_amount', 'std'),
    mean_debt_to_income_ratio=('debt_to_income_ratio', 'mean'),
    std_debt_to_income_ratio=('debt_to_income_ratio', 'std'),
    mean_loan_to_value_ratio=('loan_to_value_ratio', 'mean'),
    std_loan_to_value_ratio=('loan_to_value_ratio', 'std')
).reset_index()

print("Rate Spread and Loan Metrics Statistics for Women by Race:")
print(rate_spread_stats_women_by_race)

safe_save_csv(rate_spread_stats_women_by_race, f'{OUT_TAB_DIR}/rate_spread_stats_women_by_race.csv')

# Merge approval rates and rate spread statistics
women_summary_stats_by_race = pd.merge(
    approval_rates_women_by_race,
    rate_spread_stats_women_by_race,
    on='derived_race',
    how='left'
)

print("Summary Statistics for Women by Race:")
print(women_summary_stats_by_race)

print("Loan Amount Summary Statistics for Women by Race:")
print(women_summary_stats_by_race[['derived_race', 'mean_loan_amount', 'std_loan_amount']])

rate_spread_stats_men_by_race = df_men.groupby('derived_race', observed=False).agg(
    mean_rate_spread=('rate_spread', 'mean'),
    std_rate_spread=('rate_spread', 'std'),
    mean_loan_amount=('loan_amount', 'mean'),
    std_loan_amount=('loan_amount', 'std'),
    mean_debt_to_income_ratio=('debt_to_income_ratio', 'mean'),
    std_debt_to_income_ratio=('debt_to_income_ratio', 'std'),
    mean_loan_to_value_ratio=('loan_to_value_ratio', 'mean'),
    std_loan_to_value_ratio=('loan_to_value_ratio', 'std')
).reset_index()

print("Rate Spread and Loan Metrics Statistics for Men by Race:")
print(rate_spread_stats_men_by_race)

safe_save_csv(rate_spread_stats_men_by_race, f'{OUT_TAB_DIR}/rate_spread_stats_men_by_race.csv')

"""### 4.c. Loan Characteristics of Approved Female Applicants

"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.barplot(x='derived_race', y='mean_rate_spread', hue='derived_race', data=rate_spread_stats_women_by_race, palette='viridis', legend=False)
plt.title('Mean Rate Spread for Women by Race')
plt.xlabel('Applicant Race')
plt.ylabel('Mean Rate Spread')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 8))
sns.violinplot(x='derived_race', y='rate_spread', hue='derived_race', data=df_women, palette='viridis', legend=False)
plt.title('Rate Spread Distribution for Women by Race (Outliers Excluded)')
plt.xlabel('Applicant Race')
plt.ylabel('Rate Spread')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

"""#### Loan Characteristics"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

plt.figure(figsize=(12, 7))
sns.barplot(x='derived_race', y='mean_loan_amount', hue='derived_race', data=women_summary_stats_by_race, palette='viridis', legend=False)
plt.title('Mean Loan Amount for Women by Race')
plt.xlabel('Applicant Race')
plt.ylabel('Mean Loan Amount')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/mean_loan_amount_women_by_race.png')

plt.show()

"""### 4.d. Loan Characteristics of Approved Male Applicants

"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(14, 8))
sns.violinplot(x='derived_race', y='rate_spread', hue='derived_race', data=df_men, palette='viridis', legend=False)
plt.title('Rate Spread Distribution for Men by Race (Outliers Excluded)')
plt.xlabel('Applicant Race')
plt.ylabel('Rate Spread')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.barplot(x='derived_race', y='mean_rate_spread', hue='derived_race', data=rate_spread_stats_men_by_race, palette='viridis', legend=False)
plt.title('Mean Rate Spread for Men by Race')
plt.xlabel('Applicant Race')
plt.ylabel('Mean Rate Spread')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/mean_rate_spread_men_by_race.png')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 7))
sns.barplot(x='derived_race', y='mean_loan_amount', hue='derived_race', data=rate_spread_stats_men_by_race, palette='viridis', legend=False)
plt.title('Mean Loan Amount for Men by Race')
plt.xlabel('Applicant Race')
plt.ylabel('Mean Loan Amount')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/mean_loan_amount_men_by_race.png')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

plt.figure(figsize=(14, 8))
sns.violinplot(x='derived_race', y='debt_to_income_ratio', hue='derived_race', data=df_men, palette='viridis', legend=False)
plt.title('Debt-to-Income Ratio Distribution for Men by Race (Outliers Excluded)')
plt.xlabel('Applicant Race')
plt.ylabel('Debt-to-Income Ratio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/dti_distribution_men_by_race.png')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

plt.figure(figsize=(14, 8))
sns.violinplot(x='derived_race', y='rate_spread', hue='derived_race', data=df_men, palette='viridis', legend=False)
plt.title('Rate Spread Distribution for Men by Race (Outliers Excluded)')
plt.xlabel('Applicant Race')
plt.ylabel('Rate Spread')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/rate_spread_distribution_men_by_race.png')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

plt.figure(figsize=(14, 8))
sns.violinplot(x='derived_race', y='loan_amount', hue='derived_race', data=df_men, palette='viridis', legend=False)
plt.title('Loan Amount Distribution for Men by Race (Outliers Excluded)')
plt.xlabel('Applicant Race')
plt.ylabel('Loan Amount')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/loan_amount_distribution_men_by_race.png')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

plt.figure(figsize=(14, 8))
sns.violinplot(x='derived_race', y='loan_to_value_ratio', hue='derived_race', data=df_men, palette='viridis', legend=False)
plt.title('Loan-to-Value Ratio Distribution for Men by Race (Outliers Excluded)')
plt.xlabel('Applicant Race')
plt.ylabel('Loan-to-Value Ratio')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/ltv_distribution_men_by_race.png')

plt.show()

"""## 5) Visualizing Distruibtrions (KDE Plots)

This next visualization is helpful to better understand the background of loan applicants as a way to better understand the rate_spread offered to applicants. The 2D Kernel Density Estimate (KDE) plots provide a richer understanding of the joint distribution of `rate_spread` and `debt_to_income_ratios` across race and gender.

These plots allow a visual comparison of the entire distribution, not just the mean, which is helpful to visualize differences in applicant profiles.

"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

g = sns.FacetGrid(df_filtered, col="derived_sex", hue="derived_race", height=6, aspect=1.2, col_wrap=2, palette="tab20")
g.map(sns.kdeplot, "loan_to_value_ratio", "debt_to_income_ratio", fill=True, levels=5)
g.add_legend(title='Applicant Race', bbox_to_anchor=(1.05, 1), loc='upper left')

for ax in g.axes.flat:
    ax.set_title(f'LTV vs DTI Ratio for {ax.get_title().split("=")[1].strip()}')
    ax.set_xlabel('Loan-to-Value Ratio')
    ax.set_ylabel('Debt-to-Income Ratio')

plt.xlim(0, 125) # Set x-axis limit for loan_to_value_ratio

plt.suptitle('2D KDE of Loan-to-Value vs Debt-to-Income Ratio by Race and Gender', y=1.02, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
safe_save_fig(f'{OUT_FIG_DIR}/kde_ltv_dti_race_gender.png')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

g = sns.FacetGrid(df_filtered, row="derived_race", col="derived_sex", height=4, aspect=1.5, hue="derived_race", palette="tab20", sharex=False, sharey=False)
g.map(sns.kdeplot, "loan_to_value_ratio", "debt_to_income_ratio", fill=True, levels=5) # Removed hue='derived_race'
g.add_legend(title='Applicant Race', bbox_to_anchor=(1.05, 1), loc='upper left')

# Set titles for each facet correctly using row_name and col_name
g.set_titles(row_template='Race: {row_name}', col_template='Gender: {col_name}')

# Set common x and y labels if needed, otherwise FacetGrid handles it
g.set_xlabels('Loan to Value Ratio')
g.set_ylabels('Debt to Income Ratio')

# Set x-axis limit for loan_to_value_ratio on all subplots
for ax in g.axes.flat:
    ax.set_xlim(0, 125) # Fixed LTV x-axis range

plt.suptitle('2D KDE of Loan Value vs Debt-to-Income Ratio by Race and Gender', y=1.02, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
safe_save_fig(f'{OUT_FIG_DIR}/kde_ltv_dti_race_gender_grid.png')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

g = sns.FacetGrid(df_filtered, row="derived_race", col="derived_sex", height=4, aspect=1.5, hue="derived_race", palette="tab20", sharex=False, sharey=False)
g.map(sns.kdeplot, "loan_amount", "loan_to_value_ratio", fill=True, levels=5) # Removed hue='derived_race'
g.add_legend(title='Applicant Race', bbox_to_anchor=(1.05, 1), loc='upper left')

# Set titles for each facet correctly using row_name and col_name
g.set_titles(row_template='Race: {row_name}', col_template='Gender: {col_name}')

# Set common x and y labels if needed, otherwise FacetGrid handles it
g.set_xlabels('Loan Amount')
g.set_ylabels('Loan to Value Ratio')

# Set y-axis limit for loan_to_value_ratio and x-axis for loan_amount on all subplots
for ax in g.axes.flat:
    ax.set_ylim(0, 150) # Fixed LTV y-axis range
    ax.set_xlim(0, 1_500_000) # Fixed Loan Amount x-axis range (e.g., up to 1.5 million)

plt.suptitle('2D KDE of Loan to Value Ratio vs Loan Amount by Race and Gender', y=1.02, fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
safe_save_fig(f'{OUT_FIG_DIR}/kde_rate_spread_l2v_race_gender_grid.png')

plt.show()

"""## 6) Comparison of Mean Rate Spreads & Loan Characteristics"""

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

plt.figure(figsize=(14, 8))
sns.violinplot(x='derived_race', y='loan_to_value_ratio', hue='derived_sex', data=df_filtered, palette='viridis')
plt.title('Loan to Value Ratio Distribution by Race and Gender (Outliers Excluded)')
plt.xlabel('Applicant Race')
plt.ylabel('Loan to Ratio')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Applicant Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/dti_distribution_race_gender_violin.png')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

plt.figure(figsize=(14, 8))
sns.violinplot(x='derived_race', y='loan_amount', hue='derived_sex', data=df_filtered, palette='viridis')
plt.title('Loan Amount Distribution by Race and Gender (Outliers Excluded)')
plt.xlabel('Applicant Race')
plt.ylabel('Loan Amount')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Applicant Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/loanamount_distribution_race_gender_violin.png')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

plt.figure(figsize=(14, 8))
sns.violinplot(x='derived_race', y='rate_spread', hue='derived_sex', data=df_filtered, palette='viridis')
plt.title('Rate Spread Distribution by Race and Gender (Outliers Excluded)')
plt.xlabel('Applicant Race')
plt.ylabel('Rate Spread')
plt.xticks(rotation=45, ha='right')
plt.legend(title='Applicant Gender', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
safe_save_fig(f'{OUT_FIG_DIR}/ratespread_distribution_race_gender_violin.png')

plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Set the seaborn style to 'darkgrid'.
sns.set(style="darkgrid")

# Define the order for derived_sex to suppress the warning and ensure consistent plotting
sex_order = ['male', 'female']

# 2. Create a FacetGrid object named g
# Pass hue='derived_sex' to FacetGrid for consistent coloring across columns and for the legend.
# The palette will then be applied to this hue variable.
g = sns.FacetGrid(df_filtered, row="derived_race", col="derived_sex", hue="derived_sex", height=4, aspect=1.5, palette="viridis", sharex=False, sharey=False, col_order=sex_order)

# 3. Map a seaborn.violinplot to the FacetGrid
# To resolve the ValueError, explicitly provide an 'x' variable.
# Since 'derived_sex' is already used for 'col', passing it as 'x' to violinplot will
# create one violin per facet, grouped by gender. 'inner=None' removes individual points.
# Pass the defined sex_order to the violinplot.
g.map(sns.violinplot, "derived_sex", "debt_to_income_ratio", order=sex_order, inner=None)

# 4. Set the titles for each facet
g.set_titles(row_template='Race: {row_name}', col_template='Gender: {col_name}')

# 5. Set the x-axis label to 'Gender' and the y-axis label to 'Debt-to-Income Ratio'
# Since 'x' is now explicitly 'derived_sex', label it as such.
g.set_xlabels('Gender')
g.set_ylabels('Debt-to-Income Ratio')

# 6. Add a legend
g.add_legend(title='Applicant Gender', bbox_to_anchor=(1.05, 1), loc='upper left')

# 7. Add a main title to the entire plot
plt.suptitle('Debt-to-Income Ratio Distribution by Race and Gender', y=1.02, fontsize=16)

# 8. Adjust the plot layout
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust rect to make space for suptitle and legend

# 9. Save the figure
safe_save_fig(f'{OUT_FIG_DIR}/dti_distribution_race_gender_facet_violin.png')

# 10. Display the plot
plt.show()