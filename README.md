# Customer Segmentation Analysis

## Project Overview
This project analyzes mall customer data and groups customers into four segments using their annual income and spending score. It also explores customer demographics and creates visualizations to help understand customer behavior.

## Objectives
- Load and inspect the mall customers dataset.
- Check data types, missing values, duplicate rows, and summary statistics.
- Explore age, annual income, spending score, and gender.
- Classify customers into four rule-based segments.
- Compare segment profiles and spending patterns across age groups.
- Generate charts and export the segmented dataset to a CSV file.

## Technologies Used
- Python
- Pandas
- Matplotlib

## Dataset
The script attempts to load `Mall_Customers.csv` from one of two online sources. If both attempts fail, it looks for a local file named `Mall_Customers.csv` in the same directory.

Expected columns include:
- `CustomerID`
- `Gender` (some dataset versions use `Genre`)
- `Age`
- `Annual Income (k$)` (renamed to `Income`)
- `Spending Score (1-100)` (renamed to `Spending`)

If the dataset cannot be downloaded, obtain `Mall_Customers.csv` from the [Customer Segmentation Tutorial dataset on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python) and place it beside the Python script.

## Customer Segments
The project uses an annual-income cutoff of **60 (thousand dollars)** and a spending-score cutoff of **50**.

| Segment | Rule |
|---|---|
| Premium | Income ≥ 60 and spending score ≥ 50 |
| Careful | Income ≥ 60 and spending score < 50 |
| Impulsive | Income < 60 and spending score ≥ 50 |
| Budget | Income < 60 and spending score < 50 |

These are simple rule-based groups, not clusters produced by a machine-learning algorithm.

The script also assigns age groups using these bins: `18-25`, `26-35`, `36-50`, and `50+`.

## Installation
1. Install Python 3.
2. Install the required libraries:

   ```bash
   pip install pandas matplotlib
   ```

3. Keep `customer_segmentation.py` and, if needed, `Mall_Customers.csv` in the same folder.

## How to Run
From the project directory, run:

```bash
python customer_segmentation.py
```

The script prints dataset information, exploratory analysis, segment counts, segment profiles, gender shares, and average spending scores by age group. It also displays plots. A working internet connection is needed to download the dataset unless the CSV is already available locally.

## Output Files
After a successful run, the script saves these files in the current working directory:

- `1_distributions.png` — distributions of age, income, and spending score.
- `2_income_vs_spending.png` — annual income versus spending score.
- `3_segment_sizes.png` — number of customers in each segment.
- `4_customer_segments.png` — scatter plot showing the four segments.
- `customers_with_segments.csv` — dataset with the added `Segment` and `AgeGroup` columns.

## Project Workflow
1. Load the dataset.
2. Standardize selected column names.
3. Inspect and summarize the data.
4. Perform exploratory data analysis.
5. Assign rule-based customer segments.
6. Summarize and visualize segment characteristics.
7. Export the results to CSV.

## Notes and Limitations
- The script reports missing values and duplicate rows but does not remove or impute them.
- Segment labels depend on the selected income and spending cutoffs.
- The script uses fixed rules rather than a clustering algorithm such as K-Means.
- The age-group bins start at age 17 and use the labels listed above; check boundary behavior if you change the dataset or bins.
