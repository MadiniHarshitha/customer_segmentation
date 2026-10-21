import os
import sys

import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1. LOAD THE DATA
# ---------------------------------------------------------------

DATA_URLS = [
    # Packt "Data Science for Marketing Analytics" GitHub repo (MIT license)
    "https://raw.githubusercontent.com/PacktPublishing/"
    "Data-Science-for-Marketing-Analytics-Second-Edition/master/"
    "Chapter03/Datasets/Mall_Customers.csv",
    # Hugging Face copy of the same file
    "https://huggingface.co/datasets/Garsa3112/AssignmentDataset/"
    "resolve/main/Mall_Customers.csv",
]

df = None
for url in DATA_URLS:
    try:
        df = pd.read_csv(url)
        print("Data loaded from:", url)
        break
    except Exception as error:
        print("Could not load from", url, "->", type(error).__name__)

if df is None and os.path.exists("Mall_Customers.csv"):
    df = pd.read_csv("Mall_Customers.csv")
    print("Data loaded from local file: Mall_Customers.csv")

if df is None:
    sys.exit("Could not load the dataset. Check your internet connection, or "
             "download Mall_Customers.csv from\n"
             "https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python\n"
             "and place it in the same folder as this script.")

# Some versions of the file call the column "Genre" instead of "Gender"
df = df.rename(columns={"Genre": "Gender",
                        "Annual Income (k$)": "Income",
                        "Spending Score (1-100)": "Spending"})

print("First 5 rows:")
print(df.head())

# ---------------------------------------------------------------
# 2. UNDERSTAND AND CLEAN THE DATA
# ---------------------------------------------------------------
print("\nShape (rows, columns):", df.shape)
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
print("\nSummary statistics:")
print(df.describe())

# ---------------------------------------------------------------
# 3. EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
print("\nGender count:")
print(df["Gender"].value_counts())

print("\nAverage values by gender:")
print(df.groupby("Gender")[["Age", "Income", "Spending"]].mean().round(1))

# Histograms of the main columns
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, col in zip(axes, ["Age", "Income", "Spending"]):
    ax.hist(df[col], bins=15, color="steelblue", edgecolor="white")
    ax.set_title("Distribution of " + col)
    ax.set_xlabel(col)
    ax.set_ylabel("Customers")
plt.tight_layout()
plt.savefig("1_distributions.png")
plt.show()

# Correlation between numeric columns
print("\nCorrelation:")
print(df[["Age", "Income", "Spending"]].corr().round(2))

# Income vs Spending scatter plot
plt.figure(figsize=(6, 5))
plt.scatter(df["Income"], df["Spending"], color="teal")
plt.title("Income vs Spending Score")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.savefig("2_income_vs_spending.png")
plt.show()

# ---------------------------------------------------------------
# 4. SEGMENT CUSTOMERS USING SIMPLE RULES
# ---------------------------------------------------------------
# Split at the middle: income 60k$ and spending score 50
INCOME_CUT = 60
SPEND_CUT = 50


def make_segment(row):
    high_income = row["Income"] >= INCOME_CUT
    high_spend = row["Spending"] >= SPEND_CUT
    if high_income and high_spend:
        return "Premium (high income, high spend)"
    elif high_income and not high_spend:
        return "Careful (high income, low spend)"
    elif not high_income and high_spend:
        return "Impulsive (low income, high spend)"
    else:
        return "Budget (low income, low spend)"


df["Segment"] = df.apply(make_segment, axis=1)

# Age groups (extra segmentation)
df["AgeGroup"] = pd.cut(df["Age"],
                        bins=[17, 25, 35, 50, 100],
                        labels=["18-25", "26-35", "36-50", "50+"])

# ---------------------------------------------------------------
# 5. ANALYSE EACH SEGMENT
# ---------------------------------------------------------------
print("\nCustomers per segment:")
print(df["Segment"].value_counts())

summary = df.groupby("Segment").agg(
    Customers=("CustomerID", "count"),
    Avg_Age=("Age", "mean"),
    Avg_Income=("Income", "mean"),
    Avg_Spending=("Spending", "mean"),
).round(1)
print("\nSegment profile:")
print(summary)

print("\nGender share (%) in each segment:")
print((pd.crosstab(df["Segment"], df["Gender"], normalize="index") * 100).round(1))

print("\nAverage spending score by age group:")
print(df.groupby("AgeGroup", observed=True)["Spending"].mean().round(1))

# ---------------------------------------------------------------
# 6. VISUALISE THE SEGMENTS
# ---------------------------------------------------------------
# Bar chart: size of each segment
df["Segment"].value_counts().plot(kind="bar", color="orange", figsize=(8, 4))
plt.title("Number of Customers in Each Segment")
plt.ylabel("Customers")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("3_segment_sizes.png")
plt.show()

# Scatter plot coloured by segment
colors = {"Premium (high income, high spend)": "green",
          "Careful (high income, low spend)": "blue",
          "Impulsive (low income, high spend)": "red",
          "Budget (low income, low spend)": "gray"}

plt.figure(figsize=(8, 6))
for name, group in df.groupby("Segment"):
    plt.scatter(group["Income"], group["Spending"],
                label=name, color=colors[name])
plt.axvline(INCOME_CUT, color="black", linestyle="--")
plt.axhline(SPEND_CUT, color="black", linestyle="--")
plt.title("Customer Segments")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend(fontsize=8)
plt.savefig("4_customer_segments.png")
plt.show()

# ---------------------------------------------------------------
# 7. SAVE RESULT
# ---------------------------------------------------------------
df.to_csv("customers_with_segments.csv", index=False)
print("\nSaved: customers_with_segments.csv")
