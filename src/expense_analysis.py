
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Locate project folder
project_folder = Path(__file__).resolve().parents[1]

# Load cleaned dataset
file_path = project_folder / "dataset" / "cleaned_expenses.csv"

df = pd.read_csv(file_path)
df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------
# 1. BASIC EXPENSE ANALYSIS
# --------------------------------

total_expense = df["Amount"].sum()
average_expense = df["Amount"].mean()
highest_expense = df["Amount"].max()

# Category-wise expenses
category_expense = df.groupby("Category")["Amount"].sum()

# Monthly expenses
df["Month"] = df["Date"].dt.to_period("M").astype(str)
monthly_expense = df.groupby("Month")["Amount"].sum()

# --------------------------------
# 2. DISPLAY RESULTS
# --------------------------------

print("\nPERSONAL EXPENSE ANALYSIS")
print("-------------------------")

print(f"Total Expenses: ₹{total_expense:,.2f}")
print(f"Average Transaction: ₹{average_expense:,.2f}")
print(f"Highest Transaction: ₹{highest_expense:,.2f}")

print("\nCategory-wise Expenses:")
print(category_expense.sort_values(ascending=False))

print("\nMonthly Expenses:")
print(monthly_expense)

# --------------------------------
# 3. VISUALIZATION
# --------------------------------

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Graph 1: Expense Distribution
axes[0].pie(
    category_expense.values,
    labels=category_expense.index,
    autopct="%1.1f%%",
    startangle=90
)

axes[0].set_title("Expense Distribution")

# Graph 2: Monthly Expenses
axes[1].bar(
    monthly_expense.index,
    monthly_expense.values
)

axes[1].set_title("Monthly Expenses")
axes[1].set_xlabel("Month")
axes[1].set_ylabel("Amount (₹)")
axes[1].tick_params(axis="x", rotation=90)

# Graph 3: Category Comparison
axes[2].bar(
    category_expense.index,
    category_expense.values
)

axes[2].set_title("Category-wise Expenses")
axes[2].set_xlabel("Category")
axes[2].set_ylabel("Amount (₹)")
axes[2].tick_params(axis="x", rotation=90)

plt.tight_layout()

# Display graphs
plt.show()

print("\nExpense analysis completed!")
