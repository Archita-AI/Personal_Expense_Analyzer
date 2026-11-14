
import pandas as pd
import numpy as np
from pathlib import Path

# Make the dataset reproducible
np.random.seed(42)

# Number of transactions
num_records = 500

# Expense categories
categories = [
    "Food",
    "Transport",
    "Shopping",
    "Bills",
    "Entertainment",
    "Health",
    "Education",
    "Groceries"
]

# Probability of each category
category_weights = [0.22, 0.14, 0.13, 0.12, 0.10, 0.08, 0.09, 0.12]

# Expense amount ranges
amount_ranges = {
    "Food": (50, 800),
    "Transport": (20, 500),
    "Shopping": (200, 5000),
    "Bills": (300, 4000),
    "Entertainment": (100, 2000),
    "Health": (100, 3000),
    "Education": (200, 5000),
    "Groceries": (100, 2500)
}

# Descriptions for each category
descriptions = {
    "Food": ["Breakfast", "Lunch", "Dinner", "Cafe"],
    "Transport": ["Bus", "Auto", "Cab", "Fuel"],
    "Shopping": ["Clothes", "Accessories", "Electronics"],
    "Bills": ["Electricity", "Internet", "Mobile Recharge"],
    "Entertainment": ["Movies", "Games", "Subscriptions"],
    "Health": ["Medicine", "Checkup", "Pharmacy"],
    "Education": ["Books", "Courses", "Stationery"],
    "Groceries": ["Vegetables", "Fruits", "Household Items"]
}

payment_methods = ["UPI", "Cash", "Credit Card", "Debit Card"]

# Generate random dates
dates = pd.date_range(
    start="2026-01-01",
    end="2026-11-13"
)

expense_dates = np.random.choice(
    dates,
    size=num_records
)

# Generate categories
expense_categories = np.random.choice(
    categories,
    size=num_records,
    p=category_weights
)

# Generate amounts according to category
amounts = [
    np.random.randint(
        amount_ranges[category][0],
        amount_ranges[category][1] + 1
    )
    for category in expense_categories
]

# Generate descriptions
expense_descriptions = [
    np.random.choice(descriptions[category])
    for category in expense_categories
]

# Generate payment methods
payments = np.random.choice(
    payment_methods,
    size=num_records,
    p=[0.55, 0.20, 0.15, 0.10]
)

# Create DataFrame
df = pd.DataFrame({
    "Date": expense_dates,
    "Amount": amounts,
    "Category": expense_categories,
    "Payment_Method": payments,
    "Description": expense_descriptions
})

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

# Save dataset
project_folder = Path(__file__).resolve().parents[1]
dataset_folder = project_folder / "dataset"
dataset_folder.mkdir(exist_ok=True)

file_path = dataset_folder / "expenses.csv"
df.to_csv(file_path, index=False)

print("Dataset created successfully!")
print("Total transactions:", len(df))
print("\nFirst 10 records:")
print(df.head(10))

print("\nDataset information:")
print(df.info())

print("\nTotal expense: ₹", df["Amount"].sum())
print("\nCategory-wise expenses:")
print(df.groupby("Category")["Amount"].sum())
