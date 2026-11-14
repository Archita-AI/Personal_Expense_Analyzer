
import pandas as pd
from pathlib import Path

# Locate project folders
project_folder = Path(__file__).resolve().parents[1]

input_file = project_folder / "dataset" / "expenses.csv"
output_file = project_folder / "dataset" / "cleaned_expenses.csv"

# Load dataset
df = pd.read_csv(input_file)

print("Original Dataset:")
print(df.head())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Remove duplicate records
df = df.drop_duplicates()

# Convert Date column into datetime format
df["Date"] = pd.to_datetime(df["Date"])

# Handle missing values
df = df.dropna(subset=["Date", "Amount", "Category"])

# Ensure amounts are numeric
df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")
df = df.dropna(subset=["Amount"])

# Remove invalid or non-positive expenses
df = df[df["Amount"] > 0]

# Extract additional date features
df["Month"] = df["Date"].dt.month
df["Month_Name"] = df["Date"].dt.month_name()
df["Day"] = df["Date"].dt.day
df["Day_of_Week"] = df["Date"].dt.day_name()

# Save cleaned dataset
df.to_csv(output_file, index=False)

print("\nPreprocessing completed!")
print("Final dataset shape:", df.shape)
print("\nCleaned Dataset:")
print(df.head())

print("\nCleaned dataset saved successfully!")
