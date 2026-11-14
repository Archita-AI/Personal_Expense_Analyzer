
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# --------------------------------
# 1. LOAD DATASET
# --------------------------------

project_folder = Path(__file__).resolve().parents[1]

file_path = project_folder / "dataset" / "cleaned_expenses.csv"

df = pd.read_csv(file_path)

df["Date"] = pd.to_datetime(df["Date"])

# --------------------------------
# 2. CALCULATE MONTHLY EXPENSES
# --------------------------------

df["Month"] = df["Date"].dt.to_period("M")

monthly_expenses = (
    df.groupby("Month")["Amount"]
    .sum()
    .reset_index()
)

monthly_expenses["Month_Number"] = range(
    1, len(monthly_expenses) + 1
)

X = monthly_expenses[["Month_Number"]]
y = monthly_expenses["Amount"]

print("\nMONTHLY EXPENSE HISTORY")
print("-----------------------")
print(monthly_expenses)

# --------------------------------
# 3. TRAIN LINEAR REGRESSION
# --------------------------------

model = LinearRegression()

model.fit(X, y)

predicted_values = model.predict(X)

# --------------------------------
# 4. MODEL EVALUATION
# --------------------------------

mae = mean_absolute_error(y, predicted_values)
r2 = r2_score(y, predicted_values)

print("\nMODEL PERFORMANCE")
print("-----------------")
print(f"Training MAE: ₹{mae:,.2f}")
print(f"Training R2 Score: {r2:.3f}")

# --------------------------------
# 5. PREDICT NEXT MONTH
# --------------------------------

next_month_number = len(monthly_expenses) + 1

next_month_prediction = model.predict(
    pd.DataFrame({"Month_Number": [next_month_number]})
)[0]

next_month_prediction = max(0, next_month_prediction)

last_month = monthly_expenses["Month"].iloc[-1]
next_month = last_month + 1

print("\nNEXT MONTH PREDICTION")
print("--------------------")
print("Predicted Month:", next_month)
print(f"Estimated Expenses: ₹{next_month_prediction:,.2f}")

# --------------------------------
# 6. VISUALIZATION
# --------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_expenses["Month"].astype(str),
    y,
    marker="o",
    label="Actual Expenses"
)

plt.plot(
    monthly_expenses["Month"].astype(str),
    predicted_values,
    linestyle="--",
    marker="s",
    label="Model Fit"
)

plt.scatter(
    str(next_month),
    next_month_prediction,
    marker="*",
    s=200,
    label="Next Month Prediction"
)

plt.title("Monthly Expense Prediction")
plt.xlabel("Month")
plt.ylabel("Expenses (₹)")
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()

print("\nExpense prediction completed!")
