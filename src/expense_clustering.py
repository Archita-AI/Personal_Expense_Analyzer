
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score

# --------------------------------
# 1. LOAD CLEANED DATASET
# --------------------------------

project_folder = Path(__file__).resolve().parents[1]

file_path = project_folder / "dataset" / "cleaned_expenses.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")
print("Total records:", len(df))

# --------------------------------
# 2. SELECT FEATURES
# --------------------------------

features = df[["Amount", "Category"]].copy()

# --------------------------------
# 3. PREPROCESS DATA
# --------------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("amount", StandardScaler(), ["Amount"]),
        ("category", OneHotEncoder(handle_unknown="ignore"), ["Category"])
    ]
)

# --------------------------------
# 4. APPLY K-MEANS CLUSTERING
# --------------------------------

model = Pipeline([
    ("preprocessing", preprocessor),
    ("kmeans", KMeans(
        n_clusters=3,
        random_state=42,
        n_init=10
    ))
])

df["Cluster"] = model.fit_predict(features)

# --------------------------------
# 5. IDENTIFY SPENDING GROUPS
# --------------------------------

cluster_average = df.groupby("Cluster")["Amount"].mean()

sorted_clusters = cluster_average.sort_values().index.tolist()

cluster_names = {
    sorted_clusters[0]: "Low Spending",
    sorted_clusters[1]: "Medium Spending",
    sorted_clusters[2]: "High Spending"
}

df["Spending_Pattern"] = df["Cluster"].map(cluster_names)

# --------------------------------
# 6. DISPLAY RESULTS
# --------------------------------

print("\nK-MEANS CLUSTERING RESULTS")
print("--------------------------")

print("\nNumber of transactions in each group:")

print(df["Spending_Pattern"].value_counts())

print("\nAverage amount in each group:")

print(
    df.groupby("Spending_Pattern")["Amount"]
    .mean()
    .sort_values()
)

print("\nCategory distribution in each group:")

print(pd.crosstab(
    df["Spending_Pattern"],
    df["Category"]
))

# Silhouette score
transformed_data = model.named_steps["preprocessing"].transform(features)

score = silhouette_score(
    transformed_data,
    df["Cluster"]
)

print("\nSilhouette Score:", round(score, 3))

# --------------------------------
# 7. SAVE CLUSTERED DATASET
# --------------------------------

output_file = project_folder / "dataset" / "clustered_expenses.csv"

df.to_csv(output_file, index=False)

print("\nClustered dataset saved successfully!")

# --------------------------------
# 8. VISUALIZATION
# --------------------------------

sns.set_theme(style="whitegrid")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Graph 1: Spending pattern distribution

sns.countplot(
    data=df,
    x="Spending_Pattern",
    order=["Low Spending", "Medium Spending", "High Spending"],
    ax=axes[0]
)

axes[0].set_title("Transactions by Spending Pattern")
axes[0].set_xlabel("Spending Pattern")
axes[0].set_ylabel("Number of Transactions")

# Graph 2: Amount distribution

sns.boxplot(
    data=df,
    x="Spending_Pattern",
    y="Amount",
    order=["Low Spending", "Medium Spending", "High Spending"],
    ax=axes[1]
)

axes[1].set_title("Expense Amount by Spending Pattern")
axes[1].set_xlabel("Spending Pattern")
axes[1].set_ylabel("Amount (₹)")

plt.tight_layout()
plt.show()

print("\nK-Means clustering completed successfully!")
