
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import date

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LinearRegression

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Personal Expense Analyzer",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Personal Expense Analyzer")
st.caption("Track expenses | Analyze spending | Predict future expenses")

# ==========================================
# 2. LOAD DATASET
# ==========================================

PROJECT_FOLDER = Path(__file__).resolve().parent
DATASET_FOLDER = PROJECT_FOLDER / "dataset"
DATASET_FOLDER.mkdir(exist_ok=True)

FILE_PATH = DATASET_FOLDER / "expenses.csv"

CATEGORIES = [
    "Food",
    "Transport",
    "Shopping",
    "Bills",
    "Entertainment",
    "Health",
    "Education",
    "Groceries"
]

PAYMENT_METHODS = [
    "UPI",
    "Cash",
    "Credit Card",
    "Debit Card"
]


def load_data():
    if FILE_PATH.exists():
        data = pd.read_csv(FILE_PATH)
    else:
        data = pd.DataFrame(columns=[
            "Date",
            "Amount",
            "Category",
            "Payment_Method",
            "Description"
        ])

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["Amount"] = pd.to_numeric(data["Amount"], errors="coerce")

    data = data.dropna(
        subset=["Date", "Amount", "Category"]
    )

    data = data[data["Amount"] > 0]

    return data


df = load_data()

# ==========================================
# 3. SIDEBAR - ADD EXPENSE
# ==========================================

st.sidebar.header("➕ Add New Expense")

with st.sidebar.form("expense_form"):

    expense_date = st.date_input(
        "Expense Date",
        value=date.today()
    )

    expense_amount = st.number_input(
        "Amount (₹)",
        min_value=1.0,
        value=100.0,
        step=10.0
    )

    expense_category = st.selectbox(
        "Category",
        CATEGORIES
    )

    payment_method = st.selectbox(
        "Payment Method",
        PAYMENT_METHODS
    )

    description = st.text_input(
        "Description",
        placeholder="Example: Lunch"
    )

    submitted = st.form_submit_button(
        "Add Expense",
        use_container_width=True
    )

    if submitted:

        new_expense = pd.DataFrame([{
            "Date": pd.Timestamp(expense_date),
            "Amount": expense_amount,
            "Category": expense_category,
            "Payment_Method": payment_method,
            "Description": description
        }])

        updated_data = pd.concat(
            [df, new_expense],
            ignore_index=True
        )

        updated_data.to_csv(
            FILE_PATH,
            index=False
        )

        st.success("Expense added successfully!")

        st.rerun()


# ==========================================
# 4. SIDEBAR - BUDGET
# ==========================================

st.sidebar.divider()

st.sidebar.header("🎯 Monthly Budget")

monthly_budget = st.sidebar.number_input(
    "Set your monthly budget (₹)",
    min_value=1,
    value=15000,
    step=500
)

# ==========================================
# 5. EMPTY DATA CHECK
# ==========================================

if df.empty:

    st.warning(
        "No valid transactions found. Add an expense using the sidebar."
    )

    st.stop()

# ==========================================
# 6. PREPARE DATA
# ==========================================

df = df.sort_values("Date").reset_index(drop=True)

df["Month"] = df["Date"].dt.to_period("M").astype(str)

total_expense = df["Amount"].sum()

average_expense = df["Amount"].mean()

highest_expense = df["Amount"].max()

total_transactions = len(df)

current_month = pd.Timestamp.today().strftime("%Y-%m")

current_month_data = df[
    df["Month"] == current_month
]

current_month_expense = current_month_data["Amount"].sum()

budget_remaining = monthly_budget - current_month_expense

# ==========================================
# 7. DASHBOARD NAVIGATION
# ==========================================

overview_tab, transaction_tab, analysis_tab, ml_tab, budget_tab = st.tabs([
    "🏠 Overview",
    "📋 Transactions",
    "📊 Spending Analysis",
    "🤖 Machine Learning",
    "🎯 Budget & Insights"
])

# ==========================================
# TAB 1 - OVERVIEW
# ==========================================

with overview_tab:

    st.subheader("Financial Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Expenses",
        f"₹{total_expense:,.0f}"
    )

    col2.metric(
        "Average Transaction",
        f"₹{average_expense:,.0f}"
    )

    col3.metric(
        "Highest Transaction",
        f"₹{highest_expense:,.0f}"
    )

    col4.metric(
        "Transactions",
        total_transactions
    )

    st.divider()

    left, right = st.columns(2)

    category_totals = (
        df.groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    monthly_totals = (
        df.groupby("Month")["Amount"]
        .sum()
        .sort_index()
    )

    with left:

        st.subheader("Category-wise Expenses")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.pie(
            category_totals.values,
            labels=category_totals.index,
            autopct="%1.1f%%",
            startangle=90
        )

        ax.set_title("Expense Distribution")

        st.pyplot(fig)
        plt.close(fig)

    with right:

        st.subheader("Monthly Spending")

        fig, ax = plt.subplots(figsize=(7, 5))

        ax.bar(
            monthly_totals.index,
            monthly_totals.values
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Amount (₹)")
        ax.set_title("Monthly Expense Trend")

        plt.xticks(rotation=45)
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    st.subheader("Recent Transactions")

    st.dataframe(
        df.sort_values("Date", ascending=False).head(10),
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# TAB 2 - TRANSACTIONS
# ==========================================

with transaction_tab:

    st.subheader("All Expense Transactions")

    category_filter = st.selectbox(
        "Filter by Category",
        ["All"] + sorted(df["Category"].unique().tolist())
    )

    filtered_df = df.copy()

    if category_filter != "All":

        filtered_df = filtered_df[
            filtered_df["Category"] == category_filter
        ]

    st.write(
        f"Showing {len(filtered_df)} transactions"
    )

    st.dataframe(
        filtered_df.sort_values("Date", ascending=False),
        use_container_width=True,
        hide_index=True
    )

    csv_data = filtered_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Transactions CSV",
        data=csv_data,
        file_name="my_expenses.csv",
        mime="text/csv"
    )

# ==========================================
# TAB 3 - SPENDING ANALYSIS
# ==========================================

with analysis_tab:

    st.subheader("Detailed Expense Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.write("### Category Comparison")

        st.bar_chart(category_totals)

    with col2:

        st.write("### Payment Method Analysis")

        payment_totals = df.groupby(
            "Payment_Method"
        )["Amount"].sum()

        st.bar_chart(payment_totals)

    st.write("### Monthly Expense Summary")

    monthly_summary = (
        df.groupby("Month")
        .agg(
            Total_Expenses=("Amount", "sum"),
            Average_Expense=("Amount", "mean"),
            Transaction_Count=("Amount", "count")
        )
        .reset_index()
    )

    st.dataframe(
        monthly_summary,
        use_container_width=True,
        hide_index=True
    )

    st.write("### Category-wise Expense Table")

    st.dataframe(
        category_totals.reset_index(),
        use_container_width=True,
        hide_index=True
    )

# ==========================================
# TAB 4 - MACHINE LEARNING
# ==========================================

with ml_tab:

    st.subheader("Machine Learning Analysis")

    # --------------------------------------
    # K-MEANS CLUSTERING
    # --------------------------------------

    st.write("### 1. K-Means Spending Pattern Analysis")

    if len(df) >= 3 and df["Amount"].nunique() >= 3:

        features = df[["Amount", "Category"]].copy()

        preprocessor = ColumnTransformer(
            transformers=[
                ("amount", StandardScaler(), ["Amount"]),
                (
                    "category",
                    OneHotEncoder(handle_unknown="ignore"),
                    ["Category"]
                )
            ]
        )

        clustering_model = Pipeline([
            ("preprocessing", preprocessor),
            (
                "kmeans",
                KMeans(
                    n_clusters=3,
                    random_state=42,
                    n_init=10
                )
            )
        ])

        clusters = clustering_model.fit_predict(features)

        cluster_data = df.copy()

        cluster_data["Cluster"] = clusters

        cluster_averages = (
            cluster_data.groupby("Cluster")["Amount"].mean()
        )

        ordered_clusters = cluster_averages.sort_values().index.tolist()

        names = {
            ordered_clusters[0]: "Low Spending",
            ordered_clusters[1]: "Medium Spending",
            ordered_clusters[2]: "High Spending"
        }

        cluster_data["Spending_Pattern"] = (
            cluster_data["Cluster"].map(names)
        )

        st.write("Transactions grouped by spending pattern:")

        st.bar_chart(
            cluster_data["Spending_Pattern"].value_counts()
        )

        st.write("Average amount in each group:")

        st.dataframe(
            cluster_data.groupby("Spending_Pattern")["Amount"]
            .mean()
            .reset_index(),
            use_container_width=True,
            hide_index=True
        )

        transformed = clustering_model.named_steps[
            "preprocessing"
        ].transform(features)

        score = silhouette_score(
            transformed,
            clusters
        )

        st.metric(
            "Silhouette Score",
            f"{score:.3f}"
        )

        st.caption(
            "A higher silhouette score generally indicates better-separated clusters. "
            "Interpret the score in the context of this dataset."
        )

        st.download_button(
            "Download Clustered Transactions",
            data=cluster_data.to_csv(index=False).encode("utf-8"),
            file_name="clustered_expenses.csv",
            mime="text/csv"
        )

    else:

        st.info(
            "At least three transactions with distinct amounts are needed for clustering."
        )

    st.divider()

    # --------------------------------------
    # LINEAR REGRESSION
    # --------------------------------------

    st.write("### 2. Future Expense Prediction")

    monthly_data = (
        df.groupby(df["Date"].dt.to_period("M"))["Amount"]
        .sum()
        .sort_index()
        .reset_index()
    )

    monthly_data.columns = ["Month", "Amount"]

    monthly_data["Month_Number"] = np.arange(
        len(monthly_data)
    )

    if len(monthly_data) >= 3:

        X = monthly_data[["Month_Number"]]
        y = monthly_data["Amount"]

        regression_model = LinearRegression()

        regression_model.fit(X, y)

        fitted_values = regression_model.predict(X)

        mae = np.mean(np.abs(y - fitted_values))

        next_month_number = len(monthly_data)

        prediction = regression_model.predict(
            pd.DataFrame({
                "Month_Number": [next_month_number]
            })
        )[0]

        prediction = max(0, prediction)

        next_month = monthly_data["Month"].iloc[-1] + 1

        st.metric(
            f"Estimated Expenses for {next_month}",
            f"₹{prediction:,.2f}"
        )

        st.metric(
            "Training Mean Absolute Error",
            f"₹{mae:,.2f}"
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        month_labels = monthly_data["Month"].astype(str)

        ax.plot(
            month_labels,
            y,
            marker="o",
            label="Actual Expenses"
        )

        ax.plot(
            month_labels,
            fitted_values,
            linestyle="--",
            marker="s",
            label="Model Fit"
        )

        ax.scatter(
            [str(next_month)],
            [prediction],
            marker="*",
            s=200,
            label="Next Month Prediction"
        )

        ax.set_title("Monthly Expense Prediction")
        ax.set_xlabel("Month")
        ax.set_ylabel("Amount (₹)")

        plt.xticks(rotation=45)
        ax.legend()
        plt.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "This is a simple linear trend estimate based on historical monthly totals. "
            "It is not a guarantee of future spending. Training MAE is not a test-set accuracy score."
        )

        st.dataframe(
            monthly_data[["Month", "Amount"]],
            use_container_width=True,
            hide_index=True
        )

    else:

        st.info(
            "At least three different months of data are required for expense prediction."
        )

# ==========================================
# TAB 5 - BUDGET AND SMART INSIGHTS
# ==========================================

with budget_tab:

    st.subheader("Monthly Budget Analysis")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Monthly Budget",
        f"₹{monthly_budget:,.0f}"
    )

    col2.metric(
        "Current Month Spending",
        f"₹{current_month_expense:,.0f}"
    )

    col3.metric(
        "Remaining Budget",
        f"₹{budget_remaining:,.0f}",
        delta=f"₹{budget_remaining:,.0f}"
    )

    budget_progress = current_month_expense / monthly_budget

    st.write("### Budget Usage")

    st.progress(
        min(budget_progress, 1.0)
    )

    st.write(
        f"You have used {budget_progress * 100:.1f}% of your monthly budget."
    )

    if current_month_expense > monthly_budget:

        st.error("You have exceeded your monthly budget!")

    elif budget_progress >= 0.8:

        st.warning("You are approaching your monthly budget limit.")

    else:

        st.success("Your spending is currently within your budget.")

    st.divider()

    st.subheader("💡 Smart Spending Insights")

    highest_category = category_totals.idxmax()

    highest_category_amount = category_totals.max()

    highest_category_percentage = (
        highest_category_amount / total_expense
    ) * 100

    st.info(
        f"Your highest spending category is {highest_category}, "
        f"accounting for {highest_category_percentage:.1f}% "
        f"of your total recorded expenses."
    )

    st.info(
        f"Your average transaction amount is ₹{average_expense:,.2f}."
    )

    highest_payment = (
        df.groupby("Payment_Method")["Amount"].sum().idxmax()
    )

    st.info(
        f"Your most-used spending method by total amount is {highest_payment}."
    )

    if len(monthly_totals) >= 2:

        previous_month_expense = monthly_totals.iloc[-2]

        latest_month_expense = monthly_totals.iloc[-1]

        change = latest_month_expense - previous_month_expense

        if change > 0:

            st.warning(
                f"Your latest recorded month has ₹{change:,.2f} "
                "more spending than the previous recorded month."
            )

        elif change < 0:

            st.success(
                f"Your latest recorded month has ₹{abs(change):,.2f} "
                "less spending than the previous recorded month."
            )

        else:

            st.info("Your spending was unchanged between the two latest months.")

# ==========================================
# 8. FOOTER
# ==========================================

st.divider()

st.caption(
    "Personal Expense Analyzer | Python | Pandas | Scikit-learn | Streamlit"
)

st.caption(
    "This application uses sample expense data unless you add your own transactions."
)
