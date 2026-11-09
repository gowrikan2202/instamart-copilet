import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import joblib

def run_pipeline():
    print("==========================================")
    print("STARTING INSTAMART COPILOT PIPELINE")
    print("==========================================")

    # Ensure all required directories exist
    os.makedirs("data/clean", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("state", exist_ok=True)
    os.makedirs("sql", exist_ok=True)

    # ==========================================
    # PART A: Data Foundation & Cleaning
    # ==========================================
    print("\n[Part A] Loading and cleaning raw data...")
    if not os.path.exists("data/raw/orders.csv"):
        print("Error: Raw data not found! Please run 'python generate_data.py' first.")
        return

    customers_raw = pd.read_csv("data/raw/customers.csv")
    products_raw = pd.read_csv("data/raw/products.csv")
    orders_raw = pd.read_csv("data/raw/orders.csv", keep_default_na=False)

    # 1. Clean Customers
    df_cust = customers_raw.copy()
    df_cust["city"] = df_cust["city"].str.strip().str.title()

    def clean_gender(g):
        if pd.isna(g) or str(g).strip() == "":
            return "unknown"
        g_lower = str(g).strip().lower()
        if g_lower in ["m", "male"]:
            return "male"
        elif g_lower in ["f", "female"]:
            return "female"
        return "unknown"

    df_cust["gender"] = df_cust["gender"].apply(clean_gender)
    df_cust["age"] = pd.to_numeric(df_cust["age"], errors="coerce")
    df_cust.loc[(df_cust["age"] < 10) | (df_cust["age"] > 100), "age"] = np.nan
    city_medians = df_cust.groupby("city")["age"].transform("median")
    df_cust["age"] = df_cust["age"].fillna(city_medians).round().astype(int)
    df_cust["signup_date"] = pd.to_datetime(df_cust["signup_date"])

    # 2. Clean Orders
    df_ord = orders_raw.copy()
    df_ord["order_ts"] = pd.to_datetime(df_ord["order_ts"])
    df_ord = df_ord.drop_duplicates()
    df_ord = df_ord.sort_values("order_ts").groupby("order_id", as_index=False).last()

    df_ord = df_ord.merge(products_raw[["product_id", "price_inr", "category"]], on="product_id", how="left")
    df_ord["order_amount_num"] = pd.to_numeric(df_ord["order_amount"], errors="coerce")
    missing_amount_mask = df_ord["order_amount_num"].isna()
    calc_amounts = np.round(df_ord["price_inr"] * df_ord["quantity"] * (1 - df_ord["discount_pct"] / 100), 2)
    df_ord.loc[missing_amount_mask, "order_amount_num"] = calc_amounts[missing_amount_mask]
    df_ord["order_amount"] = df_ord["order_amount_num"]

    df_ord["delivery_minutes"] = pd.to_numeric(df_ord["delivery_minutes"], errors="coerce")
    df_ord.loc[df_ord["delivery_minutes"] < 0, "delivery_minutes"] = np.nan
    completed_mask = df_ord["status"] == "completed"
    cat_medians = df_ord[completed_mask].groupby("category")["delivery_minutes"].transform("median")
    df_ord.loc[completed_mask & df_ord["delivery_minutes"].isna(), "delivery_minutes"] = cat_medians[completed_mask & df_ord["delivery_minutes"].isna()]
    df_ord["delivery_minutes"] = df_ord["delivery_minutes"].round()
    df_ord.loc[df_ord["status"] == "cancelled", "delivery_minutes"] = np.nan

    df_ord = df_ord[["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes", "category"]]

    # Save Clean Tables & Store in SQLite DB
    df_prod = products_raw.copy()
    df_cust.to_csv("data/clean/customers.csv", index=False)
    df_prod.to_csv("data/clean/products.csv", index=False)
    df_ord.to_csv("data/clean/orders.csv", index=False)

    db_path = "data/instamart.db"
    conn = sqlite3.connect(db_path)
    df_ord_db = df_ord.drop(columns=["category"]).copy()
    df_ord_db["order_ts"] = pd.to_datetime(df_ord_db["order_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

    df_cust.to_sql("customers", conn, if_exists="replace", index=False)
    df_prod.to_sql("products", conn, if_exists="replace", index=False)
    df_ord_db.to_sql("orders", conn, if_exists="replace", index=False)
    conn.close()

    assert len(df_ord) == 12309, f"Orders count mismatch: {len(df_ord)}"
    assert df_ord["status"].value_counts()["completed"] == 11110
    assert df_ord["order_amount"].isnull().sum() == 0
    print("[Part A] ALL CHECKPOINTS PASSED")

    # ==========================================
    # PART B: Customer Segments & Lapse Model
    # ==========================================
    print("\n[Part B] Running RFM clustering and building lapse prediction model...")
    completed_orders = df_ord[df_ord["status"] == "completed"].copy()
    snapshot = pd.to_datetime("2025-10-01 00:00:00")

    # Task 1: RFM Table (Full Window for Segments)
    rfm = completed_orders.groupby("customer_id").agg(
        last_order_ts=("order_ts", "max"),
        frequency=("order_id", "nunique"),
        monetary=("order_amount", "sum")
    ).reset_index()
    rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
    rfm_full = df_cust.merge(rfm, on="customer_id", how="inner")

    # Task 2 & 3: K-Means (K=4) & PCA
    scaler = StandardScaler()
    X_rfm = scaler.fit_transform(rfm_full[["recency", "frequency", "monetary"]])
    kmeans = KMeans(n_clusters=4, n_init=10, random_state=42)
    rfm_full["segment_id"] = kmeans.fit_predict(X_rfm)

    seg_profile = rfm_full.groupby("segment_id")[["recency", "frequency", "monetary"]].mean()
    premium_seg_id = seg_profile["monetary"].idxmax()
    rfm_full["is_premium"] = rfm_full["segment_id"] == premium_seg_id
    rfm_full["segment_name"] = rfm_full["segment_id"].map({i: f"Segment_{i}" for i in range(4)})
    rfm_full.loc[rfm_full["is_premium"], "segment_name"] = "Premium"

    segments_output = rfm_full[["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"]]
    segments_output.to_csv("outputs/customer_segments.csv", index=False)

    # Task 4 & 5: Lapse Model Feature Engineering (Cutoff: 2025-09-01, No Leakage!)
    cutoff = pd.to_datetime("2025-09-01 00:00:00")
    pre_cutoff_orders = df_ord[df_ord["order_ts"] < cutoff]
    pre_completed = pre_cutoff_orders[pre_cutoff_orders["status"] == "completed"]

    cohort_cust_ids = pre_completed["customer_id"].unique()
    cohort_df = df_cust[df_cust["customer_id"].isin(cohort_cust_ids)].copy()

    # Target label: Lapsed = 1 if no completed order in Sept 2025
    sept_completed = completed_orders[
        (completed_orders["order_ts"] >= cutoff) & 
        (completed_orders["order_ts"] <= pd.to_datetime("2025-09-30 23:59:59"))
    ]["customer_id"].unique()
    cohort_df["lapsed"] = cohort_df["customer_id"].apply(lambda cid: 0 if cid in sept_completed else 1)
    cohort_df["tenure"] = (cutoff - cohort_df["signup_date"]).dt.days

    # Compute behavioral features using pre-cutoff data only
    feat_recency = pre_completed.groupby("customer_id")["order_ts"].max().apply(lambda x: (cutoff - x).days).rename("recency")
    feat_freq = pre_completed.groupby("customer_id")["order_id"].nunique().rename("frequency")
    feat_monetary = pre_completed.groupby("customer_id")["order_amount"].sum().rename("monetary")
    feat_disc = pre_completed.groupby("customer_id")["discount_pct"].mean().rename("avg_discount")
    feat_deliv = pre_completed.groupby("customer_id")["delivery_minutes"].mean().rename("avg_delivery")

    # Cancellation rate before cutoff
    all_pre = pre_cutoff_orders.groupby("customer_id")["status"].agg(total="count", cancelled=lambda x: (x == "cancelled").sum())
    canc_rate = (all_pre["cancelled"] / all_pre["total"]).rename("cancellation_rate")

    # Favorite category
    fav_cat = pre_completed.groupby(["customer_id", "category"]).size().reset_index(name="count")
    fav_cat = fav_cat.sort_values(["customer_id", "count"], ascending=[True, False]).drop_duplicates("customer_id").set_index("customer_id")["category"].rename("favourite_category")

    # Merge features
    cohort_df = cohort_df.set_index("customer_id").join([feat_recency, feat_freq, feat_monetary, feat_disc, feat_deliv, canc_rate, fav_cat]).reset_index()
    cohort_df["favourite_category"] = cohort_df["favourite_category"].fillna("Grocery")
    cohort_df["avg_delivery"] = cohort_df["avg_delivery"].fillna(cohort_df["avg_delivery"].median())

    # Define X and y
    numeric_cols = ["recency", "frequency", "monetary", "avg_discount", "avg_delivery", "cancellation_rate", "age", "tenure"]
    nominal_cols = ["city", "gender", "favourite_category"]
    
    X = cohort_df[numeric_cols + nominal_cols]
    y = cohort_df["lapsed"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Preprocessing Pipeline (No Leakage: drop='first' for K-1 columns)
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal_cols)
        ]
    )

    # Task 6: Train Random Forest with Pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    print(f"Test F1 Score (Lapsed Class): {f1_score(y_test, preds, zero_division=0):.4f}")

    joblib.dump(pipeline, "outputs/lapse_model.joblib")
    print("[Part B] Leakage-free pipeline and model saved successfully to outputs/lapse_model.joblib!")

    print("\n==========================================")
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("==========================================")

if __name__ == "__main__":
    run_pipeline()