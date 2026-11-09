import os
import sqlite3
import numpy as np
import pandas as pd

# 1. Ensure directories exist
os.makedirs("data/clean", exist_ok=True)
os.makedirs("outputs", exist_ok=True)
os.makedirs("state", exist_ok=True)

print("Loading raw data...")
customers_raw = pd.read_csv("data/raw/customers.csv")
products_raw = pd.read_csv("data/raw/products.csv")
orders_raw = pd.read_csv("data/raw/orders.csv")

# ==========================================
# 2. CLEAN CUSTOMERS
# ==========================================
print("Cleaning customers...")
df_cust = customers_raw.copy()

# (a) Standardise city: strip spaces and apply title case
df_cust["city"] = df_cust["city"].str.strip().str.title()

# (b) Map gender: standardise to male, female, or unknown (blank -> unknown)
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

# (c) Treat ages outside 10-100 as missing, then fill missing ages with city median age
df_cust["age"] = pd.to_numeric(df_cust["age"], errors="coerce")
df_cust.loc[(df_cust["age"] < 10) | (df_cust["age"] > 100), "age"] = np.nan

# Fill missing ages with median age of customer's city
city_medians = df_cust.groupby("city")["age"].transform("median")
df_cust["age"] = df_cust["age"].fillna(city_medians)
df_cust["age"] = df_cust["age"].round().astype(int)

# (d) Convert signup_date to datetime
df_cust["signup_date"] = pd.to_datetime(df_cust["signup_date"])

# ==========================================
# 3. CLEAN ORDERS
# ==========================================
print("Cleaning orders...")
df_ord = orders_raw.copy()

# 1. Convert order_ts to datetime
df_ord["order_ts"] = pd.to_datetime(df_ord["order_ts"])

# 2. Remove exact duplicate rows
df_ord = df_ord.drop_duplicates()

# 3. For each order_id keep only the record with the latest order_ts
df_ord = df_ord.sort_values("order_ts").groupby("order_id", as_index=False).last()

# 4. Convert order_amount to numeric, fill missing using product price formula
# Merge products to get price_inr
df_ord = df_ord.merge(products_raw[["product_id", "price_inr", "category"]], on="product_id", how="left")

df_ord["order_amount_num"] = pd.to_numeric(df_ord["order_amount"], errors="coerce")
missing_amount_mask = df_ord["order_amount_num"].isna()

# Formula: round(price_inr * quantity * (1 - discount_pct / 100), 2)
calculated_amounts = np.round(
    df_ord["price_inr"] * df_ord["quantity"] * (1 - df_ord["discount_pct"] / 100), 2
)
df_ord.loc[missing_amount_mask, "order_amount_num"] = calculated_amounts[missing_amount_mask]
df_ord["order_amount"] = df_ord["order_amount_num"]
df_ord = df_ord.drop(columns=["order_amount_num"])

# 5. Handle delivery_minutes
df_ord["delivery_minutes"] = pd.to_numeric(df_ord["delivery_minutes"], errors="coerce")
# Set negative delivery_minutes to missing
df_ord.loc[df_ord["delivery_minutes"] < 0, "delivery_minutes"] = np.nan

# For completed orders, fill missing delivery_minutes with category median among completed orders
completed_mask = df_ord["status"] == "completed"
cat_medians = df_ord[completed_mask].groupby("category")["delivery_minutes"].transform("median")
df_ord.loc[completed_mask & df_ord["delivery_minutes"].isna(), "delivery_minutes"] = cat_medians[completed_mask & df_ord["delivery_minutes"].isna()]
df_ord["delivery_minutes"] = df_ord["delivery_minutes"].round()

# For cancelled orders, leave delivery_minutes missing (NaN / blank)
df_ord.loc[df_ord["status"] == "cancelled", "delivery_minutes"] = np.nan

# Drop helper columns if needed, keeping exact schema required
df_ord = df_ord[["order_id", "customer_id", "product_id", "quantity", "order_ts", "status", "discount_pct", "order_amount", "delivery_minutes"]]

# ==========================================
# 4. SAVE CLEAN TABLES & STORE IN SQLITE
# ==========================================
print("Saving clean tables to CSV and SQLite...")

# Clean Products table as well
df_prod = products_raw.copy()

# Save to CSV
df_cust.to_csv("data/clean/customers.csv", index=False)
df_prod.to_csv("data/clean/products.csv", index=False)
df_ord.to_csv("data/clean/orders.csv", index=False)

# Store in SQLite database (data/instamart.db)
db_path = "data/instamart.db"
conn = sqlite3.connect(db_path)

# Store order_ts as text format 'YYYY-MM-DD HH:MM:SS'
df_ord_db = df_ord.copy()
df_ord_db["order_ts"] = pd.to_datetime(df_ord_db["order_ts"]).dt.strftime("%Y-%m-%d %H:%M:%S")

df_cust.to_sql("customers", conn, if_exists="replace", index=False)
df_prod.to_sql("products", conn, if_exists="replace", index=False)
df_ord_db.to_sql("orders", conn, if_exists="replace", index=False)
conn.close()

print("Data cleaning and storage completed successfully!")