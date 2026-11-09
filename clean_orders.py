import os
import numpy as np
import pandas as pd

# 1. Load raw data safely
orders_df = pd.read_csv("data/raw/orders.csv", keep_default_na=False)
products_df = pd.read_csv("data/raw/products.csv")

# 2. Convert order_ts to datetime
orders_df["order_ts"] = pd.to_datetime(orders_df["order_ts"], errors="coerce")

# 3. Remove exact duplicate rows
orders_df = orders_df.drop_duplicates()

# 4. Keep only the latest record for each order_id
orders_df = orders_df.sort_values("order_ts").drop_duplicates(
    subset=["order_id"], keep="last"
)

# 5. Convert order_amount to numeric and fill missing values using product price formula
price_map = products_df.set_index("product_id")["price_inr"].to_dict()
orders_df["order_amount"] = pd.to_numeric(
    orders_df["order_amount"], errors="coerce"
)


def fix_order_amount(row):
  if pd.isna(row["order_amount"]):
    price = price_map.get(row["product_id"], 0)
    qty = row["quantity"]
    disc = row["discount_pct"]
    return round(price * qty * (1 - disc / 100), 2)
  return row["order_amount"]


orders_df["order_amount"] = orders_df.apply(fix_order_amount, axis=1)

# 6. Clean and impute delivery_minutes safely
orders_df = orders_df.merge(
    products_df[["product_id", "category"]], on="product_id", how="left"
)

# Convert delivery_minutes completely to numeric
orders_df["delivery_minutes"] = pd.to_numeric(
    orders_df["delivery_minutes"], errors="coerce"
)

# Set negative delivery times to NaN
orders_df.loc[orders_df["delivery_minutes"] < 0, "delivery_minutes"] = np.nan


# Impute missing delivery_minutes for 'completed' orders using category median
def fill_median_delivery(group):
  median_val = group.loc[group["status"] == "completed", "delivery_minutes"].median()
  mask = (group["status"] == "completed") & (group["delivery_minutes"].isna())
  group.loc[mask, "delivery_minutes"] = median_val
  return group


orders_df = orders_df.groupby("category", group_keys=False).apply(
    fill_median_delivery
)

# Drop temporary category column safely using errors='ignore'
orders_df = orders_df.drop(columns=["category"], errors="ignore")

# 7. Save clean orders to data/clean/orders.csv
os.makedirs("data/clean", exist_ok=True)
orders_df.to_csv("data/clean/orders.csv", index=False)

# Verification prints
print("--- Orders Cleaned Successfully ---")
print(f"Total rows: {len(orders_df)}")
print(f"Completed orders: {(orders_df['status'] == 'completed').sum()}")
print(f"Cancelled orders: {(orders_df['status'] == 'cancelled').sum()}")
print(
    "Sum of completed order_amount: INR"
    f" {orders_df.loc[orders_df['status'] == 'completed', 'order_amount'].sum():,.2f}"
)