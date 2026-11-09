import pandas as pd

# Load cleaned data from Part A
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Filter completed orders
completed_orders = orders[orders["status"] == "completed"].copy()
snapshot = pd.to_datetime("2025-10-01 00:00:00")

# Compute RFM table
rfm = completed_orders.groupby("customer_id").agg(
    last_order_ts=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum")
).reset_index()

rfm["recency"] = (snapshot - rfm["last_order_ts"]).dt.days
rfm = rfm.drop(columns=["last_order_ts"])

# Merge with customers table
rfm_full = customers.merge(rfm, on="customer_id", how="inner")
print(f"Total customers with completed orders: {len(rfm_full)}")
