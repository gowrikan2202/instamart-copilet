import pandas as pd
import os

def audit_raw_data():
    raw_dir = os.path.join("data", "raw")
    
    # Paths to raw files
    customers_path = os.path.join(raw_dir, "customers.csv")
    products_path = os.path.join(raw_dir, "products.csv")
    orders_path = os.path.join(raw_dir, "orders.csv")
    tickets_path = os.path.join(raw_dir, "tickets.csv")
    
    # Load datasets (using keep_default_na=False for tickets as instructed)
    cust_df = pd.read_csv(customers_path)
    prod_df = pd.read_csv(products_path)
    ord_df = pd.read_csv(orders_path)
    ticket_df = pd.read_csv(tickets_path, keep_default_na=False)
    
    print("=" * 60)
    print("RAW DATA AUDIT REPORT")
    print("=" * 60)
    
    # 1. Shapes and Data Types
    print("\n--- SHAPES ---")
    print(f"Customers shape : {cust_df.shape}")
    print(f"Products shape  : {prod_df.shape}")
    print(f"Orders shape    : {ord_df.shape}")
    print(f"Tickets shape   : {ticket_df.shape}")
    
    # 2. Missing Values Count and Percentage
    print("\n--- MISSING VALUES (Count & Percentage) ---")
    for name, df in [("Customers", cust_df), ("Products", prod_df), ("Orders", ord_df), ("Tickets", ticket_df)]:
        print(f"\n[{name}]")
        missing_count = df.isnull().sum()
        missing_pct = (df.isnull().mean() * 100).round(2)
        missing_summary = pd.DataFrame({"Missing Count": missing_count, "Missing %": missing_pct})
        print(missing_summary[missing_summary["Missing Count"] > 0])
        if missing_summary["Missing Count"].sum() == 0:
            print("No missing values found.")

    # 3. Specific Audit Figures for Orders
    print("\n--- ORDERS SPECIFIC AUDIT ---")
    total_orders = len(ord_df)
    exact_duplicates = ord_df.duplicated().sum()
    pending_rows = (ord_df["status"] == "pending").sum()
    distinct_orders = ord_df["order_id"].nunique()
    
    # Non-numeric amounts check
    def is_not_numeric(val):
        try:
            float(val)
            return False
        except (ValueError, TypeError):
            return True
            
    non_numeric_amounts = ord_df["order_amount"].apply(is_not_numeric).sum()
    
    # Negative delivery times
    # Convert delivery_minutes to numeric coercing errors/blanks to NaN
    deliv_numeric = pd.to_numeric(ord_df["delivery_minutes"], errors="coerce")
    negative_delivery = (deliv_numeric < 0).sum()
    blank_delivery = (ord_df["delivery_minutes"] == "").sum() + ord_df["delivery_minutes"].isnull().sum()
    
    print(f"Raw orders.csv rows                  : {total_orders}")
    print(f"Number of exact duplicate rows       : {exact_duplicates}")
    print(f"Number of pending status rows        : {pending_rows}")
    print(f"Number of distinct order_id values   : {distinct_orders}")
    print(f"Unparseable order_amount entries     : {non_numeric_amounts}")
    print(f"Negative delivery_minutes entries    : {negative_delivery}")
    print(f"Blank delivery_minutes entries       : {blank_delivery}")

    # 4. Specific Audit Figures for Customers
    print("\n--- CUSTOMERS SPECIFIC AUDIT ---")
    blank_age = (cust_df["age"] == "").sum() + cust_df["age"].isnull().sum()
    
    # Ages outside 10-100 (coerce to numeric first)
    age_numeric = pd.to_numeric(cust_df["age"], errors="coerce")
    age_out_of_bounds = ((age_numeric < 10) | (age_numeric > 100)).sum()
    
    distinct_cities = cust_df["city"].nunique()
    
    # Non-blank raw genders
    non_blank_genders = cust_df[cust_df["gender"].notnull() & (cust_df["gender"] != "")]
    distinct_genders = non_blank_genders["gender"].nunique()
    blank_gender = (cust_df["gender"] == "").sum() + cust_df["gender"].isnull().sum()
    
    print(f"Number of blank age entries          : {blank_age}")
    print(f"Number of ages outside 10–100        : {age_out_of_bounds}")
    print(f"Number of distinct raw city spellings: {distinct_cities}")
    print(f"Number of distinct non-blank genders : {distinct_genders}")
    print(f"Number of blank gender entries       : {blank_gender}")
    print("=" * 60)

if __name__ == "__main__":
    audit_raw_data()