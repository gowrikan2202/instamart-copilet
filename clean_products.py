import os
import sqlite3
import pandas as pd

# 1. Read products.csv from raw folder
raw_path = "data/raw/products.csv"
clean_dir = "data/clean"
os.makedirs(clean_dir, exist_ok=True)

products_df = pd.read_csv(raw_path)

# 2. Save cleaned CSV to the clean folder
clean_path = os.path.join(clean_dir, "products.csv")
products_df.to_csv(clean_path, index=False)

# 3. Store the 'products' table in the SQLite database (data/instamart.db)
db_path = "data/instamart.db"
conn = sqlite3.connect(db_path)
products_df.to_sql("products", conn, if_exists="replace", index=False)
conn.close()

print("Products table successfully cleaned and stored in the database!")