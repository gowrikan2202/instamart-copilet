import pandas as pd
import numpy as np
import os

# 1. Load raw data (customers.csv)
raw_path = os.path.join("data", "raw", "customers.csv")
df_cust = pd.read_csv(raw_path)

# 2. (a) Standardise city: remove extra spaces and apply title case
df_cust['city'] = df_cust['city'].str.strip().str.title()

# 2. (b) Standardise gender: map variants to male, female, or unknown (blank -> unknown)
df_cust['gender'] = df_cust['gender'].astype(str).str.strip().str.lower()
gender_mapping = {
    'm': 'male',
    'male': 'male',
    'f': 'female',
    'female': 'female'
}
df_cust['gender'] = df_cust['gender'].map(gender_mapping).fillna('unknown')
df_cust.loc[df_cust['gender'] == '', 'gender'] = 'unknown'

# 2. (c) Age cleaning: treat ages outside 10-100 as missing, then fill with city median
df_cust['age'] = pd.to_numeric(df_cust['age'], errors='coerce')
df_cust.loc[(df_cust['age'] < 10) | (df_cust['age'] > 100), 'age'] = np.nan

# Fill missing ages using the median age of the respective customer's city (groupby + transform)
df_cust['age'] = df_cust['age'].fillna(
    df_cust.groupby('city')['age'].transform('median')
)
df_cust['age'] = df_cust['age'].round().astype(int)

# 2. (d) Convert signup_date to datetime
df_cust['signup_date'] = pd.to_datetime(df_cust['signup_date'])

# Optional: Save cleaned customers to data/clean/ folder
clean_dir = os.path.join("data", "clean")
os.makedirs(clean_dir, exist_ok=True)
df_cust.to_csv(os.path.join(clean_dir, "customers.csv"), index=False)

print("Customers cleaned successfully!")

import os
import pandas as pd

# 1. 'data/clean' folder irukku-nu ensure panrathukku (illana auto-va create aagum)
os.makedirs("data/clean", exist_ok=True)

# 2. Cleaned customer DataFrame-ai CSV-aaga save panrathu
df_cust.to_csv("data/clean/customers.csv", index=False)

print("Customer cleaned CSV file successfully save aayiduchu!")

import pandas as pd

# Cleaned data-a read panrom check panrathuku
df_clean = pd.read_csv("data/clean/customers.csv")

# Reference check values-a print panrom
print("--- CLEANED CUSTOMERS CHECKPOINTS ---")
print("City Distribution:")
print(df_clean["city"].value_counts())

print("\nGender Distribution:")
print(df_clean["gender"].value_counts())

print(f"\nMissing Ages remaining: {df_clean['age'].isnull().sum()}")
print(f"Age range: Min = {df_clean['age'].min()}, Max = {df_clean['age'].max()}")
print("-------------------------------------")