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