import sqlite3
import pandas as pd

# 1. Connect to the SQLite database
conn = sqlite3.connect("data/instamart.db")

# 2. Read the SQL queries from sql/queries.sql
with open("sql/queries.sql", "r", encoding="utf-8") as f:
    sql_script = f.read()

# 3. Split the script into individual queries separated by ';'
queries = [q.strip() for q in sql_script.split(";") if q.strip()]

# 4. Execute each query and display summary results
for i, query in enumerate(queries, 1):
    print(f"\n--- Running Query Q{i} ---")
    try:
        df = pd.read_sql_query(query, conn)
        print(df.head(3)) # Display top 3 rows of the result
        print(f"Total Rows returned: {len(df)}")
        
        # Specific checkpoint verifications from the project brief
        if i == 1:
            print("Q1 Check -> Expected: 30 customers")
        elif i == 5:
            print("Q5 Check -> Expected: 550 customers, Average spend around INR 2,552.06")
        elif i == 7:
            print("Q7 Check -> Grand Total (Last cumulative row):", df.iloc[-1].values if not df.empty else "No data")
            
    except Exception as e:
        print(f"Error executing Q{i}: {e}")

conn.close()