import pandas as pd

# 1. Load clean orders data
orders = pd.read_csv("data/clean/orders.csv")

# 2. Convert order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# 3. Filter completed orders
completed_orders = orders[orders["status"] == "completed"].copy()

# 4. Define snapshot date
snapshot_date = pd.to_datetime("2025-10-01 00:00:00")

# 5. Build RFM Table using standard agg method (No error varathu)
rfm_table = completed_orders.groupby("customer_id").agg(
    recency=('order_ts', lambda x: (snapshot_date - x.max()).days),
    frequency=('order_id', 'nunique'),
    monetary=('order_amount', 'sum')
).reset_index()

# Check result
print(rfm_table.head())
print(f"Total customers: {len(rfm_table)}")
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd

# 1. Select RFM features for clustering
rfm_features = rfm_table[['recency', 'frequency', 'monetary']]

# 2. Scale the features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

# 3. Loop through K from 2 to 8 to record inertia and silhouette scores
results = []
for k in range(2, 9):
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = kmeans.fit_predict(rfm_scaled)
    
    inertia = kmeans.inertia_
    sil_score = silhouette_score(rfm_scaled, labels)
    
    results.append({'K': k, 'Inertia': inertia, 'Silhouette_Score': sil_score})

# Convert results to DataFrame and view
results_df = pd.DataFrame(results)
print(results_df)
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import os

# 1. Choose the best K value (e.g., K=4) and apply K-Means clustering
best_k = 4
kmeans_final = KMeans(n_clusters=best_k, n_init=10, random_state=42)

# Assign cluster labels to the rfm_table as 'segment_id'
rfm_table['segment_id'] = kmeans_final.fit_predict(rfm_scaled)

# 2. Compute aggregate statistics (size, mean recency, frequency, monetary) for each segment
segment_profile = rfm_table.groupby('segment_id').agg(
    size=('customer_id', 'count'),
    mean_recency=('recency', 'mean'),
    mean_frequency=('frequency', 'mean'),
    mean_monetary=('monetary', 'mean')
).reset_index()

print("Segment Profiles:")
print(segment_profile)

# 3. Identify the Premium segment (the cluster with the highest mean monetary value)
premium_cluster_id = segment_profile.loc[segment_profile['mean_monetary'].idxmax(), 'segment_id']

# 4. Define a helper function to assign descriptive names and the 'is_premium' flag
def assign_names_and_premium(row):
    sid = row['segment_id']
    if sid == premium_cluster_id:
        return pd.Series(['Premium', True])
    elif row['mean_frequency'] > segment_profile['mean_frequency'].median():
        return pd.Series(['Frequent Buyers', False])
    elif row['mean_recency'] > segment_profile['mean_recency'].median():
        return pd.Series(['At-Risk / Lapsing', False])
    else:
        return pd.Series(['Standard / Occasional', False])

# Merge segment profiles back to map names accurately
merged_df = rfm_table.merge(segment_profile, on='segment_id')

# Apply names and premium flags to the rfm_table
names_flags = merged_df.apply(assign_names_and_premium, axis=1)
rfm_table['segment_name'] = names_flags[0]
rfm_table['is_premium'] = names_flags[1]

# 5. Apply PCA (n_components=2) to reduce dimensions for visualization/analysis
pca = PCA(n_components=2, random_state=42)
pca_coords = pca.fit_transform(rfm_scaled)
print("\nExplained Variance Ratio:", pca.explained_variance_ratio_)

# 6. Create the 'outputs' directory if it doesn't exist and save the final customer segments CSV file
os.makedirs('outputs', exist_ok=True)
rfm_table.to_csv('outputs/customer_segments.csv', index=False)
print("\nSuccessfully saved outputs/customer_segments.csv!")
import pandas as pd
customers = pd.read_csv("data/clean/customers.csv")
print(customers.columns)
import pandas as pd

# 1. Load clean orders and customers data
orders = pd.read_csv("data/clean/orders.csv")
customers = pd.read_csv("data/clean/customers.csv")

# Convert timestamp columns to datetime format (Using 'signup_date' instead of signup_ts)
orders["order_ts"] = pd.to_datetime(orders["order_ts"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# 2. Define cutoff and observation dates for the lapse problem
cutoff_date = pd.to_datetime("2025-09-01 00:00:00")
end_observation_date = pd.to_datetime("2025-10-01 00:00:00")

# 3. Filter orders before cutoff date for feature extraction
historical_orders = orders[orders["order_ts"] < cutoff_date]
active_customers = historical_orders["customer_id"].unique()

# 4. Create target variable 'lapsed': 
# 1 if no completed orders between Sep 1 and Sep 30, 2025; otherwise 0
recent_completed_orders = orders[
    (orders["order_ts"] >= cutoff_date) & 
    (orders["order_ts"] < end_observation_date) & 
    (orders["status"] == "completed")
]
customers_with_recent_orders = recent_completed_orders["customer_id"].unique()

target_df = pd.DataFrame({"customer_id": active_customers})
target_df["lapsed"] = target_df["customer_id"].apply(lambda x: 0 if x in customers_with_recent_orders else 1)

# 5. Extract historical features (Recency, Frequency, Monetary, etc.)
hist_completed = historical_orders[historical_orders["status"] == "completed"]

features_df = hist_completed.groupby("customer_id").agg(
    recency_at_cutoff=("order_ts", lambda x: (cutoff_date - x.max()).days),
    frequency_pre_cutoff=("order_id", "nunique"),
    monetary_pre_cutoff=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery_minutes=("delivery_minutes", "mean")
).reset_index()

# 6. Calculate customer tenure up to cutoff date using 'signup_date'
customers["tenure_days"] = (cutoff_date - customers["signup_date"]).dt.days

# 7. Merge everything safely without key errors
feature_dataset = target_df.merge(features_df, on="customer_id", how="left")
feature_dataset = feature_dataset.merge(
    customers[["customer_id", "age", "gender", "city", "tenure_days"]], 
    on="customer_id", 
    how="left"
)

# 8. Handle missing values safely
feature_dataset["recency_at_cutoff"] = feature_dataset["recency_at_cutoff"].fillna(180)
feature_dataset["frequency_pre_cutoff"] = feature_dataset["frequency_pre_cutoff"].fillna(0)
feature_dataset["monetary_pre_cutoff"] = feature_dataset["monetary_pre_cutoff"].fillna(0)
feature_dataset["avg_discount"] = feature_dataset["avg_discount"].fillna(feature_dataset["avg_discount"].mean())
feature_dataset["avg_delivery_minutes"] = feature_dataset["avg_delivery_minutes"].fillna(feature_dataset["avg_delivery_minutes"].mean())

print("Feature Dataset Created Successfully!")
print("Dataset Shape:", feature_dataset.shape)
print(feature_dataset.head())
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

# 1. Separate Features (X) and Target (y)
# Drop customer_id as it is just an identifier
X = feature_dataset.drop(columns=["customer_id", "lapsed"])
y = feature_dataset["lapsed"]

# Check class distribution & baseline accuracy reference
print("Target Distribution:\n", y.value_counts(normalize=True))

# 2. Train-Test Split (80% train, 20% test, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("\nNumeric Features:", numeric_features)
print("Categorical Features:", categorical_features)

# 4. Create Preprocessing Pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# 5. Train Baseline Logistic Regression Model
lr_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)
print("\n--- Logistic Regression Baseline Results ---")
print("Test F1 Score:", f1_score(y_test, lr_preds))
print(classification_report(y_test, lr_preds))

# 6. Train Baseline Random Forest Model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)
print("\n--- Random Forest Baseline Results ---")
print("Test F1 Score:", f1_score(y_test, rf_preds))
print(classification_report(y_test, rf_preds))
import os
import joblib
import pandas as pd
import numpy as np
# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

def generate_retention_list(best_rf_model, X_test, y_test, test_customer_ids, segment_df):
    """
    Safely predicts lapse probabilities, filters for the Premium segment,
    ranks them, and writes the top 10 to outputs/retention_list.csv
    """
    print("--- Generating Retention List (Part B - Task 8c) ---")
    
    # 1. Predict lapse probabilities for the test set
    # predict_proba() returns probabilities for [class_0, class_1 (lapsed)]
    if hasattr(best_rf_model, "predict_proba"):
        lapse_probs = best_rf_model.predict_proba(X_test)[:, 1]
    else:
        lapse_probs = best_rf_model.decision_function(X_test)
        
    # 2. Build a DataFrame for test results
    test_results = pd.DataFrame({
        "customer_id": test_customer_ids,
        "lapse_probability": lapse_probs,
        "actually_lapsed": y_test.values if hasattr(y_test, "values") else y_test
    })
    
    # 3. Merge with segment information (specifically is_premium)
    # segment_df must contain columns: ['customer_id', 'is_premium']
    merged_test = pd.merge(test_results, segment_df[["customer_id", "is_premium"]], on="customer_id", how="left")
    
    # 4. Filter for test-set customers who are in the Premium segment
    if merged_test["is_premium"].dtype == bool:
        premium_test_customers = merged_test[merged_test["is_premium"] == True]
    else:
        premium_test_customers = merged_test[merged_test["is_premium"].isin([True, 1, "True", "true"])]
        
    # 5. Rank by predicted lapse probability (descending) and take the top 10
    retention_list = premium_test_customers.sort_values(by="lapse_probability", ascending=False).head(10)
    
    # Select exact required columns: customer_id, lapse_probability, actually_lapsed
    output_retention = retention_list[["customer_id", "lapse_probability", "actually_lapsed"]]
    
    # 6. Save to outputs/retention_list.csv
    retention_csv_path = os.path.join("outputs", "retention_list.csv")
    output_retention.to_csv(retention_csv_path, index=False)
    
    # Report results
    truly_lapsed_count = output_retention["actually_lapsed"].sum()
    print(f"Retention list written to {retention_csv_path} successfully!")
    print(f"Total customers in list: {len(output_retention)}")
    print(f"Number of customers who truly lapsed in this top list: {truly_lapsed_count}")
    print(output_retention)
    
    return output_retention
import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Ensure outputs directory exists
os.makedirs("outputs", exist_ok=True)

def run_part_b_remaining_tasks(X_train, X_test, y_train, y_test, numeric_features, categorical_features, test_customer_ids, segment_df):
    
    # -------------------------------------------------------------
    # Build Preprocessing Pipeline
    # -------------------------------------------------------------
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # -------------------------------------------------------------
    # Task 6: Train, Tune and Evaluate
    # -------------------------------------------------------------
    print("\n=== Task 6: Tuning Logistic Regression & Random Forest ===")
    
    # 1. Tune Logistic Regression
    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    lr_param_grid = {'classifier__C': [0.01, 0.1, 1, 10, 100]}
    lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=5, scoring='f1')
    lr_grid.fit(X_train, y_train)
    
    best_lr = lr_grid.best_estimator_
    print(f"Logistic Regression Best Params: {lr_grid.best_params_}, Best CV F1: {lr_grid.best_score_:.4f}")

    # 2. Tune Random Forest
    rf_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    
    rf_param_grid = {
        'classifier__max_depth': [3, 5, 8],
        'classifier__min_samples_leaf': [5, 10, 20]
    }
    rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=5, scoring='f1')
    rf_grid.fit(X_train, y_train)
    
    best_rf = rf_grid.best_estimator_
    print(f"Random Forest Best Params: {rf_grid.best_params_}, Best CV F1: {rf_grid.best_score_:.4f}")

    # Evaluate on Test Set
    print("\n--- Tuned Random Forest Test Evaluation ---")
    y_pred_rf = best_rf.predict(X_test)
    print(classification_report(y_test, y_pred_rf, zero_division=0))
    
    # Select Best Model (Random Forest usually outperforms here)
    selected_model = best_rf

    # -------------------------------------------------------------
    # Task 7: Class-Balance Experiment
    # -------------------------------------------------------------
    print("\n=== Task 7: Class-Balance Experiment ===")
    # Combine training features and labels to oversample minority class
    train_df = X_train.copy()
    train_df['lapsed'] = y_train.values if hasattr(y_train, 'values') else y_train
    
    class_0 = train_df[train_df['lapsed'] == 0]
    class_1 = train_df[train_df['lapsed'] == 1]
    
    # Oversample class_1 to match class_0 count
    class_1_oversampled = class_1.sample(n=len(class_0), replace=True, random_state=42)
    balanced_train_df = pd.concat([class_0, class_1_oversampled]).sample(frac=1, random_state=42)
    
    X_train_bal = balanced_train_df.drop(columns=['lapsed'])
    y_train_bal = balanced_train_df['lapsed']
    
    # Fit Logistic Regression on balanced data
    lr_balanced_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(C=lr_grid.best_params_['classifier__C'], max_iter=1000, random_state=42))
    ])
    lr_balanced_pipeline.fit(X_train_bal, y_train_bal)
    
    y_pred_bal = lr_balanced_pipeline.predict(X_test)
    print("Balanced Logistic Regression Test Report:")
    print(classification_report(y_test, y_pred_bal, zero_division=0))

    # -------------------------------------------------------------
    # Task 8: Interpret, Persist and Use
    # -------------------------------------------------------------
    print("\n=== Task 8: Interpret, Persist, and Use ===")
    
    # Extract Feature Names after One-Hot Encoding
    cat_encoder = best_rf.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot']
    encoded_cat_features = list(cat_encoder.get_feature_names_out(categorical_features))
    all_feature_names = list(numeric_features) + encoded_cat_features
    
    # (a) Feature Importances (Random Forest)
    rf_importances = best_rf.named_steps['classifier'].feature_importances_
    top_rf_indices = np.argsort(rf_importances)[::-1][:5]
    print("\nTop 5 Random Forest Feature Importances:")
    for idx in top_rf_indices:
        print(f"  - {all_feature_names[idx]}: {rf_importances[idx]:.4f}")
        
    # Logistic Regression Coefficients
    lr_coefs = best_lr.named_steps['classifier'].coef_[0]
    print("\nLogistic Regression Coefficients (Sample of 3):")
    for idx in range(min(3, len(all_feature_names))):
        print(f"  - {all_feature_names[idx]}: {lr_coefs[idx]:.4f}")

    # (b) Persist and Reload Model
    model_path = os.path.join("outputs", "lapse_model.joblib")
    joblib.dump(selected_model, model_path)
    
    reloaded_model = joblib.load(model_path)
    assert np.array_equal(selected_model.predict(X_test), reloaded_model.predict(X_test))
    print(f"\nModel successfully persisted to {model_path} and reloaded with verified prediction consistency.")

    # (c) Premium Segment Retention List Generation
    lapse_probs = reloaded_model.predict_proba(X_test)[:, 1]
    
    test_results = pd.DataFrame({
        "customer_id": test_customer_ids,
        "lapse_probability": lapse_probs,
        "actually_lapsed": y_test.values if hasattr(y_test, "values") else y_test
    })
    
    # Merge with segment_df to get is_premium flag
    merged_test = pd.merge(test_results, segment_df[["customer_id", "is_premium"]], on="customer_id", how="left")
    
    # Filter for Premium segment customers
    if merged_test["is_premium"].dtype == bool:
        premium_test = merged_test[merged_test["is_premium"] == True]
    else:
        premium_test = merged_test[merged_test["is_premium"].isin([True, 1, "True", "true"])]
        
    # Rank by lapse probability descending and take top 10
    retention_list = premium_test.sort_values(by="lapse_probability", ascending=False).head(10)
    output_retention = retention_list[["customer_id", "lapse_probability", "actually_lapsed"]]
    
    retention_csv_path = os.path.join("outputs", "retention_list.csv")
    output_retention.to_csv(retention_csv_path, index=False)
    
    truly_lapsed_count = output_retention["actually_lapsed"].sum()
    print(f"\nRetention list saved to {retention_csv_path}")
    print(f"Total customers in list: {len(output_retention)}")
    print(f"Truly lapsed among top list: {truly_lapsed_count}")
    print(output_retention)
    
    return output_retention