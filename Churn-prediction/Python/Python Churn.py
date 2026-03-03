import pandas as pd

# 1. LOAD DATA

df = pd.read_excel(
    r"C:\Users\WELCOME\Desktop\sales-intelligence-dashboard\Churn-prediction\data\raw\Telco_customer_churn.xlsx"
)

print(" Data Loaded")
print("Shape:", df.shape)

# 2. BASIC INFO

print("\n Dataset Info")
df.info()


# 3. CHURN DISTRIBUTION

print("\n Churn Distribution (counts)")
print(df["Churn Value"].value_counts())

print("\n Churn Distribution (%)")
print(df["Churn Value"].value_counts(normalize=True) * 100)


# 4. FIX TOTAL CHARGES

df["Total Charges"] = pd.to_numeric(df["Total Charges"], errors="coerce")

print("\n Total Charges dtype after fix:")
print(df["Total Charges"].dtype)

print("\n Missing Total Charges after conversion:")
print(df["Total Charges"].isna().sum())


# 5. DROP LEAKAGE COLUMNS

drop_cols = [
    "CustomerID", "Count", "Country", "State", "City", "Zip Code",
    "Lat Long", "Latitude", "Longitude", "Churn Label",
    "Churn Score", "CLTV", "Churn Reason"
]

df_model = df.drop(columns=drop_cols)

print("\n Model dataset shape:", df_model.shape)
print(df_model.head())

df_model = df_model.dropna(subset=["Total Charges"])
print("After dropping missing Total Charges:", df_model.shape)

import numpy as np

# 1) Tenure groups
df_model["tenure_group"] = pd.cut(
    df_model["Tenure Months"],
    bins=[-1, 12, 24, 48, 72],
    labels=["0-12", "13-24", "25-48", "49-72"]
)

# 2) Service count (how many add-on services they use)
service_cols = [
    "Online Security", "Online Backup", "Device Protection",
    "Tech Support", "Streaming TV", "Streaming Movies"
]

df_model["service_count"] = df_model[service_cols].apply(
    lambda row: sum(val == "Yes" for val in row), axis=1
)

# 3) Contract risk level
risk_map = {"Month-to-month": 3, "One year": 2, "Two year": 1}
df_model["contract_risk"] = df_model["Contract"].map(risk_map)

# 4) Value ratio: TotalCharges / (MonthlyCharges * tenure)
# (shows billing consistency / customer value pattern)
df_model["charge_ratio"] = df_model["Total Charges"] / (
    df_model["Monthly Charges"] * df_model["Tenure Months"].replace(0, np.nan)
)

# fill any inf/nan from tenure=0 cases
df_model["charge_ratio"] = df_model["charge_ratio"].fillna(0)

print("Feature engineering done!")
print(df_model[["Tenure Months","tenure_group","service_count","contract_risk","charge_ratio"]].head())

print("\nChurn rate by tenure group (%)")
print(df_model.groupby("tenure_group")["Churn Value"].mean() * 100)

print("\nChurn rate by contract (%)")
print(df_model.groupby("Contract")["Churn Value"].mean() * 100)

print("\nChurn rate by service_count (%)")
print(df_model.groupby("service_count")["Churn Value"].mean() * 100)

# 6. PREPARE FEATURES & TARGET


y = df_model["Churn Value"]
X = df_model.drop(columns=["Churn Value"])

print("X shape:", X.shape)
print("y shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)


# 7. ENCODE CATEGORICAL VARIABLES


X_train_encoded = pd.get_dummies(X_train, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, drop_first=True)

# align columns (VERY IMPORTANT)
X_train_encoded, X_test_encoded = X_train_encoded.align(
    X_test_encoded,
    join="left",
    axis=1,
    fill_value=0
)

print("Encoded train shape:", X_train_encoded.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# train model
log_model = LogisticRegression(max_iter=2000, solver="lbfgs")
log_model.fit(X_train_encoded, y_train)

# predictions
y_pred = log_model.predict(X_test_encoded)
y_prob = log_model.predict_proba(X_test_encoded)[:, 1]

print("\n Logistic Regression Results")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_prob))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

from sklearn.ensemble import RandomForestClassifier


# RANDOM FOREST MODEL


rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train_encoded, y_train)

# predictions
y_pred_rf = rf_model.predict(X_test_encoded)
y_prob_rf = rf_model.predict_proba(X_test_encoded)[:, 1]

print("\n Random Forest Results")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("ROC-AUC:", roc_auc_score(y_test, y_prob_rf))
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

import pandas as pd

feat_imp = pd.Series(
    rf_model.feature_importances_,
    index=X_train_encoded.columns
).sort_values(ascending=False)

print("\n Top 15 Feature Importances (Random Forest)")
print(feat_imp.head(15))


# 8. HIGH-RISK CUSTOMER LIST EXPORT

import os

processed_dir = r"C:\Users\WELCOME\Desktop\sales-intelligence-dashboard\Churn-prediction\data\processed"
os.makedirs(processed_dir, exist_ok=True)

# churn probability from the Random Forest you already trained
churn_prob = rf_model.predict_proba(X_test_encoded)[:, 1]

# risk segment based on probability
risk_segment = pd.cut(
    churn_prob,
    bins=[-0.01, 0.33, 0.66, 1.0],
    labels=["Low", "Medium", "High"]
)

# Build output table using the SAME X_test rows
risk_df = X_test.copy()
risk_df["Churn_Prob"] = churn_prob
risk_df["Risk_Segment"] = risk_segment
risk_df["Actual_Churn"] = y_test.values

# Add CustomerID back using the same indices from df_model -> df
risk_df = risk_df.join(df.loc[risk_df.index, ["CustomerID"]])

# nice column order for Power BI
cols_order = [
    "CustomerID", "Churn_Prob", "Risk_Segment", "Actual_Churn",
    "Contract", "Tenure Months", "Monthly Charges", "Total Charges",
    "Internet Service", "Payment Method", "Paperless Billing",
    "Online Security", "Tech Support", "service_count"
]
cols_order = [c for c in cols_order if c in risk_df.columns] + [c for c in risk_df.columns if c not in cols_order]
risk_df = risk_df[cols_order]

out_path = os.path.join(processed_dir, "high_risk_customers.csv")
risk_df.to_csv(out_path, index=False)

print("\n Exported high-risk customer list to:")
print(out_path)
print(risk_df.head())