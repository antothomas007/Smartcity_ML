# quick_lr_fix.py
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

print("Loading processed 311 file...")
df = pd.read_csv("data/processed/311_proc.csv", parse_dates=["created_dt"])

# FEATURES
features = [
    "complaint_type", "agency", "borough", "zip",
    "created_hour", "created_dow", "created_month",
    "hour_sin", "hour_cos", "lat", "lon"
]
features = [f for f in features if f in df.columns]
X = df[features].copy()
y = df["resolved_48h"].astype(int)

# ---- HANDLE MISSING NUMERIC ----
X["lat"].fillna(0, inplace=True)
X["lon"].fillna(0, inplace=True)

# ---- FREQUENCY ENCODE CATEGORICAL ----
cat_cols = ["complaint_type", "agency", "borough", "zip"]
for col in cat_cols:
    if col in X.columns:
        freq_map = X[col].astype(str).value_counts(normalize=True)
        X[col] = X[col].astype(str).map(freq_map)

# ---- Train/test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---- Pipeline for numeric scaling/imputing ----
num_cols = X_train.columns
pipe = Pipeline([
    ("impute", SimpleImputer(strategy="mean")),
    ("scale", StandardScaler())
])

X_train_p = pipe.fit_transform(X_train)
X_test_p = pipe.transform(X_test)

# ---- LOGISTIC REGRESSION ----
lr = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=-1)
lr.fit(X_train_p, y_train)

# ---- Predictions ----
p = lr.predict_proba(X_test_p)[:, 1]
pred = (p >= 0.5).astype(int)

print("\n=== LOGISTIC REGRESSION (CLEAN BASELINE) ===")
print("AUC:", roc_auc_score(y_test, p))
print("Accuracy:", accuracy_score(y_test, pred))
print("Precision:", precision_score(y_test, pred))
print("Recall:", recall_score(y_test, pred))
print("F1:", f1_score(y_test, pred))
