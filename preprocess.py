"""
preprocess.py
─────────────
Reusable data cleaning and feature engineering pipeline
for the Customer Churn Prediction project.

Usage:
    from src.preprocess import load_and_preprocess
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess()
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
    "/master/data/Telco-Customer-Churn.csv"
)


def load_data(path: str = None) -> pd.DataFrame:
    """Load dataset from local path or remote URL."""
    source = path if path else DATASET_URL
    df = pd.read_csv(source)
    print(f"✅ Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix data types and handle missing values."""
    df = df.copy()

    # Fix TotalCharges (stored as string in source)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Drop non-informative ID column
    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    print(f"   Missing values after cleaning : {df.isnull().sum().sum()}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create new predictive features from existing ones."""
    df = df.copy()

    # Average spend per month (signal for price sensitivity)
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

    # Bucket tenure into human-readable groups
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"],
    )

    # Total number of paid services (more services = more locked in)
    df["HasMultipleServices"] = (
        (df["PhoneService"]    == "Yes").astype(int)
        + (df["InternetService"] != "No").astype(int)
        + (df["StreamingTV"]     == "Yes").astype(int)
        + (df["StreamingMovies"] == "Yes").astype(int)
    )

    # Has any protective support service
    df["HasSupport"] = (
        (df["OnlineSecurity"] == "Yes") | (df["TechSupport"] == "Yes")
    ).astype(int)

    print("✅ Feature engineering complete — new columns added:")
    print("   AvgMonthlySpend | TenureGroup | HasMultipleServices | HasSupport")
    return df


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for ML models."""
    df = df.copy()

    # Binary label encode simple Yes/No columns
    binary_cols = ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    le = LabelEncoder()
    for col in binary_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Encode target
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # One-hot encode multi-category columns
    ohe_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod", "TenureGroup",
    ]
    df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    # Catch any remaining object columns
    for col in df.select_dtypes("object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df


def load_and_preprocess(
    path: str = None,
    test_size: float = 0.2,
    apply_smote: bool = True,
    random_state: int = 42,
):
    """
    Full preprocessing pipeline — loads, cleans, engineers,
    encodes, splits, optionally applies SMOTE, and scales.

    Returns
    -------
    X_train_bal, X_test, y_train_bal, y_test, scaler, feature_names
    """
    df = load_data(path)
    df = clean_data(df)
    df = engineer_features(df)
    df = encode_features(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    if apply_smote:
        smote = SMOTE(random_state=random_state)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"✅ SMOTE applied — training samples: {len(X_train):,}")

    scaler = StandardScaler()
    X_train = pd.DataFrame(
        scaler.fit_transform(X_train), columns=X.columns
    )
    X_test = pd.DataFrame(
        scaler.transform(X_test), columns=X.columns
    )

    print(f"\n📦 Final split:")
    print(f"   X_train : {X_train.shape}")
    print(f"   X_test  : {X_test.shape}")
    print(f"   Features: {X_train.shape[1]}")

    return X_train, X_test, y_train, y_test, scaler, list(X.columns)
