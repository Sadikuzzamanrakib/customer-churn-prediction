"""
predict.py
──────────
Single customer churn prediction from the command line.

Usage:
    python src/predict.py \
        --tenure 12 \
        --monthly_charges 75 \
        --total_charges 900 \
        --contract "Month-to-month" \
        --internet "Fiber optic" \
        --security No \
        --tech_support No \
        --senior No \
        --services 2
"""

import argparse
import joblib
import numpy as np
import pandas as pd
import os


def load_artifacts(model_dir: str = "models"):
    model         = joblib.load(os.path.join(model_dir, "churn_model.pkl"))
    scaler        = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(model_dir, "feature_names.pkl"))
    return model, scaler, feature_names


def build_input(args, feature_names: list) -> pd.DataFrame:
    """Map CLI args → feature vector matching the training schema."""
    row = {col: 0 for col in feature_names}

    row["tenure"]              = float(args.tenure)
    row["MonthlyCharges"]      = float(args.monthly_charges)
    row["TotalCharges"]        = float(args.total_charges)
    row["SeniorCitizen"]       = 1 if args.senior.lower() == "yes" else 0
    row["HasMultipleServices"] = int(args.services)
    row["AvgMonthlySpend"]     = float(args.total_charges) / (float(args.tenure) + 1)
    row["HasSupport"]          = 1 if (
        args.security.lower() == "yes" or args.tech_support.lower() == "yes"
    ) else 0

    for col_key, val in [
        (f"Contract_{args.contract}",          1),
        (f"InternetService_{args.internet}",   1),
        (f"OnlineSecurity_{args.security}",    1),
        (f"TechSupport_{args.tech_support}",   1),
    ]:
        if col_key in row:
            row[col_key] = val

    return pd.DataFrame([row])[feature_names]


def predict(args):
    print("\n🔮 Customer Churn Predictor")
    print("─" * 40)

    model, scaler, feature_names = load_artifacts()
    X = build_input(args, feature_names)

    proba = model.predict_proba(X)[0][1]
    pred  = proba >= 0.5

    print(f"  Tenure          : {args.tenure} months")
    print(f"  Monthly Charges : ${args.monthly_charges}")
    print(f"  Contract        : {args.contract}")
    print(f"  Internet        : {args.internet}")
    print(f"  Online Security : {args.security}")
    print(f"  Tech Support    : {args.tech_support}")
    print("─" * 40)
    print(f"  PREDICTION      : {'⚠️  WILL CHURN' if pred else '✅ WILL STAY'}")
    print(f"  Churn Prob      : {proba * 100:.1f}%")

    if proba >= 0.75:
        risk = "🔴 HIGH RISK — Immediate retention action needed!"
    elif proba >= 0.50:
        risk = "🟡 MEDIUM RISK — Proactive outreach recommended."
    elif proba >= 0.25:
        risk = "🟢 LOW RISK — Maintain service quality."
    else:
        risk = "✅ VERY LOW RISK — Consider upsell opportunities."

    print(f"  Risk Level      : {risk}")
    print("─" * 40 + "\n")

    return proba


def main():
    parser = argparse.ArgumentParser(description="Predict customer churn probability.")
    parser.add_argument("--tenure",           type=float, default=12,              help="Months with company")
    parser.add_argument("--monthly_charges",  type=float, default=65,              help="Monthly charges ($)")
    parser.add_argument("--total_charges",    type=float, default=780,             help="Total charges ($)")
    parser.add_argument("--contract",         type=str,   default="Month-to-month",help="Contract type")
    parser.add_argument("--internet",         type=str,   default="Fiber optic",   help="Internet service type")
    parser.add_argument("--security",         type=str,   default="No",            help="Online security (Yes/No)")
    parser.add_argument("--tech_support",     type=str,   default="No",            help="Tech support (Yes/No)")
    parser.add_argument("--senior",           type=str,   default="No",            help="Senior citizen (Yes/No)")
    parser.add_argument("--services",         type=int,   default=2,               help="Number of services (0-4)")

    args = parser.parse_args()
    predict(args)


if __name__ == "__main__":
    main()
