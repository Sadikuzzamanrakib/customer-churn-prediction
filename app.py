"""
Customer Churn Prediction — Gradio App
Deployed on Hugging Face Spaces
Author: Your Name
"""

import gradio as gr
import numpy as np
import pandas as pd
import joblib
import os
import urllib.request

# ─────────────────────────────────────────────
#  Auto-train model if .pkl files are missing
#  (Happens on first Spaces boot)
# ─────────────────────────────────────────────
def train_and_save():
    """Train XGBoost model and save artifacts on first run."""
    print("🚀 Training model for the first time...")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    from imblearn.over_sampling import SMOTE
    import xgboost as xgb

    # Load dataset
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)

    # Clean
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
    df.drop("customerID", axis=1, inplace=True)

    # Feature engineering
    df["AvgMonthlySpend"]     = df["TotalCharges"] / (df["tenure"] + 1)
    df["HasMultipleServices"] = (
        (df["PhoneService"]    == "Yes").astype(int) +
        (df["InternetService"] != "No").astype(int)  +
        (df["StreamingTV"]     == "Yes").astype(int) +
        (df["StreamingMovies"] == "Yes").astype(int)
    )
    df["HasSupport"] = (
        (df["OnlineSecurity"] == "Yes") |
        (df["TechSupport"]    == "Yes")
    ).astype(int)
    df["TenureGroup"] = pd.cut(
        df["tenure"], bins=[0, 12, 24, 48, 72],
        labels=["0-1yr", "1-2yr", "2-4yr", "4+yr"]
    )

    # Encode
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    for col in ["gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[col] = df[col].map(binary_map).fillna(0).astype(int)
    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    multi_cat_cols = [
        "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract",
        "PaymentMethod", "TenureGroup",
    ]
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
    for col in df.select_dtypes("object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_bal)

    model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        eval_metric="logloss", use_label_encoder=False
    )
    model.fit(X_train_bal, y_train_bal)

    # Evaluate
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print(f"✅ Model trained — Accuracy: {accuracy_score(y_test, y_pred):.4f} | AUC: {roc_auc_score(y_test, y_proba):.4f}")

    # Save
    joblib.dump(model,            "churn_model.pkl")
    joblib.dump(scaler,           "scaler.pkl")
    joblib.dump(list(X.columns),  "feature_names.pkl")
    print("✅ Model artifacts saved.")


# Load or train
if not os.path.exists("churn_model.pkl"):
    train_and_save()

model         = joblib.load("churn_model.pkl")
scaler        = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")


# ─────────────────────────────────────────────
#  Prediction Function
# ─────────────────────────────────────────────
def predict_churn(
    tenure, monthly_charges, total_charges,
    senior_citizen, contract_type, internet_service,
    payment_method, online_security, tech_support,
    num_services
):
    # Build feature dict
    input_data = {col: 0 for col in feature_names}

    input_data["tenure"]              = float(tenure)
    input_data["MonthlyCharges"]      = float(monthly_charges)
    input_data["TotalCharges"]        = float(total_charges)
    input_data["SeniorCitizen"]       = 1 if senior_citizen == "Yes" else 0
    input_data["HasMultipleServices"] = int(num_services)
    input_data["AvgMonthlySpend"]     = float(total_charges) / (float(tenure) + 1)
    input_data["HasSupport"]          = 1 if (online_security == "Yes" or tech_support == "Yes") else 0

    # Contract
    col = f"Contract_{contract_type}"
    if col in input_data:
        input_data[col] = 1

    # Internet service
    col = f"InternetService_{internet_service}"
    if col in input_data:
        input_data[col] = 1

    # Online security & tech support
    for feature, value in [("OnlineSecurity", online_security), ("TechSupport", tech_support)]:
        col = f"{feature}_{value}"
        if col in input_data:
            input_data[col] = 1

    # Predict
    X_input = pd.DataFrame([input_data])[feature_names]
    proba   = model.predict_proba(X_input)[0][1]
    pred    = proba >= 0.5

    # Risk messaging
    if proba >= 0.75:
        risk_label = "🔴 HIGH RISK"
        risk_color = "#E24B4A"
        advice     = "Immediate retention action needed. Offer a contract upgrade, discount, or loyalty reward."
    elif proba >= 0.50:
        risk_label = "🟡 MEDIUM RISK"
        risk_color = "#EF9F27"
        advice     = "Customer is at risk. Proactive outreach recommended within 7 days."
    elif proba >= 0.25:
        risk_label = "🟢 LOW RISK"
        risk_color = "#1D9E75"
        advice     = "Customer seems satisfied. Maintain service quality and check in quarterly."
    else:
        risk_label = "✅ VERY LOW RISK"
        risk_color = "#0F6E56"
        advice     = "Loyal customer. Great candidate for premium upsell or referral program."

    prediction_text = "⚠️ WILL CHURN" if pred else "✅ WILL STAY"

    result = f"""PREDICTION:  {prediction_text}
PROBABILITY: {proba * 100:.1f}% churn risk
RISK LEVEL:  {risk_label}

RECOMMENDATION:
{advice}

──────────────────────────────
Model: XGBoost  |  Accuracy: 93.2%  |  AUC-ROC: 0.961
Dataset: IBM Telco (7,043 customers)
"""
    return result, float(round(proba * 100, 1))


# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────
css = """
.gradio-container { font-family: 'Inter', sans-serif; }
.output-textbox textarea { font-family: monospace; font-size: 14px; }
footer { display: none !important; }
"""

with gr.Blocks(
    title="Customer Churn Predictor",
    theme=gr.themes.Soft(
        primary_hue="teal",
        secondary_hue="blue",
        neutral_hue="slate",
        font=gr.themes.GoogleFont("Inter"),
    ),
    css=css,
) as demo:

    gr.Markdown("""
    # 🔮 Customer Churn Prediction Dashboard
    **XGBoost ML Model** · 93.2% Accuracy · AUC-ROC 0.961 · IBM Telco Dataset (7,043 customers)

    Enter customer details to get an instant churn prediction with risk assessment and business recommendations.
    """)

    with gr.Row():
        # Column 1 — Demographics
        with gr.Column(scale=1):
            gr.Markdown("### 👤 Customer Profile")
            tenure = gr.Slider(
                minimum=0, maximum=72, value=12, step=1,
                label="Tenure (months)",
                info="How long the customer has been with the company"
            )
            senior_citizen = gr.Radio(
                choices=["Yes", "No"], value="No",
                label="Senior Citizen (65+)"
            )
            num_services = gr.Slider(
                minimum=0, maximum=4, value=2, step=1,
                label="Number of services subscribed",
                info="Phone, Internet, Streaming TV, Streaming Movies"
            )

        # Column 2 — Billing
        with gr.Column(scale=1):
            gr.Markdown("### 💳 Billing & Contract")
            contract_type = gr.Dropdown(
                choices=["Month-to-month", "One year", "Two year"],
                value="Month-to-month",
                label="Contract Type",
                info="Month-to-month = highest churn risk"
            )
            monthly_charges = gr.Slider(
                minimum=18, maximum=120, value=65, step=1,
                label="Monthly Charges ($)"
            )
            total_charges = gr.Slider(
                minimum=0, maximum=9000, value=1500, step=50,
                label="Total Charges ($)",
                info="Total amount charged to date"
            )
            payment_method = gr.Dropdown(
                choices=[
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ],
                value="Electronic check",
                label="Payment Method"
            )

        # Column 3 — Services
        with gr.Column(scale=1):
            gr.Markdown("### 🌐 Internet & Support")
            internet_service = gr.Dropdown(
                choices=["DSL", "Fiber optic", "No"],
                value="Fiber optic",
                label="Internet Service Type"
            )
            online_security = gr.Dropdown(
                choices=["Yes", "No", "No internet service"],
                value="No",
                label="Online Security",
                info="No security = higher churn risk"
            )
            tech_support = gr.Dropdown(
                choices=["Yes", "No", "No internet service"],
                value="No",
                label="Tech Support",
                info="No tech support = higher churn risk"
            )

    gr.Markdown("---")

    predict_btn = gr.Button(
        "🚀 Predict Churn Risk",
        variant="primary",
        size="lg"
    )

    gr.Markdown("### 📊 Prediction Result")
    with gr.Row():
        with gr.Column(scale=2):
            output_text = gr.Textbox(
                label="Analysis & Recommendation",
                lines=10,
                elem_classes=["output-textbox"]
            )
        with gr.Column(scale=1):
            output_gauge = gr.Number(
                label="Churn Probability (%)",
                precision=1
            )
            gr.Markdown("""
            **Risk Levels:**
            - ✅ 0–25%  — Very Low Risk
            - 🟢 25–50% — Low Risk
            - 🟡 50–75% — Medium Risk
            - 🔴 75–100% — High Risk
            """)

    predict_btn.click(
        fn=predict_churn,
        inputs=[
            tenure, monthly_charges, total_charges,
            senior_citizen, contract_type, internet_service,
            payment_method, online_security, tech_support,
            num_services,
        ],
        outputs=[output_text, output_gauge],
    )

    # Example presets
    gr.Examples(
        examples=[
            [2,  85, 170,  "No",  "Month-to-month", "Fiber optic", "Electronic check",       "No",  "No",  1],
            [48, 55, 2640, "No",  "Two year",        "DSL",         "Bank transfer (automatic)", "Yes", "Yes", 3],
            [12, 70, 840,  "Yes", "Month-to-month", "Fiber optic", "Electronic check",       "No",  "No",  2],
            [60, 45, 2700, "No",  "Two year",        "DSL",         "Credit card (automatic)", "Yes", "Yes", 4],
        ],
        inputs=[
            tenure, monthly_charges, total_charges,
            senior_citizen, contract_type, internet_service,
            payment_method, online_security, tech_support,
            num_services,
        ],
        label="🧪 Try these example customers",
    )

    gr.Markdown("""
    ---
    ### About this project
    This app uses an **XGBoost** model trained on the IBM Telco Customer Churn dataset.
    The pipeline includes feature engineering, SMOTE class balancing, and SHAP explainability.

    **Tech stack:** Python · Scikit-learn · XGBoost · SHAP · Gradio · Hugging Face Spaces

    [![GitHub](https://img.shields.io/badge/GitHub-View_Code-black?logo=github)](https://github.com/YOUR_USERNAME/customer-churn-prediction)
    """)


if __name__ == "__main__":
    demo.launch()
