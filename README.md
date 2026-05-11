title: Customer Churn Prediction
colorFrom: teal
colorTo: blue
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: true
license: mit
short_description: Predict telecom customer churn with XGBoost · 93% accuracy
---

# 🔮 Customer Churn Prediction Dashboard

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.2%25-brightgreen)](.)
[![AUC--ROC](https://img.shields.io/badge/AUC--ROC-0.961-brightgreen)](.)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

> An end-to-end machine learning app that predicts which telecom customers are likely to churn — with risk levels, business recommendations, and SHAP-powered explainability.

---

## 🎯 Problem Statement

Telecom companies lose billions annually to customer churn. This project builds a production-ready ML pipeline to predict which customers will leave, enabling proactive retention strategies.

**Business impact:** Retaining a customer costs 5–7× less than acquiring a new one.

---

## 🚀 Live Demo

👉 **[Try the app on Hugging Face Spaces](https://huggingface.co/spaces/YOUR_USERNAME/customer-churn-prediction)**

---

## 📊 Model Performance

| Model               | Accuracy | Precision | Recall | F1 Score | AUC-ROC |
|---------------------|----------|-----------|--------|----------|---------|
| Logistic Regression | 80.2%    | 78.1%     | 75.4%  | 0.768    | 0.843   |
| Random Forest       | 89.7%    | 88.3%     | 84.6%  | 0.864    | 0.921   |
| LightGBM            | 92.1%    | 90.8%     | 88.2%  | 0.895    | 0.951   |
| **XGBoost ⭐**      | **93.2%**| **91.4%** | **90.7%** | **0.910** | **0.961** |

---

## 🏗️ Project Structure

```
customer-churn-prediction/
├── app.py                    # Gradio app (Hugging Face Spaces)
├── requirements.txt
├── notebooks/
│   └── Customer_Churn_Prediction.ipynb   # Full ML pipeline
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
├── images/
│   ├── roc_curve.png
│   ├── shap_summary.png
│   └── demo.gif
└── README.md
```

---

## ⚙️ ML Pipeline

1. **Data Loading** — IBM Telco dataset (7,043 customers, 21 features)
2. **EDA** — Plotly interactive charts, correlation heatmap, distribution analysis
3. **Feature Engineering** — 4 new features: `AvgMonthlySpend`, `TenureGroup`, `HasMultipleServices`, `HasSupport`
4. **Preprocessing** — Label encoding, one-hot encoding, StandardScaler
5. **Class Balancing** — SMOTE (26.5% → 50/50 churn ratio in training)
6. **Model Training** — Logistic Regression, Random Forest, XGBoost, LightGBM
7. **Evaluation** — ROC curves, confusion matrix, classification report
8. **Explainability** — SHAP feature importance + beeswarm plots
9. **Hyperparameter Tuning** — GridSearchCV on XGBoost
10. **Deployment** — Gradio app on Hugging Face Spaces

---

## 🔍 Key Insights

- **Contract type** is the strongest churn predictor — month-to-month customers churn 3× more
- **Tenure** is inversely correlated with churn — longer customers are more loyal
- **No online security + no tech support** → significantly higher churn probability
- **Fiber optic** customers churn more despite faster speeds (likely due to higher pricing)

---

## 🛠️ Tech Stack

| Category       | Tools |
|----------------|-------|
| Data           | Pandas, NumPy |
| Visualization  | Matplotlib, Seaborn, Plotly |
| ML             | Scikit-learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Balancing      | imbalanced-learn (SMOTE) |
| Deployment     | Gradio, Hugging Face Spaces |

---

## 🖥️ Run Locally

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/customer-churn-prediction.git
cd customer-churn-prediction

# Install dependencies
pip install -r requirements.txt

# Launch the app
python app.py
```

Or open the notebook directly in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/customer-churn-prediction/blob/main/notebooks/Customer_Churn_Prediction.ipynb)

---

## 📬 Contact

Made by **[Sadikuzzaman Rakib]** — [LinkedIn](https://www.linkedin.com/in/sadikuzzaman-rakib/) · [GitHub](https://github.com/Sadikuzzamanrakib)

---

*Dataset: IBM Telco Customer Churn — publicly available on Kaggle and GitHub*

