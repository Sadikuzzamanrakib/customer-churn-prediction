"""
train.py
────────
Train, evaluate, and save the best churn prediction model.

Usage:
    python src/train.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, roc_curve,
)
from sklearn.model_selection import GridSearchCV

import xgboost as xgb
import lightgbm as lgb

from preprocess import load_and_preprocess

os.makedirs("models",  exist_ok=True)
os.makedirs("images",  exist_ok=True)

COLORS = {"churn": "#E24B4A", "retain": "#1D9E75", "accent": "#378ADD", "purple": "#B47FDD"}


def evaluate_model(model, X_test, y_test, name, use_scaled=False, scaler=None):
    """Run predictions and return metrics dict."""
    X = scaler.transform(X_test) if use_scaled and scaler else X_test
    y_pred  = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "name"     : name,
        "accuracy" : accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall"   : recall_score(y_test, y_pred),
        "f1"       : f1_score(y_test, y_pred),
        "auc_roc"  : roc_auc_score(y_test, y_proba),
        "y_pred"   : y_pred,
        "y_proba"  : y_proba,
    }


def plot_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6), facecolor="white")
    palette = list(COLORS.values())
    for r, color in zip(results, palette):
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        ax.plot(fpr, tpr, lw=2, color=color,
                label=f'{r["name"]} (AUC={r["auc_roc"]:.3f})')
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="lower right")
    plt.tight_layout()
    plt.savefig("images/roc_curves.png", dpi=150)
    plt.close()
    print("   Saved → images/roc_curves.png")


def plot_confusion_matrix(y_test, y_pred, title="Best Model — Confusion Matrix"):
    cm = confusion_matrix(y_test, y_pred)
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5), facecolor="white")
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="RdYlGn",
        xticklabels=["Retained", "Churned"],
        yticklabels=["Retained", "Churned"],
        ax=ax, linewidths=2, annot_kws={"size": 14, "weight": "bold"},
    )
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11)
    plt.tight_layout()
    plt.savefig("images/confusion_matrix.png", dpi=150)
    plt.close()
    print("   Saved → images/confusion_matrix.png")


def plot_feature_importance(model, feature_names, top_n=15):
    importances = model.feature_importances_
    idx  = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(10, 7), facecolor="white")
    ax.barh(
        np.array(feature_names)[idx], importances[idx],
        color=COLORS["retain"], edgecolor="white", height=0.7,
    )
    ax.set_title(f"Feature Importance — Top {top_n}", fontsize=14, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=12)
    plt.tight_layout()
    plt.savefig("images/feature_importance.png", dpi=150)
    plt.close()
    print("   Saved → images/feature_importance.png")


def train():
    print("=" * 55)
    print("  CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 55)

    # ── Load data
    X_train, X_test, y_train, y_test, scaler, feature_names = load_and_preprocess()

    # ── Define models
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42), True
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1), False
        ),
        "XGBoost": (
            xgb.XGBClassifier(
                n_estimators=200, learning_rate=0.1, max_depth=6,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric="logloss", use_label_encoder=False,
            ), False
        ),
        "LightGBM": (
            lgb.LGBMClassifier(
                n_estimators=200, learning_rate=0.1,
                max_depth=6, random_state=42, verbose=-1,
            ), False
        ),
    }

    # ── Train & evaluate
    print("\n📈 Training models...\n")
    results = []
    trained = {}

    for name, (model, use_scaled) in models.items():
        Xtr = scaler.transform(X_train) if use_scaled else X_train
        model.fit(Xtr, y_train)
        r = evaluate_model(model, X_test, y_test, name,
                           use_scaled=use_scaled, scaler=scaler)
        results.append(r)
        trained[name] = model
        print(f"  ✅ {name:<22}  Acc={r['accuracy']:.3f}  "
              f"F1={r['f1']:.3f}  AUC={r['auc_roc']:.3f}")

    # ── Plots
    print("\n📊 Generating plots...")
    plot_roc_curves(results, y_test)

    best_r = max(results, key=lambda x: x["auc_roc"])
    plot_confusion_matrix(y_test, best_r["y_pred"],
                          title=f'{best_r["name"]} — Confusion Matrix')
    plot_feature_importance(trained[best_r["name"]], feature_names)

    # ── Hyperparameter tuning on best model (XGBoost)
    print(f'\n🎯 Tuning {best_r["name"]} with GridSearchCV...')
    param_grid = {
        "n_estimators" : [150, 200],
        "max_depth"    : [4, 6],
        "learning_rate": [0.05, 0.1],
        "subsample"    : [0.8, 1.0],
    }
    gs = GridSearchCV(
        xgb.XGBClassifier(random_state=42, eval_metric="logloss",
                          use_label_encoder=False),
        param_grid, cv=3, scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    gs.fit(X_train, y_train)
    tuned_model = gs.best_estimator_

    y_pred_t  = tuned_model.predict(X_test)
    y_proba_t = tuned_model.predict_proba(X_test)[:, 1]
    print(f"   Best params : {gs.best_params_}")
    print(f"   Accuracy    : {accuracy_score(y_test, y_pred_t):.4f}")
    print(f"   AUC-ROC     : {roc_auc_score(y_test, y_proba_t):.4f}")

    # ── Save artifacts
    print("\n💾 Saving model artifacts...")
    joblib.dump(tuned_model,          "models/churn_model.pkl")
    joblib.dump(scaler,               "models/scaler.pkl")
    joblib.dump(feature_names,        "models/feature_names.pkl")
    print("   Saved → models/churn_model.pkl")
    print("   Saved → models/scaler.pkl")
    print("   Saved → models/feature_names.pkl")

    print("\n🏆 TRAINING COMPLETE!")
    print(f"   Best model  : {best_r['name']}")
    print(f"   Accuracy    : {best_r['accuracy']:.4f}")
    print(f"   AUC-ROC     : {best_r['auc_roc']:.4f}")
    print("=" * 55)


if __name__ == "__main__":
    train()
