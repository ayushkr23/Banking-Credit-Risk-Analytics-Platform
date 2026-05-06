import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report, roc_curve)

from src.data_preprocessing import load_and_preprocess_data

def evaluate_model(model, X_test, y_test, model_name, visuals_dir):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print(f"--- Evaluation for {model_name} ---")
    print(classification_report(y_test, y_pred))
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}\n")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, f'confusion_matrix_{model_name.replace(" ", "_")}.png'))
    plt.close()
    
    return acc, prec, rec, f1, roc_auc, y_prob

def train_and_evaluate():
    data_path = r"d:\ds project\credit-risk-project\data\german_credit_data.csv"
    models_dir = r"d:\ds project\credit-risk-project\models"
    visuals_dir = r"d:\ds project\credit-risk-project\visuals"
    
    X_train, X_test, y_train, y_test, feature_names, _ = load_and_preprocess_data(data_path, models_dir)
    
    print("\nTraining Logistic Regression with GridSearchCV...")
    lr_params = {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['liblinear', 'lbfgs'],
        'max_iter': [1000]
    }
    lr = LogisticRegression(random_state=42)
    lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='f1', n_jobs=-1)
    lr_grid.fit(X_train, y_train)
    best_lr = lr_grid.best_estimator_
    print(f"Best LR Params: {lr_grid.best_params_}")
    
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # Evaluate models
    metrics = {}
    
    models = {
        "Logistic Regression": best_lr,
        "Random Forest": rf
    }
    
    plt.figure(figsize=(8, 6))
    
    for name, model in models.items():
        acc, prec, rec, f1, roc_auc, y_prob = evaluate_model(model, X_test, y_test, name, visuals_dir)
        metrics[name] = f1
        
        # Plot ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, 'roc_curves_comparison.png'))
    plt.close()
    
    # Select best model (Logistic Regression is requested as primary, but let's check F1)
    best_model_name = max(metrics, key=metrics.get)
    print(f"Best model based on F1-score: {best_model_name}")
    
    # Save the Logistic Regression model as the primary application model
    print("Saving Logistic Regression model to disk...")
    joblib.dump(best_lr, os.path.join(models_dir, 'logistic_regression_model.joblib'))
    
    # Save best model too, if different
    if best_model_name != "Logistic Regression":
        joblib.dump(models[best_model_name], os.path.join(models_dir, 'best_model.joblib'))
        
    print("Model training and evaluation complete.")

if __name__ == "__main__":
    train_and_evaluate()
