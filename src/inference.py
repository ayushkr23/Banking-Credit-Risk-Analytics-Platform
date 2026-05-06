import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class CreditRiskPredictor:
    def __init__(self, models_dir=r"d:\ds project\credit-risk-project\models"):
        self.models_dir = models_dir
        self.preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.joblib'))
        self.model = joblib.load(os.path.join(models_dir, 'logistic_regression_model.joblib'))
        
        # We need the feature names for SHAP
        self.categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
        self.numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']
        self.cat_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
        self.feature_names = self.numerical_features + list(self.cat_feature_names)

    def predict(self, input_data):
        """
        input_data: A pandas DataFrame containing a single row of user input
        Returns: probability of default, risk category, and processed input
        """
        # Ensure correct column order
        cols = ['Age', 'Sex', 'Job', 'Housing', 'Saving accounts', 'Checking account', 'Credit amount', 'Duration', 'Purpose']
        input_df = input_data[cols].copy()
        
        # Preprocess
        X_processed = self.preprocessor.transform(input_df)
        X_processed_df = pd.DataFrame(X_processed, columns=self.feature_names)
        
        # Predict
        prob = self.model.predict_proba(X_processed_df)[0][1]
        
        # Categorize
        if prob < 0.3:
            category = "Low Risk"
        elif prob < 0.7:
            category = "Medium Risk"
        else:
            category = "High Risk"
            
        return prob, category, X_processed_df

    def explain_prediction(self, X_processed_df, visuals_dir=r"d:\ds project\credit-risk-project\visuals"):
        """
        Generate local explainer for a single prediction using Logistic Regression coefficients.
        """
        os.makedirs(visuals_dir, exist_ok=True)
        
        # Get coefficients and feature values
        coefs = self.model.coef_[0]
        feature_vals = X_processed_df.iloc[0].values
        
        # Calculate impacts (feature value * coefficient)
        impacts = feature_vals * coefs
        
        # Plot feature impacts with dark premium theme
        plt.style.use('dark_background')
        fig, ax = plt.figure(figsize=(10, 5), facecolor='none'), plt.gca()
        ax.set_facecolor('none')
        
        impact_series = pd.Series(impacts, index=self.feature_names)
        colors = ['#ff4b2b' if val > 0 else '#00b09b' for val in impact_series.sort_values()]
        
        impact_series.sort_values().plot(kind='barh', color=colors, ax=ax)
        
        plt.title('Local Feature Impacts (Log-Odds)', color='white', pad=20, fontsize=16, fontweight='bold')
        plt.grid(color='#333333', linestyle='--', linewidth=0.5, alpha=0.5)
        
        # Remove spines
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        plt.tight_layout()
        plt.savefig(os.path.join(visuals_dir, 'shap_local_explanation.png'), bbox_inches='tight', transparent=True, dpi=150)
        plt.close()
        
        # Return top positive and negative feature impacts for UI display
        impacts_list = list(zip(self.feature_names, impacts))
        impacts_list.sort(key=lambda x: x[1], reverse=True)
        
        top_risk_factors = [f"{feat} (+{val:.3f})" for feat, val in impacts_list[:3] if val > 0]
        top_safe_factors = [f"{feat} ({val:.3f})" for feat, val in impacts_list[-3:] if val < 0]
        
        return top_risk_factors, top_safe_factors

if __name__ == "__main__":
    predictor = CreditRiskPredictor()
    print("Predictor initialized successfully.")
