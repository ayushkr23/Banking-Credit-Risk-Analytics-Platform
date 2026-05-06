import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import joblib
import os

def load_and_preprocess_data(data_path, models_dir):
    print("Loading data...")
    df = pd.read_csv(data_path)
    
    # Handle missing values ('NA' strings or NaN)
    # In German Credit dataset, NA in Checking/Saving means "no account"
    df['Saving accounts'] = df['Saving accounts'].fillna('no_account')
    df['Checking account'] = df['Checking account'].fillna('no_account')
    
    # Separate features and target
    X = df.drop('Risk', axis=1)
    y = df['Risk']
    
    # Define categorical and numerical features
    categorical_features = ['Sex', 'Housing', 'Saving accounts', 'Checking account', 'Purpose']
    numerical_features = ['Age', 'Job', 'Credit amount', 'Duration']
    
    print("Applying transformations...")
    # Preprocessing pipeline for numerical and categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ])
    
    # Fit and transform features
    X_processed = preprocessor.fit_transform(X)
    
    # Get feature names after one-hot encoding
    cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    feature_names = numerical_features + list(cat_feature_names)
    
    # Convert processed data back to dataframe for easier handling (optional but good for SHAP)
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)
    
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Applying SMOTE to handle class imbalance on training data...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Original training shape: {X_train.shape}, Target distribution:\n{y_train.value_counts()}")
    print(f"Resampled training shape: {X_train_resampled.shape}, Target distribution:\n{y_train_resampled.value_counts()}")
    
    # Save the preprocessor
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(models_dir, 'preprocessor.joblib'))
    
    return X_train_resampled, X_test, y_train_resampled, y_test, feature_names, preprocessor

if __name__ == "__main__":
    data_path = r"d:\ds project\credit-risk-project\data\german_credit_data.csv"
    models_dir = r"d:\ds project\credit-risk-project\models"
    X_train, X_test, y_train, y_test, features, prep = load_and_preprocess_data(data_path, models_dir)
    print("Preprocessing complete!")
