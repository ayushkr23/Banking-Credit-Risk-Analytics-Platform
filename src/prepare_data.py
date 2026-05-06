import pandas as pd
from sklearn.datasets import fetch_openml
import os

def prepare_dataset():
    user_data_path = r"d:\ds project\german_credit_data.csv"
    output_path = r"d:\ds project\credit-risk-project\data\german_credit_data.csv"
    
    print("Loading user dataset...")
    df_user = pd.read_csv(user_data_path, index_col=0)
    
    print("Fetching OpenML credit-g dataset for target variable...")
    # Fetch the original dataset to get the target variable
    data = fetch_openml('credit-g', version=1, as_frame=True)
    df_openml = data.frame
    
    # Map target: 'good' -> 0 (Safe Customer), 'bad' -> 1 (High Credit Risk)
    print("Mapping target variable...")
    target_mapping = {'good': 0, 'bad': 1}
    df_user['Risk'] = df_openml['class'].map(target_mapping)
    
    print(f"Dataset shape with Risk column: {df_user.shape}")
    print(f"Saving to {output_path}...")
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save without the unnamed index column if we want, but since index_col=0 was used, 
    # we can just save it with index=False if we want to reset it, or keep it.
    # Let's keep it clean without the unnamed index.
    df_user.to_csv(output_path, index=False)
    print("Data preparation complete.")

if __name__ == "__main__":
    prepare_dataset()
