import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_and_evaluate
from src.prepare_data import prepare_dataset

def main():
    print("=========================================")
    print("  Credit Risk Prediction System Pipeline ")
    print("=========================================")
    
    # 1. Prepare data (fetching target)
    print("\n--- Step 1: Preparing Data ---")
    prepare_dataset()
    
    # 2. Train Models
    print("\n--- Step 2: Training and Evaluating Models ---")
    train_and_evaluate()
    
    print("\n=========================================")
    print(" Pipeline completed successfully! ")
    print(" To view the EDA visuals, check the 'visuals' folder.")
    print(" To start the app, run: streamlit run app/app.py")
    print("=========================================")

if __name__ == "__main__":
    main()
