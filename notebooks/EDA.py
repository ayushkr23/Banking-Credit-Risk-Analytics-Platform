import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def perform_eda():
    print("Starting Exploratory Data Analysis...")
    data_path = r"d:\ds project\credit-risk-project\data\german_credit_data.csv"
    visuals_dir = r"d:\ds project\credit-risk-project\visuals"
    
    os.makedirs(visuals_dir, exist_ok=True)
    
    df = pd.read_csv(data_path)
    
    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="muted")
    
    # 1. Null Value Analysis
    plt.figure(figsize=(8, 5))
    null_counts = df.isnull().sum()
    sns.barplot(x=null_counts.values, y=null_counts.index, palette="viridis")
    plt.title("Missing Values per Feature")
    plt.xlabel("Count of Missing Values")
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "missing_values.png"))
    plt.close()
    
    # 2. Class Imbalance (Risk Distribution)
    plt.figure(figsize=(6, 5))
    sns.countplot(data=df, x='Risk', palette="Set2")
    plt.title("Risk Distribution (0: Safe, 1: High Risk)")
    plt.xlabel("Risk Category")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "risk_distribution.png"))
    plt.close()
    
    # 3. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=['int64', 'float64'])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Numeric Features")
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "correlation_heatmap.png"))
    plt.close()
    
    # 4. Job vs Default (proxy for Income vs Default)
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x='Job', hue='Risk', palette="Set1")
    plt.title("Job Category vs Default Risk")
    plt.xlabel("Job (0: Unskilled Non-resident, 1: Unskilled Resident, 2: Skilled, 3: Highly Skilled)")
    plt.ylabel("Count")
    plt.legend(title='Risk', labels=['0: Safe', '1: High Risk'])
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "job_vs_risk.png"))
    plt.close()
    
    # 5. Loan Amount vs Default (Boxplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Risk', y='Credit amount', palette="Pastel1")
    plt.title("Credit Amount Distribution by Risk Category")
    plt.xlabel("Risk Category")
    plt.ylabel("Credit Amount (DM)")
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "loan_amount_vs_risk.png"))
    plt.close()
    
    # 6. Duration vs Default (Boxplot)
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x='Risk', y='Duration', palette="Pastel2")
    plt.title("Loan Duration Distribution by Risk Category")
    plt.xlabel("Risk Category")
    plt.ylabel("Duration (months)")
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "duration_vs_risk.png"))
    plt.close()
    
    # 7. Age distribution Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(data=df, x='Age', hue='Risk', multiple="stack", bins=30, palette="husl")
    plt.title("Age Distribution by Risk")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(visuals_dir, "age_distribution_by_risk.png"))
    plt.close()
    
    print("EDA Visuals generated successfully in visuals/ folder.")

if __name__ == "__main__":
    perform_eda()
