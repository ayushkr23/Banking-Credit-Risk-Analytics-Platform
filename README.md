# 🏦 AI-Powered Credit Risk Prediction System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

An end-to-end Machine Learning pipeline and interactive web application that predicts the likelihood of a customer defaulting on a loan based on financial and personal attributes. Built with **Python, Scikit-Learn, and Streamlit**, featuring a stunning dark-mode Glassmorphism UI.

## ✨ Features
* **Full Data Pipeline**: Automated preprocessing, missing value imputation, and One-Hot Encoding.
* **Class Balancing**: Utilizes SMOTE (Synthetic Minority Over-sampling Technique) to ensure the model learns equally from Safe and High-Risk customers.
* **Hyperparameter Optimization**: Uses `GridSearchCV` to find the absolute best Logistic Regression parameters.
* **Real-time Inference**: Interactive Streamlit dashboard to test user data instantly.
* **Local Model Explainability**: Instead of black-box predictions, the app transparently explains exactly *which* features (and by how much) increased or decreased the applicant's risk score using Logistic Regression log-odds coefficients.
* **Premium UI/UX**: Custom CSS featuring glassmorphism, animated gradients, and high-fidelity data visualization.

## 📁 Project Structure

```text
credit-risk-project/
│
├── app/
│   └── app.py                  # Streamlit Web Dashboard
├── data/
│   └── german_credit_data.csv  # Dataset used for training
├── models/
│   ├── preprocessor.joblib     # Saved scaling/encoding pipeline
│   └── logistic_regression_model.joblib # Saved best model
├── notebooks/
│   └── EDA.py                  # Exploratory Data Analysis Script
├── src/
│   ├── prepare_data.py         # Script to fetch/merge the dataset
│   ├── data_preprocessing.py   # Data cleaning and SMOTE processing
│   ├── model_training.py       # ML Training and evaluation
│   └── inference.py            # Real-time prediction and explainability API
├── visuals/                    # Generated EDA and Model Evaluation Charts
├── main.py                     # Master execution script to run the full ML pipeline
├── requirements.txt            # Project dependencies
└── README.md                   # This file
```

## 🚀 Quick Start (Run Locally)

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/credit-risk-predictor.git
   cd credit-risk-predictor
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Run the complete Machine Learning Pipeline:**
   If you want to retrain the models from scratch and regenerate the EDA visualizations, run:
   ```bash
   python main.py
   ```

4. **Launch the Streamlit Web App:**
   ```bash
   streamlit run app/app.py
   ```

## ☁️ Deployment (Streamlit Community Cloud)

This project is perfectly structured for instantaneous deployment via **Streamlit Community Cloud**.
1. Push this code to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **"New App"**.
4. Select your GitHub repository.
5. Set the Main file path to: `app/app.py`
6. Click **Deploy!**

## 📊 Dataset Attribution
This project utilizes the German Credit Risk dataset originally sourced from OpenML/Kaggle, modified to merge the definitive binary target variable.
