# Telecom Customer Churn Analysis & Prediction Platform

## Project Overview
A comprehensive solution for telecommunications customer churn analysis and prediction, consisting of two main components:
1. Data Analysis & Model Development
2. Interactive Prediction Platform (Streamlit App)

## Key Features

### Analysis Component
- Data wrangling and preprocessing
- Exploratory Data Analysis (EDA)
- Customer segmentation
- Machine learning model development
- Feature importance analysis

### Prediction Platform (Streamlit App)
- Real time churn prediction
- Individual customer analysis
- Batch processing capabilities
- Interactive feature adjustment
- Visual risk assessment
- Automated recommendations
- Downloadable reports

## 📈 Model Performance
| Model | F1 Score | ROC AUC | Training Time (s) |
|-------|-----------|----------|------------------|
| Logistic Regression | 0.770 | 0.835 | 0.11|
| Random Forest | 0.888 | 0.947 | 41.50 |

## Advantages of the Solution

### Analysis Benefits
- Comprehensive data insights
- Robust feature engineering
- High-performance predictive modeling
- Actionable customer segments

### Streamlit App Benefits
- User-friendly interface
- Real-time predictions
- Adjustable parameters
- Visual insights
- Batch processing capability
- Automated recommendations
- Exportable results

## Technical Stack
- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- Streamlit
- Plotly
- Joblib

## Repository Structure

├── analysis/
│ ├── Deployment_Implementation.ipynb
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── app/
│ ├── app.py
│ ├── random_forest_model.pkl
│ └── Engineered_Customer_Churn.csv
├── requirements.txt
└── README.md


## 🚀 Getting Started

### Prerequisites
```bash
pip install -r requirements.txt

jupyter notebook analysis/Deployment_Implementation.ipynb

cd app
streamlit run app.py

Sample Visualizations

[image](https://github.com/user-attachments/assets/f6c43aa8-38ae-4dab-8c33-3b27666e3d1b)

[Uploading data_image_png;base64,iVBORw0KGgoAAAANSUhEUgAAAiQAAAHFCAYAAADCA+LKAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjguNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8fJSN1AAAACXBIWXMAAA9hAAAPYQGoP6dpAABClUlEQVR4nO3df3zP9f7_8fvbfrxtM2sb++W3WkVTREejQn6GWB0_ikoHHRJa.url…]()


Training Time (seconds) 	F1 Score 	ROC AUC Score
Logistic Regression 	0.11 	0.770499 	0.835225
Random Forest 	41.5 	0.887809 	0.946634



Application

[Watch the demo video](https://www.veed.io/view/b58aa23b-2c04-4fd4-ab38-13e241ebe1ab?panel=share)



