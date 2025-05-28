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

##  Model Performance
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
- Realistic predictions
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
```bash
├── analysis/
│ ├── Deployment_Implementation.py
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── app/
│ ├── app.py
│ ├── random_forest_model.pkl
│ └── Engineered_Customer_Churn.csv
└── README.md
```D


## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt

jupyter notebook analysis/Deployment_Implementation.ipynb

cd app
streamlit run app.py
```
## Sample Visualizations

## Sample Visualizations

<table align="center">
  <tr>
    <td width="50%" align="center"><img src="https://github.com/user-attachments/assets/f6c43aa8-38ae-4dab-8c33-3b27666e3d1b" width="400px"></td>
    <td width="50%" align="center"><img src="https://github.com/user-attachments/assets/75566e05-c423-4325-a301-41871f34630d" width="400px"></td>
  </tr>
  <tr>
    <td colspan="2" align="center" padding="20px"><img src="https://github.com/user-attachments/assets/fd338906-c0d4-4d88-987f-e37fc59fa98d" width="400px"></td>
  </tr>
</table>

## Model Results
| Class      | Precision | Recall | F1-Score | Support |
|------------|-----------|--------|----------|---------|
| Non-Churn  | 0.96      | 0.78   | 0.86     | 1035    |
| Churn      | 0.82      | 0.97   | 0.89     | 1035    |
| **Accuracy** |          |        | **0.88**  | 2070    |
| Macro Avg  | 0.89      | 0.88   | 0.88     | 2070    |
| Weighted Avg| 0.89      | 0.88   | 0.88     | 2070    |


##  Future Improvements
- Enhanced feature engineering
- Additional ML models (XGBoost, LightGBM)
- API deployment
- Realistic monitoring dashboard
- Integration with CRM systems


## Application

[Watch the demo video](https://www.veed.io/view/b58aa23b-2c04-4fd4-ab38-13e241ebe1ab?panel=share)



