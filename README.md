Problem Statement

The goal of this project is to analyze customer churn for a telecommunications company using the "WA_Fn-UseC_-Telco-Customer-Churn.csv" dataset. 
The project involves data wrangling, exploratory data analysis, customer segmentation, and predictive modeling to identify factors that influence customer churn 
and predict whether a customer is likely to churn.

Approach

Data Wrangling: Missing values in the TotalCharges column were handled by replacing empty entries with NaN, then filling those with 0. Categorical features such as SeniorCitizen were recoded to numerical types for clarity and analysis. Multiple service related features were converted to categorical types for model compatibility.

Exploratory Data Analysis (EDA): Box plots were generated for key numerical features (tenure, MonthlyCharges, TotalCharges) to assess distributions and detect outliers. Tenure was also segmented into groups for deeper pattern recognition.

Customer Segmentation: Customers were segmented based on tenure groups, using bins to segment customers by how long they’ve been with the company i.e (0-1yr, 1-2yr, etc). 
New features like average monthly charges and service usage were created. A new binary feature called NoAddServices was made to flag customers who don’t use any additional 
services i.e (OnlineBackup, StreamingTV, etc)

Modeling and Evaluation

Used StandardScaler and OneHotEncoder via ColumnTransformer to prepare data for modeling.

Two models (Logistic Regression and Random Forest) were trained and evaluated to predict customer churn, using GridSearchCV for 
hyperparameter tuning and cross validation. The best performing model was selected based on F1 score and ROC AUC metrics.
The Random Forest model achieving the best results with a (F1: 0.89, ROC AUC: 0.947).

Key Results

Customers with shorter tenure, higher monthly charges, and no contract based services were more likely to churn.
Segmenting customers and creating interaction features improved model performance and interpretability.
The Random Forest model was selected for deployment due to its higher predictive power compared to Logistic Regression.

Important Files

Deployment Implementation.ipynb: The main Jupyter notebook containing the code for data wrangling, EDA, segmentation, and machine learning modeling.
WA_Fn-UseC_-Telco-Customer-Churn.csv: The dataset used for the analysis.

Application

[Watch the demo video](https://www.veed.io/view/b58aa23b-2c04-4fd4-ab38-13e241ebe1ab?panel=share)



