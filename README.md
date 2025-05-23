Problem Statement

The goal of this project is to analyze customer churn for a telecommunications company using the "WA_Fn-UseC_-Telco-Customer-Churn.csv" dataset. 
The project involves data wrangling, exploratory data analysis, customer segmentation, and predictive modeling to identify factors that influence customer churn 
and predict whether a customer is likely to churn.

Approach

Data Wrangling: The dataset was cleaned and preprocessed to handle missing values, convert categorical variables, and create new features for better analysis.
Exploratory Data Analysis (EDA): Visualizations and statistical summaries were used to understand the distribution of key variables such as tenure, monthly charges, 
and total charges.
Customer Segmentation: Customers were segmented based on tenure groups, and new features like average monthly charges and service usage were created.
Machine Learning Modeling: Two models (Logistic Regression and Random Forest) were trained and evaluated to predict customer churn. The best-performing model 
was selected based on F1 score and ROC AUC metrics.

Results

The Random Forest model outperformed Logistic Regression with an F1 score of 0.63 and ROC AUC of 0.84.
Key factors influencing churn include tenure, monthly charges, and the type of internet service contract.
The analysis provides actionable insights for reducing customer churn, such as targeting customers with shorter tenure or those not using additional services.

Important Files

Experiment With Various Models.ipynb: The main Jupyter notebook containing the code for data wrangling, EDA, segmentation, and machine learning modeling.
WA_Fn-UseC_-Telco-Customer-Churn.csv: The dataset used for the analysis.


Methodology

Data Cleaning: Handled missing values in the "TotalCharges" column by replacing them with zeros.
Converted numerical columns like "SeniorCitizen" to categorical for better analysis.
Created new features such as "tenure_group" to segment customers based on their tenure.

Exploratory Data Analysis

Visualized the distribution of numerical variables using box plots to identify outliers and understand central tendencies.
Analyzed the relationship between tenure and churn to identify patterns.

Customer Segmentation

Segmented customers into groups based on tenure i.e (0-1yr, 1-2yr, etc).
Created interaction features like "InternetContract" to analyze the combined effect of internet service type and contract length.

Machine Learning

Preprocessed the data using StandardScaler for numerical features and OneHotEncoder for categorical features.
Trained and evaluated Logistic Regression and Random Forest models using GridSearchCV for hyperparameter tuning.
Selected the best model based on performance metrics and interpreted the results to identify key churn factors.
