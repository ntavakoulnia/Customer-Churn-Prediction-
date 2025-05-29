#!/usr/bin/env python
# coding: utf-8

#  # Data Wrangling

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time, pickle, os
from sklearn import datasets, model_selection, metrics, svm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, learning_curve

# Dataset loaded in
df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
df


# # Analyze the DataSet

# In[3]:


# Data Set inspection
df.shape
# Looking for Missing Values
df.info()
# Data Statistics
df.describe()


# # DATA CLEANING

# In[5]:


# Handling the missing values in TotalCharges
df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan).astype(float)

# Add 0's for missing TotalCharges 
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# Convert SeniorCitizen from int to categorical to identify the Category of individual citizens 
df['SeniorCitizen'] = df['SeniorCitizen'].map({0: 'No', 1: 'Yes'})
df.columns.value_counts().count()


# ## Converted columns to categorical to optimize the dataset before further analysis
# 

# In[7]:


categories = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaperlessBilling', 'PaymentMethod', 'Churn']
# Turned each column into categorical variables based on (Yes, No).
for col in categories:
    df[col] = df[col].astype('category')
df[col]


#  ## Box plot created for numerical columns of outliers

# In[9]:


# Wanted to Analyize the tenure, monthlycharge, and totalcharges to get an idea of the average charges.
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# looping through the num_cols columns
for x, col in enumerate(num_cols):
    # A boxplot is created for each 'x' subplot
    sns.boxplot(data=df, y=col, ax=axes[x])
    # title
    axes[x].set_title(col)
# Show the plot
plt.show()


# ## Box Plot Analysis
# The median tenure of 29 months shows that half of the customers stay for less than 2.5 years. A median monthly charge of 70 suggests that pricing is typically mid-range, while the median total spend of 1,000 indicates that most customers contribute a modest lifetime value before possibly churning.

# # Customer Segmentation Based on Tenure

# In[17]:


# I grouped the customers into categories based on the tenure for the years instead of the values.
# I created unique bins correlating with each of the years.
bins = [-1, 12, 24, 36, 48, 60, 72]
labels = ['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5-6yr']
# I then grouped each of the tenures, using the cut function I can turn the numerical data into unique variables so I can categorize the numerical
# column easier for instance tensure.
df['tenure_group'] = pd.cut(df['tenure'], bins=bins, labels=labels)

# I want to create a column that indicates what customers use none of the additional services.
services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
# The new column is if the customer says no to all of these services, then NoAddServices is 1, or else it is 0. 
# Returns True if the customer answered No for that particular service.
df['NoAddServices'] = (df[services] == 'No').all(axis=1).astype(int)

# I then assign each of the customers' monthly charges at the moment as their average.
df['AvgMonthlyCharge'] = df['MonthlyCharges']  
# Looks to see if the customer has any tensure, being that they worked for the company for at least 1 month. So I do not divide by 0.
some_tensure = df['tenure'] > 0  
# For customers that dont have 0 tensure, I calcualted the total monthly charge to give a more real scenario regarding their monthly spending.
# By dividing their TotalCharges by the tenure.
df.loc[some_tensure, 'AvgMonthlyCharge'] = df['TotalCharges'] / df['tenure']

# I created an interaction feature between InternetService and Contract to obtain a better analysis if possible.
# I converted the InternetService and Contract into strings and seperated by an underscore to better understand how 
# these combinations impact churn, and making it easier when creating my machine learning models. 
df['InternetContract'] = df['InternetService'].astype(str) + "_" + df['Contract'].astype(str)

# Results
print(df['tenure_group'])
print(df['AvgMonthlyCharge'])
print(df['InternetContract'])
df.to_csv('Engineered Customer Churn.csv', index=False)

# ### Post Processing
# 
# Customers are classified into tenure categories such as "0-1yr," "2-3yr," and "5-6yr" based on the results, some of the customers do not engage with additional services (NoAddServices = 1). The InternetContract feature combines internet service types and contract lengths such as 'DSL_Month-to-month'and 'Fiber optic_One year'. The average monthly charge (AvgMonthlyCharge) ranges from 29.85 to 103.20. This suggests that they have distinct churn risks and service usage trends.

# # MACHINE LEARNING MODELING PORTION CHURN OR NO CHURN
# 
# ## (RESAMPLING DATA)
# 

# In[21]:


from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Resampled the classes for the Churn and NonChurn the have the same number of samples

X = df.drop(['customerID', 'Churn'], axis=1)
# Convert target variable to binary if not already
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Separate majority and minority classes
df_majority = df[df['Churn'] == 0]
df_minority = df[df['Churn'] == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority,
                                 replace=True,                 # Sample with replacement
                                 n_samples=len(df_majority),  # Match majority class count
                                 random_state=42)             # Reproducibility

# Combine majority class with upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Shuffle the combined dataset
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into features and target
X = df_balanced.drop(['customerID', 'Churn'], axis=1)
y = df_balanced['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Display new class distribution
print("New class distribution after resampling:")
print(y_train.value_counts())


# In[23]:


# Split the Data for X into categorical and numerical columns.
# categorical_cols takes all the columns in X that contain text data/ 'Object Data' - Strings
categorical_cols = X.select_dtypes(include='object').columns
# numerical_cols is the list of names of all numerical columns like charges and tenure to name a few.
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns


# In[25]:


# I then preprocess the data to apply the proper techniques to the numerical and categorical columns.
# The numerical columns are scaled, while the categorical columns (objects) are converted to numbers via (OneHotEncoder())
# so that the machine learning algorithms can use them.
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(), categorical_cols)
])
preprocessor.fit(X_train)


# In[27]:


# Two Models are used: 'Logistic Regression' and 'Random Forest'.
# Logistic Regression is used to illustrate and predict a binary outcome, like whether a customer will say yes or no.
# Random Forest uses multiple decision trees to vote on the best decision. It reduces the weakness of a single decision tree
# by averaging over many trees, reducing overfitting.
# !pip install lightgbm
from lightgbm import LGBMClassifier

models = {
    "Logistic Regression": {
        # up to 1000 iterations are trained with the class weight being balanced to 
        # adjust the weights inversely proportional to the frequencies of the classes to treat classes fairly.
        'model': LogisticRegression(max_iter=1000,random_state=42),
        # classifier_C represents the model inside a pipeline, using values small to large, illustrates the complexity of the model
        # smaller=less complex, with larger=potential overfitting
        'params': {
            'classifier__C': [0.1, 1],  # Reduced options
            'classifier__solver': ['liblinear'],  # Fastest solver
            'classifier__class_weight': ['balanced', None]
        }
    },
        "Random Forest": {
        # the class weight being balanced to adjust the weights inversely proportional to the frequencies 
        # of the classes to treat classes fairly.
        'model': RandomForestClassifier(random_state=42,n_jobs=-1,warm_start=True,oob_score=True),
        # n_estimators represnts the number of decision tress to build 
        'params': {
            'classifier__n_estimators': [200, 250],  # More trees for stability
            'classifier__max_depth': [None],         # Let trees grow fully
            'classifier__min_samples_split': [2, 3], # More flexible splits
            'classifier__min_samples_leaf': [1, 2],  # Smaller leaves
            'classifier__max_samples': [0.6],        # More diversity in trees
            'classifier__max_features': [0.2, 0.25], # Stricter feature sampling
            'classifier__class_weight': [
                {0:1, 1:3},  # Aggressive churner weighting
                'balanced_subsample'  # Better than 'balanced' for RF
            ],
            'classifier__ccp_alpha': [0, 0.001]  # Cost-complexity pruning
        }
    }
}


# In[30]:


# Created empty dictionaries to store the evaluation results and best models
results = {}
best_models = {}

# Looped through each  of the models (Logistic Regression and Random Forest)
for model_name in models:
    print(f"\nTraining {model_name}")


 # Created a pipeline that preprocesses and then trains the models 
 # (Logistic Regression and Random Forest) for numeric and categorical data.
    pipeline = Pipeline([
        ('preprocessing', preprocessor),      
        ('classifier', models[model_name]['model'])      
    ])

    # Used GridSearchCV to try different model parameters and pick the best one
    grid_search = GridSearchCV(
        # Added Pipeline
        estimator=pipeline,                    
        # Parameter options 
         param_grid=models[model_name]['params'],
        # Evaluated the F1 score
        scoring='f1',
        # Used 5 fold cross validation to preserve the class balance
        cv=StratifiedKFold(5, shuffle=True, random_state=42),  # Better CV setup
        # Using all CPU cores to run it as quick as possible
        n_jobs=-1, 
        verbose=1  # Shows progress
    )

    # Measured how long the training takes
    # Starting timing
    start_time = time.time()  
    # Trained the models with the training data using grid_search and then fitting the data.
    grid_search.fit(X_train, y_train)   
    # Ending timing
    train_time = time.time() - start_time  

    # Saved the best version of the model 
    best_models[model_name] = grid_search.best_estimator_



    # Predicted the test set labels using the trained model
    predictions = grid_search.predict(X_test)

    # Predicted the probabilities of a positive class for AUC (area under the curve) score
    probabilities = grid_search.predict_proba(X_test)[:, 1]

    # Saved the performance metrics for this model
    results[model_name] = {
        # Best hyperparameters
        'Best Parameters': grid_search.best_params_, 
        # How long training took (Training Time)
        'Training Time (seconds)': round(train_time, 2),
         # F1 score based on test data 
        'F1 Score': f1_score(y_test, predictions),  
         # Area Under Curve Score 
        'ROC AUC Score': roc_auc_score(y_test, probabilities)    
    }
    # # 🔽 Save this model and preprocessor using a model-specific name
    # joblib.dump(grid_search.best_estimator_, f'{model_name.lower().replace(" ", "_")}_model.pkl')
    # joblib.dump(preprocessor, 'churn_preprocessor.pkl')


# In[42]:


# Converted Results to df
results_df = pd.DataFrame(results).T
print("Model Summary:")
results_df[['Training Time (seconds)','F1 Score','ROC AUC Score']]


# In[1]:


# Based on all the Models, I found the one with the best performance.

# the best_model_name find the model that performed the bested using the F1 Score with idxmax() 
# giving the index of the row with tthe maximum value for 'F1 Score' column.
best_model_name = results_df['F1 Score'].idxmax()

#The actual trained model pipeline was retreived from the best_models dictionary 
# using the best_model_name.
final_model = best_models[best_model_name]

import pickle

# Save the fitted preprocessor
with open('preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Save the trained model
with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)
    
# Generated class predictions (0 or 1) for the test set
# for the best model in final_model.
y_pred = final_model.predict(X_test)

# Generated predicted probabilities 
# predict_proba() returns a 2D array of probabilities for each class [P(class0), P(class1)]
y_prob = final_model.predict_proba(X_test)[:, 1]


# In[78]:


# The confusion Matrix illustrates the false positve/negative as well as the true postive and negatives plotted using seaborn
cm = confusion_matrix(y_test, y_pred)
# A heatmap was generated to illustrate the False and True Values.
sns.heatmap(cm, annot=True, fmt='d')
# Tittle and Labels of the plot using mat[plotlib.
plt.title(f'{best_model_name} - Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[29]:


# Calculates False Positive Rate (FPR) and True Positive Rate (TPR) for ROC curve
false_pos, true_pos, _ = roc_curve(y_test, y_prob)
#Ploted the ROC curve with the AUC score
plt.plot(false_pos, true_pos, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
# Added a line illustrating the diagonal random guess.
plt.plot([0, 1], [0, 1], 'k--')
plt.title(f'{best_model_name} - ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# # ROC Curve Analysis
# Based on the Area under the curve score of 0.947 for the sampled data compared to 0.841 for the non-resampled data it tells us that the model does aa even better job ranking the churning customer higher then the nonchurning customer 95% of the time and could be a strong performing model for churn prediction.



# In[84]:


# Plotting the Scores for the F1 and Training samples in regards to train and validations.
train_sizes, train_scores, val_scores = learning_curve(final_model, X_train, y_train, cv=5, scoring='f1')
plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Train")
plt.plot(train_sizes, np.mean(val_scores, axis=1), label="Validation")
plt.title(f"{best_model_name} - Learning Curve")
plt.xlabel("Training Samples")
plt.ylabel("F1 Score")
plt.legend()
plt.show()


# In[38]:


print(train_scores.mean())
print(val_scores.mean())


# 
# # Learning Curve Analysis 
# Based on my validation set after resampling the data to get the same number of Churns to Non-Churns,the results
# are much better. Almost all churn cases (97%) are captured. The classifier for Random Forest makes very few errors,
# with a precision and F1 scores above 85 percent. Resampling worked well for my model is now better at recognizing the minority class. 
# 
# 

# In[76]:


from sklearn.metrics import classification_report
# Classification report for the predictions made vs the test results
print(classification_report(y_test, y_pred, target_names=['Non-Churn', 'Churn']))


# # Classification Report Analysis
# 
# Based on my results I dont believe my model is more optimal then the unsampled model. I have a great F1 for churn at 89%. I have a high recall for churn at 78% meaning that I am catching most of the at risk customers and even have an great F1 for non churn at 89%. The model performs much bette when resampling the data.

#


