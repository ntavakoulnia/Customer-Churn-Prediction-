import streamlit as st
import pandas as pd
import joblib
import os
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from typing import Dict, List

# === Configuration ===
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# File paths and column names
DATA_PATH = "Engineered Customer Churn.csv"
MODEL_PATH = "random_forest_model.pkl"
CUSTOMER_ID_COLUMN = "customerID"  # Exact column name from your CSV

# Model performance metrics
MODEL_METRICS = {
    "accuracy": 0.82,
    "precision": 0.78,
    "recall": 0.85,
    "f1": 0.81,
    "threshold": 0.5
}

# === Helper Functions ===
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    """Load and cache the dataset."""
    try:
        df = pd.read_csv(path)
        # Clean column names and remove auto-generated indices
        df.columns = [col.strip() for col in df.columns]
        df = df.drop(columns=[col for col in df.columns 
                            if col.lower() in ['unnamed: 0', 'unnamed0', 'index']], 
                   errors='ignore')
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_resource
def load_model(path: str):
    """Load and cache the ML model."""
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def get_customer_data(df: pd.DataFrame, customer_id: str) -> Dict:
    """Get all data for a specific customer ID."""
    try:
        return df[df[CUSTOMER_ID_COLUMN] == customer_id].iloc[0].to_dict()
    except IndexError:
        st.warning(f"Customer {customer_id} not found")
        return None
    except Exception as e:
        st.error(f"Error loading customer data: {e}")
        return None

def preprocess_input(input_dict: Dict, df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess input data to match training data format."""
    input_df = pd.DataFrame([input_dict])
    
    # Convert numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
    
    # Convert categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
    
    return input_df

def get_feature_importance(model, features: List[str]) -> pd.DataFrame:
    """Extract and sort feature importance."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = model.coef_[0]
    else:
        return None
        
    return pd.DataFrame({
        'Feature': features,
        'Importance': importance
    }).sort_values('Importance', ascending=False)

def main():
    st.title("🎯 Customer Churn Predictor Pro")
    
    # Load data and model
    if not all(os.path.exists(path) for path in [DATA_PATH, MODEL_PATH]):
        st.error("Required files not found. Please check the paths.")
        st.stop()
    
    df = load_data(DATA_PATH)
    model = load_model(MODEL_PATH)
    
    if df is None or model is None:
        st.stop()
    
    # Validate customer ID column exists
    if CUSTOMER_ID_COLUMN not in df.columns:
        st.error(f"Column '{CUSTOMER_ID_COLUMN}' not found in dataset. Available columns: {list(df.columns)}")
        st.stop()
    
    # Remove target column if present
    if 'Churn' in df.columns:
        df = df.drop(columns=['Churn'])
    
    # Sidebar with model info
    with st.sidebar:
        st.header("Model Information")
        st.metric("Accuracy", f"{MODEL_METRICS['accuracy']:.1%}")
        st.metric("Precision", f"{MODEL_METRICS['precision']:.1%}")
        st.metric("Recall", f"{MODEL_METRICS['recall']:.1%}")
        st.metric("Decision Threshold", f"{MODEL_METRICS['threshold']:.0%}")
        
        st.divider()
        st.write("**Model Type:** Random Forest")
        st.write("**Training Date:** 2023-11-15")
        st.write("**Version:** 2.2.0")  # Updated version
    
    # Main interface
    tab1, tab2 = st.tabs(["Single Customer Analysis", "Batch Processing"])
    
    with tab1:
        st.subheader("🔍 Customer Profile Analysis")
        
        # Customer ID selection
        customer_ids = sorted(df[CUSTOMER_ID_COLUMN].unique())
        selected_id = st.selectbox(
            f"Select {CUSTOMER_ID_COLUMN}", 
            customer_ids,
            help="Select a customer to view their profile and predict churn risk"
        )
        
        # Load customer data
        customer_data = get_customer_data(df, selected_id)
        
        if customer_data:
            # Display customer profile in expandable sections
            with st.expander("📋 Customer Demographics", expanded=True):
                cols = st.columns(3)
                demo_fields = ['Gender', 'SeniorCitizen', 'Partner', 'Dependents']
                for i, field in enumerate(demo_fields):
                    if field in customer_data:
                        cols[i%3].text_input(
                            label=field,
                            value=str(customer_data.get(field, 'N/A')),
                            disabled=True
                        )
            
            with st.expander("📞 Service Information", expanded=False):
                cols = st.columns(3)
                service_fields = ['PhoneService', 'MultipleLines', 'InternetService',
                                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                                'TechSupport', 'StreamingTV', 'StreamingMovies']
                for i, field in enumerate(service_fields):
                    if field in customer_data:
                        cols[i%3].text_input(
                            label=field,
                            value=str(customer_data.get(field, 'N/A')),
                            disabled=True
                        )
            
            # ====== USAGE METRICS SECTION (ADJUSTABLE) ======
            with st.expander("📊 Usage Metrics (Adjustable)", expanded=True):
                usage_fields = ['MonthlyCharges', 'TotalCharges', 'Tenure']
                usage_values = {}
                
                for field in usage_fields:
                    if field in customer_data:
                        # Get min/max from dataset
                        min_val = float(df[field].min())
                        max_val = float(df[field].max())
                        current_val = float(customer_data.get(field, min_val))
                        
                        # Create adjustable input
                        usage_values[field] = st.slider(
                            label=field,
                            min_value=min_val,
                            max_value=max_val,
                            value=current_val,
                            step=0.01 if field in ['MonthlyCharges', 'TotalCharges'] else 1.0,
                            help=f"Adjust {field} value (Dataset range: {min_val:.2f}-{max_val:.2f})"
                        )
            
            with st.expander("💰 Billing Information", expanded=False):
                cols = st.columns(3)
                billing_fields = ['Contract', 'PaperlessBilling', 'PaymentMethod']
                for i, field in enumerate(billing_fields):
                    if field in customer_data:
                        cols[i%3].text_input(
                            label=field,
                            value=str(customer_data.get(field, 'N/A')),
                            disabled=True
                        )
            
            # Feature importance
            st.subheader("🔍 Model Insights")
            feature_importance = get_feature_importance(model, [col for col in df.columns if col != CUSTOMER_ID_COLUMN])
            if feature_importance is not None:
                fig = px.bar(
                    feature_importance.head(10),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 10 Most Predictive Features"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Prediction button
            if st.button("🔮 Predict Churn Risk", type="primary", use_container_width=True):
                with st.spinner("Analyzing customer..."):
                    try:
                        # Create copy of customer data with adjusted usage metrics
                        prediction_data = customer_data.copy()
                        for field, value in usage_values.items():
                            prediction_data[field] = value
                        
                        input_df = preprocess_input(prediction_data, df)
                        proba = model.predict_proba(input_df)[0][1]
                        prediction = int(proba >= MODEL_METRICS['threshold'])
                        
                        # Results display
                        with st.container(border=True):
                            st.subheader("🎯 Prediction Results")
                            col_res1, col_res2 = st.columns(2)
                            
                            with col_res1:
                                # Risk indicator
                                if prediction == 1:
                                    st.error(f"⚠️ High Risk of Churn (Probability: {proba:.1%})")
                                else:
                                    st.success(f"✅ Low Risk of Churn (Probability: {proba:.1%})")
                                
                                # Gauge chart
                                fig = go.Figure(go.Indicator(
                                    mode="gauge+number",
                                    value=proba*100,
                                    domain={'x': [0, 1], 'y': [0, 1]},
                                    title={'text': "Churn Risk Score"},
                                    gauge={
                                        'axis': {'range': [0, 100]},
                                        'bar': {'color': "darkblue"},
                                        'steps': [
                                            {'range': [0, 30], 'color': "lightgreen"},
                                            {'range': [30, 70], 'color': "yellow"},
                                            {'range': [70, 100], 'color': "red"}],
                                        'threshold': {
                                            'line': {'color': "black", 'width': 4},
                                            'thickness': 0.75,
                                            'value': proba*100
                                        }
                                    }
                                ))
                                st.plotly_chart(fig, use_container_width=True)
                            
                            with col_res2:
                                # Recommendations
                                st.subheader("💡 Retention Strategy")
                                if proba > 0.7:
                                    st.error("🔥 Immediate Action Required")
                                    st.write("- Personal retention offer (15-20% discount)")
                                    st.write("- Dedicated account manager")
                                    st.write("- Service review meeting")
                                elif proba > MODEL_METRICS['threshold']:
                                    st.warning("⚠️ Elevated Risk Detected")
                                    st.write("- Customer satisfaction survey")
                                    st.write("- Usage pattern analysis")
                                    st.write("- Targeted promotions")
                                else:
                                    st.info("✅ Healthy Customer")
                                    st.write("- Regular check-ins")
                                    st.write("- Value-added content")
                        
                        # Show what changed from original
                        st.subheader("🔀 Adjusted Parameters")
                        changes = []
                        for field in usage_fields:
                            original = float(customer_data.get(field, 0))
                            adjusted = float(prediction_data.get(field, 0))
                            if original != adjusted:
                                changes.append(f"{field}: {original:.2f} → {adjusted:.2f}")
                        
                        if changes:
                            st.write("Modified usage metrics:")
                            st.write("\n".join(changes))
                        else:
                            st.info("No usage metrics were adjusted")
                    
                    except Exception as e:
                        st.error(f"Prediction failed: {str(e)}")
    
    with tab2:
        st.subheader("📁 Batch Processing")
        uploaded_file = st.file_uploader(f"Upload CSV with {CUSTOMER_ID_COLUMN} column", type=["csv"])
        
        if uploaded_file:
            try:
                batch_df = pd.read_csv(uploaded_file)
                # Clean auto-generated columns
                batch_df = batch_df.drop(columns=[col for col in batch_df.columns 
                                               if col.lower() in ['unnamed: 0', 'unnamed0', 'index']], 
                                      errors='ignore')
                
                if CUSTOMER_ID_COLUMN not in batch_df.columns:
                    st.error(f"Uploaded file must contain '{CUSTOMER_ID_COLUMN}' column")
                    st.stop()
                
                results = []
                for _, row in batch_df.iterrows():
                    try:
                        customer_data = get_customer_data(df, row[CUSTOMER_ID_COLUMN])
                        if not customer_data:
                            st.warning(f"Customer {row[CUSTOMER_ID_COLUMN]} not found in database")
                            continue
                            
                        input_df = preprocess_input(customer_data, df)
                        proba = model.predict_proba(input_df)[0][1]
                        prediction = int(proba >= MODEL_METRICS['threshold'])
                        
                        results.append({
                            CUSTOMER_ID_COLUMN: row[CUSTOMER_ID_COLUMN],
                            'Churn_Probability': proba,
                            'Prediction': prediction,
                            'Risk_Level': 'High' if prediction == 1 else 'Low'
                        })
                    except Exception as e:
                        st.warning(f"Skipped {CUSTOMER_ID_COLUMN} {row.get(CUSTOMER_ID_COLUMN, 'Unknown')}: {str(e)}")
                
                if results:
                    result_df = pd.DataFrame(results)
                    
                    # Display results
                    st.success(f"Processed {len(result_df)}/{len(batch_df)} customers successfully")
                    
                    # Summary stats
                    st.subheader("📊 Batch Summary")
                    cols = st.columns(3)
                    with cols[0]:
                        st.metric("High Risk", f"{sum(result_df['Prediction'])} ({sum(result_df['Prediction'])/len(result_df):.1%})")
                    with cols[1]:
                        st.metric("Avg Probability", f"{result_df['Churn_Probability'].mean():.1%}")
                    with cols[2]:
                        st.metric("Max Probability", f"{result_df['Churn_Probability'].max():.1%}")
                    
                    # Visualization
                    fig = px.histogram(
                        result_df,
                        x='Churn_Probability',
                        color='Risk_Level',
                        nbins=20,
                        title='Risk Distribution by Prediction',
                        color_discrete_map={'High': 'red', 'Low': 'green'}
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download
                    st.download_button(
                        "💾 Download Results",
                        result_df.to_csv(index=False).encode('utf-8'),
                        "churn_predictions.csv",
                        "text/csv"
                    )
                
            except Exception as e:
                st.error(f"Batch processing failed: {str(e)}")

if __name__ == "__main__":
    main()