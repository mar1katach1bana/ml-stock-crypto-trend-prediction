import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, timedelta

# API configuration
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Market Prediction Dashboard",
    layout="wide"
)

# Sidebar controls
st.sidebar.title("Controls")
task = st.sidebar.radio("Task", ["Regression", "Classification"])
symbol = st.sidebar.text_input("Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Main content
st.title("Market Prediction Dashboard")

# Fetch and display predictions
if st.sidebar.button("Get Predictions"):
    try:
        # Prepare features (this should be implemented)
        features = np.random.random((10, 20)).tolist()
        
        # Make API request
        response = requests.post(
            f"{API_URL}/predict",
            json={
                "features": features,
                "task": task.lower()
            }
        )
        response.raise_for_status()
        
        # Display predictions
        predictions = response.json()
        st.subheader("Predictions")
        
        if task == "Regression":
            st.line_chart(predictions["predictions"])
        else:
            df = pd.DataFrame({
                "Prediction": predictions["predictions"],
                "Probability": predictions["probabilities"]
            })
            st.bar_chart(df["Probability"])
            st.write(df)
            
    except Exception as e:
        st.error(f"Error getting predictions: {str(e)}")

# Display model metrics
if st.sidebar.button("Get Model Metrics"):
    try:
        # Make API request
        response = requests.get(
            f"{API_URL}/metrics/{task.lower()}"
        )
        response.raise_for_status()
        
        # Display metrics
        metrics = response.json()["metrics"]
        st.subheader("Model Metrics")
        st.write(pd.DataFrame(metrics.items(), columns=["Metric", "Value"]))
        
    except Exception as e:
        st.error(f"Error getting metrics: {str(e)}")

# Display feature importance
if st.sidebar.button("Get Feature Importance"):
    try:
        # Make API request
        response = requests.post(
            f"{API_URL}/feature-importance",
            json={
                "feature_names": [f"Feature {i}" for i in range(20)],
                "top_n": 10
            }
        )
        response.raise_for_status()
        
        # Display feature importance
        importance = response.json()
        df = pd.DataFrame({
            "Feature": importance["features"],
            "Importance": importance["importance"]
        })
        
        st.subheader("Feature Importance")
        fig = px.bar(df, x="Feature", y="Importance")
        st.plotly_chart(fig)
        
    except Exception as e:
        st.error(f"Error getting feature importance: {str(e)}")

# Health check
if st.sidebar.button("Check API Health"):
    try:
        response = requests.get(f"{API_URL}/health")
        response.raise_for_status()
        st.success("API is healthy")
    except Exception as e:
        st.error(f"API health check failed: {str(e)}")
