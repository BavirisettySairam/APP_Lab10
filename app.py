import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from data_processing import load_and_process_data, handle_missing_values
from ml_models import perform_regression, perform_clustering, perform_classification

# Set page config
st.set_page_config(page_title="Women's Clothing E-Commerce Analysis", layout="wide")

# Title
st.title("Women's Clothing E-Commerce Analysis Dashboard")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Data Overview", "Data Cleaning", "EDA", "Machine Learning"])

# Load data
@st.cache_data
def load_data():
    # In a real scenario, you would load your actual dataset here
    # For now, we'll create a sample dataset
    data = {
        'Age': np.random.randint(18, 70, 1000),
        'Title': ['Review ' + str(i) for i in range(1000)],
        'Review_Text': ['Sample review text ' + str(i) for i in range(1000)],
        'Rating': np.random.randint(1, 6, 1000),
        'Recommended_IND': np.random.randint(0, 2, 1000),
        'Positive_Feedback_Count': np.random.randint(0, 100, 1000),
        'Division_Name': np.random.choice(['General', 'General Petite', 'Initmates'], 1000),
        'Department_Name': np.random.choice(['Tops', 'Dresses', 'Bottoms', 'Intimate'], 1000),
        'Class_Name': np.random.choice(['Dresses', 'Blouses', 'Pants', 'Intimates'], 1000)
    }
    return pd.DataFrame(data)

df = load_data()

if page == "Data Overview":
    st.header("Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(df.head())
    
    st.write("### Dataset Statistics")
    st.write(df.describe())
    
    st.write("### Data Types")
    st.write(df.dtypes)

elif page == "Data Cleaning":
    st.header("Data Cleaning & Wrangling")
    
    st.write("### Missing Values Analysis")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values)
    
    st.write("### Handle Missing Values")
    if st.button("Clean Data"):
        cleaned_df = handle_missing_values(df)
        st.write("Data cleaned successfully!")
        st.dataframe(cleaned_df.head())

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Age Distribution
    st.write("### Age Distribution")
    fig = px.histogram(df, x='Age', nbins=30)
    st.plotly_chart(fig)
    
    # Rating Distribution
    st.write("### Rating Distribution")
    fig = px.pie(df, names='Rating')
    st.plotly_chart(fig)
    
    # Department Distribution
    st.write("### Department Distribution")
    fig = px.bar(df['Department_Name'].value_counts())
    st.plotly_chart(fig)
    
    # Correlation Analysis
    st.write("### Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    fig = px.imshow(corr)
    st.plotly_chart(fig)

elif page == "Machine Learning":
    st.header("Machine Learning Analysis")
    
    ml_task = st.selectbox("Select ML Task", ["Regression", "Clustering", "Classification"])
    
    if ml_task == "Regression":
        st.write("### Rating Prediction")
        if st.button("Run Regression"):
            results = perform_regression(df)
            st.write(results)
    
    elif ml_task == "Clustering":
        st.write("### Customer Segmentation")
        if st.button("Run Clustering"):
            results = perform_clustering(df)
            st.write(results)
    
    elif ml_task == "Classification":
        st.write("### Recommendation Prediction")
        if st.button("Run Classification"):
            results = perform_classification(df)
            st.write(results) 