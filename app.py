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
    np.random.seed(42)
    n_samples = 1000
    
    # Generate realistic data
    data = {
        'Age': np.random.normal(35, 10, n_samples).astype(int),
        'Title': ['Review ' + str(i) for i in range(n_samples)],
        'Review_Text': ['Sample review text ' + str(i) for i in range(n_samples)],
        'Rating': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.1, 0.2, 0.3, 0.35]),
        'Recommended_IND': np.random.choice([0, 1], n_samples, p=[0.2, 0.8]),
        'Positive_Feedback_Count': np.random.poisson(5, n_samples),
        'Division_Name': np.random.choice(['General', 'General Petite', 'Initmates'], n_samples, p=[0.6, 0.3, 0.1]),
        'Department_Name': np.random.choice(['Tops', 'Dresses', 'Bottoms', 'Intimate'], n_samples, p=[0.3, 0.3, 0.2, 0.2]),
        'Class_Name': np.random.choice(['Dresses', 'Blouses', 'Pants', 'Intimates', 'Skirts', 'Sweaters'], n_samples, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Add missing values
    df.loc[df.sample(frac=0.1).index, 'Age'] = np.nan
    df.loc[df.sample(frac=0.15).index, 'Rating'] = np.nan
    df.loc[df.sample(frac=0.05).index, 'Division_Name'] = np.nan
    df.loc[df.sample(frac=0.08).index, 'Department_Name'] = np.nan
    df.loc[df.sample(frac=0.12).index, 'Class_Name'] = np.nan
    
    return df

df = load_data()

if page == "Data Overview":
    st.header("Dataset Overview")
    st.write("### Sample Data")
    st.dataframe(df.head())
    
    st.write("### Dataset Statistics")
    st.write(df.describe())
    
    st.write("### Data Types")
    st.write(df.dtypes)
    
    st.write("### Missing Values Summary")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values)

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
        
        st.write("### Missing Values After Cleaning")
        st.dataframe(cleaned_df.isnull().sum())

elif page == "EDA":
    st.header("Exploratory Data Analysis")
    
    # Age Distribution
    st.write("### Age Distribution")
    fig = px.histogram(df, x='Age', nbins=30, title='Age Distribution of Customers')
    st.plotly_chart(fig)
    
    # Rating Distribution
    st.write("### Rating Distribution")
    fig = px.pie(df, names='Rating', title='Distribution of Ratings')
    st.plotly_chart(fig)
    
    # Department Distribution
    st.write("### Department Distribution")
    fig = px.bar(df['Department_Name'].value_counts(), title='Distribution of Departments')
    st.plotly_chart(fig)
    
    # Correlation Analysis
    st.write("### Correlation Analysis")
    numeric_cols = df.select_dtypes(include=[np.number])
    corr = numeric_cols.corr()
    fig = px.imshow(corr, title='Correlation Matrix')
    st.plotly_chart(fig)
    
    # Age vs Rating
    st.write("### Age vs Rating Analysis")
    fig = px.box(df, x='Rating', y='Age', title='Age Distribution by Rating')
    st.plotly_chart(fig)
    
    # Department vs Rating
    st.write("### Department vs Rating Analysis")
    fig = px.box(df, x='Department_Name', y='Rating', title='Rating Distribution by Department')
    st.plotly_chart(fig)

elif page == "Machine Learning":
    st.header("Machine Learning Analysis")
    
    ml_task = st.selectbox("Select ML Task", ["Regression", "Clustering", "Classification"])
    
    if ml_task == "Regression":
        st.write("### Rating Prediction")
        st.write("This model predicts customer ratings based on various features.")
        if st.button("Run Regression"):
            results = perform_regression(df)
            st.write("#### Model Results")
            st.write(f"RMSE: {results['RMSE']:.4f}")
            st.write("#### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': ['Age', 'Positive_Feedback_Count', 'Division', 'Department', 'Class'],
                'Importance': results['Coefficients']
            })
            fig = px.bar(feature_importance, x='Feature', y='Importance', title='Feature Importance')
            st.plotly_chart(fig)
    
    elif ml_task == "Clustering":
        st.write("### Customer Segmentation")
        st.write("This model segments customers into different groups based on their behavior.")
        n_clusters = st.slider("Number of Clusters", 2, 5, 3)
        if st.button("Run Clustering"):
            results = perform_clustering(df, n_clusters)
            st.write("#### Model Results")
            st.write(f"Silhouette Score: {results['Silhouette Score']:.4f}")
            st.write("#### Cluster Centers")
            st.dataframe(pd.DataFrame(results['Cluster Centers'], 
                                    columns=['Age', 'Rating', 'Positive_Feedback_Count', 
                                            'Division', 'Department', 'Class']))
    
    elif ml_task == "Classification":
        st.write("### Recommendation Prediction")
        st.write("This model predicts whether a customer will recommend a product.")
        if st.button("Run Classification"):
            results = perform_classification(df)
            st.write("#### Model Results")
            st.write(f"AUC Score: {results['AUC']:.4f}")
            st.write("#### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': ['Age', 'Rating', 'Positive_Feedback_Count', 'Division', 'Department', 'Class'],
                'Importance': results['Coefficients'][0]
            })
            fig = px.bar(feature_importance, x='Feature', y='Importance', title='Feature Importance')
            st.plotly_chart(fig) 