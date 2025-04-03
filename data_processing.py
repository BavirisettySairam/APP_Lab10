import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def load_and_process_data(file_path):
    """Load and process the dataset"""
    df = pd.read_csv(file_path)
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Fill missing numeric values with mean
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col_name in numeric_cols:
        df[col_name] = df[col_name].fillna(df[col_name].mean())
    
    # Fill missing categorical values with mode
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col_name in categorical_cols:
        df[col_name] = df[col_name].fillna(df[col_name].mode()[0])
    
    return df

def prepare_features(df):
    """Prepare features for machine learning"""
    # Convert categorical columns to numeric
    categorical_cols = ['Division_Name', 'Department_Name', 'Class_Name']
    label_encoders = {}
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df[col + '_encoded'] = label_encoders[col].fit_transform(df[col].astype(str))
    
    # Select features
    feature_cols = ['Age', 'Rating', 'Positive_Feedback_Count'] + [col + '_encoded' for col in categorical_cols]
    X = df[feature_cols]
    
    return X 