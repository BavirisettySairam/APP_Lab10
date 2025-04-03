import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_process_data(file_path):
    """Load and process the dataset"""
    df = pd.read_csv(file_path)
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset"""
    # Create a copy of the dataframe
    df_cleaned = df.copy()
    
    # Fill missing numeric values with median (more robust than mean)
    numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
    for col_name in numeric_cols:
        median_value = df_cleaned[col_name].median()
        df_cleaned[col_name] = df_cleaned[col_name].fillna(median_value)
    
    # Fill missing categorical values with mode
    categorical_cols = df_cleaned.select_dtypes(include=['object']).columns
    for col_name in categorical_cols:
        mode_value = df_cleaned[col_name].mode()[0]
        df_cleaned[col_name] = df_cleaned[col_name].fillna(mode_value)
    
    return df_cleaned

def prepare_features(df):
    """Prepare features for machine learning"""
    # Create a copy of the dataframe
    df_processed = df.copy()
    
    # Convert categorical columns to numeric
    categorical_cols = ['Division_Name', 'Department_Name', 'Class_Name']
    label_encoders = {}
    
    for col in categorical_cols:
        label_encoders[col] = LabelEncoder()
        df_processed[col + '_encoded'] = label_encoders[col].fit_transform(df_processed[col].astype(str))
    
    # Create interaction features
    df_processed['Age_Rating'] = df_processed['Age'] * df_processed['Rating']
    df_processed['Age_Feedback'] = df_processed['Age'] * df_processed['Positive_Feedback_Count']
    
    # Create binary features
    df_processed['High_Rating'] = (df_processed['Rating'] >= 4).astype(int)
    df_processed['High_Feedback'] = (df_processed['Positive_Feedback_Count'] > df_processed['Positive_Feedback_Count'].median()).astype(int)
    
    # Select features
    feature_cols = [
        'Age', 'Rating', 'Positive_Feedback_Count',
        'Age_Rating', 'Age_Feedback', 'High_Rating', 'High_Feedback'
    ] + [col + '_encoded' for col in categorical_cols]
    
    # Create feature matrix
    X = df_processed[feature_cols]
    
    return X 