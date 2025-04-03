from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np
import pandas as pd
from data_processing import prepare_features

def perform_regression(df):
    """Perform regression analysis to predict ratings"""
    # Prepare features
    X = prepare_features(df)
    y = df['Rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_rmse = -cv_scores.mean()
    
    return {
        'RMSE': rmse,
        'CV_RMSE': cv_rmse,
        'Coefficients': model.feature_importances_,
        'Feature Names': X.columns.tolist()
    }

def perform_clustering(df, n_clusters=3):
    """Perform clustering analysis for customer segmentation"""
    # Prepare features
    X = prepare_features(df)
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create and train model
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(X_scaled)
    
    # Get predictions
    labels = kmeans.labels_
    
    # Evaluate model
    silhouette = silhouette_score(X_scaled, labels)
    
    # Get cluster centers in original scale
    cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
    
    # Calculate cluster statistics
    df['Cluster'] = labels
    cluster_stats = df.groupby('Cluster').agg({
        'Age': ['mean', 'std'],
        'Rating': ['mean', 'std'],
        'Positive_Feedback_Count': ['mean', 'std']
    })
    
    return {
        'Silhouette Score': silhouette,
        'Cluster Centers': cluster_centers,
        'Cluster Statistics': cluster_stats,
        'Labels': labels
    }

def perform_classification(df):
    """Perform classification to predict recommendations"""
    # Prepare features
    X = prepare_features(df)
    y = df['Recommended_IND']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create and train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)
    
    # Evaluate model
    auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='roc_auc')
    cv_auc = cv_scores.mean()
    
    return {
        'AUC': auc,
        'CV_AUC': cv_auc,
        'Classification Report': report,
        'Coefficients': model.feature_importances_,
        'Feature Names': X.columns.tolist()
    } 