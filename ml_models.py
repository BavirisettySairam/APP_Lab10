from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, silhouette_score, roc_auc_score
from sklearn.model_selection import train_test_split
from data_processing import prepare_features

def perform_regression(df):
    """Perform regression analysis to predict ratings"""
    # Prepare features
    X = prepare_features(df)
    y = df['Rating']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Evaluate model
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    
    return {
        'RMSE': rmse,
        'Coefficients': model.coef_,
        'Intercept': model.intercept_
    }

def perform_clustering(df):
    """Perform clustering analysis for customer segmentation"""
    # Prepare features
    X = prepare_features(df)
    
    # Create and train model
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(X)
    
    # Get predictions
    labels = kmeans.labels_
    
    # Evaluate model
    silhouette = silhouette_score(X, labels)
    
    return {
        'Silhouette Score': silhouette,
        'Cluster Centers': kmeans.cluster_centers_
    }

def perform_classification(df):
    """Perform classification to predict recommendations"""
    # Prepare features
    X = prepare_features(df)
    y = df['Recommended_IND']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Evaluate model
    auc = roc_auc_score(y_test, y_pred_proba)
    
    return {
        'AUC': auc,
        'Coefficients': model.coef_,
        'Intercept': model.intercept_
    } 