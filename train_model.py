from os import name
from sklearn.preprocessing import StandardScaler, PowerTransformer
import pandas as pd
from sklearn.model_selection import train_test_split
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from mlflow.models import infer_signature
import joblib


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


def train():
    df = pd.read_csv('Salary_dataset.csv')
    df.drop('Unnamed: 0', axis=1, inplace=True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, test_size=0.2, random_state=42)
    (X_train, X_val, y_train, y_val) = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    params = {
    'n_estimators': [300, 400, 450],
    'max_depth': [1, 2, 3, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 3, 5],
    'max_features': ['sqrt', 'log2']
    }
    mlflow.set_experiment("randomforest model cars")
    with mlflow.start_run():
        rf = RandomForestRegressor(random_state=42)
        clf = GridSearchCV(rf, params, cv=3, n_jobs=-1, scoring='neg_root_mean_squared_error')
        clf.fit(X_train, y_train.ravel())
    
        best = clf.best_estimator_
        y_pred = best.predict(X_val)
    
        (rmse, mae, r2) = eval_metrics(y_val, y_pred)
    
        mlflow.log_param("n_estimators", best.n_estimators)
        mlflow.log_param("max_depth", best.max_depth)
        mlflow.log_param("max_features", best.max_features)
        mlflow.log_param("min_samples_leaf", best.min_samples_leaf)
        mlflow.log_param("min_samples_split", best.min_samples_split)
    
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)
    
        predictions = best.predict(X_train)
        signature = infer_signature(X_train, predictions)
        mlflow.sklearn.log_model(best, "random_forest_model", signature=signature)