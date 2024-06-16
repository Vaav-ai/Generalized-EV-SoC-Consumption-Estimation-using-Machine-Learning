import pandas as pd
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import optuna
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import VotingRegressor, BaggingRegressor
import xgboost as xgb

# Set random seed for reproducibility
random.seed(5)
np.random.seed(5)

def read_data(path):
    df = pd.read_csv(path)
    return df

def data_cleaning(df):
    '''
    Removal of NaN's, 0's and other non-essential data points
    '''
    df = df.replace(0, np.nan)
    df = df.dropna()
    return df

def preprocessing(df):
    df = data_cleaning(df)
    X = df.drop('SoC Consumed', axis=1)
    Y = df['SoC Consumed']
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = normalize(X, axis=0)
    return X, Y

def select_features(X, y, df, method, n_features):
    if method == 'permutation_importance':
        model = LinearRegression()
        model.fit(X, y)
        perm_importance = permutation_importance(model, X, y)
        sorted_idx = perm_importance.importances_mean.argsort()[-n_features:]
    elif method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=n_features)
        selector.fit(X, y)
        sorted_idx = selector.get_support(indices=True)
    elif method == 'mutual_info_regression':
        selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
        selector.fit(X, y)
        sorted_idx = selector.get_support(indices=True)
    else:
        raise ValueError(f"Unknown feature selection method: {method}")
    
    X_selected = X[:, sorted_idx]
    feature_names = df.drop('SoC Consumed', axis=1).columns[sorted_idx]
    print(f"Selected features: {feature_names}")
    
    return X_selected, feature_names

def objective(trial, X, y, model_name):
    if model_name == 'SVR':
        svr_c = trial.suggest_float('svr_c', 1e-3, 1e2, log=True)
        regressor_obj = SVR(C=svr_c, max_iter=10000)
    elif model_name == "LinearSVR":
        lsvr_c = trial.suggest_float('lsvr_c', 1e-3, 1e2, log=True)
        regressor_obj = LinearSVR(C=lsvr_c, max_iter=10000)
    elif model_name == "DecisionTree":
        dtr_max_depth = trial.suggest_int('dtr_max_depth', 2, 64)
        regressor_obj = DecisionTreeRegressor(max_depth=dtr_max_depth)
    elif model_name == "XGBoost":
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 2, 32)
        xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 10, 1000)
        xgb_eta = trial.suggest_float('xgb_eta', 0.01, 0.3)
        regressor_obj = xgb.XGBRegressor(max_depth=xgb_max_depth, n_estimators=xgb_n_estimators, eta=xgb_eta, use_label_encoder=False)
    elif model_name == "KNN":
        knn_k = trial.suggest_int('knn_k', 3, 15)
        regressor_obj = KNeighborsRegressor(weights="distance", n_neighbors=knn_k, n_jobs=8)
    elif model_name == "VotingKNN":
        knn3 = KNeighborsRegressor(n_neighbors=3)
        knn5 = KNeighborsRegressor(n_neighbors=5)
        knn7 = KNeighborsRegressor(n_neighbors=7)
        regressor_obj = VotingRegressor([('knn3', knn3), ('knn5', knn5), ('knn7', knn7)])
    elif model_name == "BaggedKNN":
        bag_knn = trial.suggest_int('bag_knn', 3, 15)
        regressor_obj = BaggingRegressor(KNeighborsRegressor(n_neighbors=bag_knn), n_estimators=10, n_jobs=8)
    elif model_name == "LinearRegression":
        regressor_obj = LinearRegression()
    else:
        raise ValueError(f"Unknown model: {model_name}")

    X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=0)

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_val)

    error = mean_squared_error(y_val, y_pred)

    return error

def main(config_path):
    # Default configuration
    default_config = {
        "model": "XGBoost",
        "n_trials": 100,
        "n_input_params": 5,
        "n_features": 5,
        "feature_selection_method": "permutation_importance",
        "selected_features_count": 2
    }

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Update default configuration with values from config file
    config = {**default_config, **config}

    # Read and preprocess data
    df = read_data(config['data_path'])
    X, y = preprocessing(df)
    
    # Select features
    X_selected, feature_names = select_features(X, y, df, config['feature_selection_method'], config['n_features'])
    
    # Limit to the number of selected features specified in the config
    X_selected = X_selected[:, :config['selected_features_count']]

    # Hyperparameter tuning
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_selected, y, config['model']), n_trials=config['n_trials'])

    # Train the best model
    best_trial = study.best_trial
    best_params = best_trial.params
    print("Best parameters:", best_params)

    if config['model'] == 'SVR':
        final_model = SVR(C=best_params['svr_c'], max_iter=10000)
    elif config['model'] == "LinearSVR":
        final_model = LinearSVR(C=best_params['lsvr_c'], max_iter=10000)
    elif config['model'] == "DecisionTree":
        final_model = DecisionTreeRegressor(max_depth=best_params['dtr_max_depth'])
    elif config['model'] == "XGBoost":
        final_model = xgb.XGBRegressor(max_depth=best_params['xgb_max_depth'], n_estimators=best_params['xgb_n_estimators'], eta=best_params['xgb_eta'], use_label_encoder=False)
    elif config['model'] == "KNN":
        final_model = KNeighborsRegressor(weights="distance", n_neighbors=best_params['knn_k'], n_jobs=8)
    elif config['model'] == "VotingKNN":
        knn3 = KNeighborsRegressor(n_neighbors=3)
        knn5 = KNeighborsRegressor(n_neighbors=5)
        knn7 = KNeighborsRegressor(n_neighbors=7)
        final_model = VotingRegressor([('knn3', knn3), ('knn5', knn5), ('knn7', knn7)])
    elif config['model'] == "BaggedKNN":
        final_model = BaggingRegressor(KNeighborsRegressor(n_neighbors=best_params['bag_knn']), n_estimators=10, n_jobs=8)
    elif config['model'] == "LinearRegression":
        final_model = LinearRegression()
    else:
        raise ValueError(f"Unknown model: {config['model']}")

    final_model.fit(X_selected, y)
    y_pred = final_model.predict(X_selected)

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y, y_pred, label='Actual vs Predicted', color='blue', s=10)  # Actual vs Predicted as scatter plot
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=3)  # 45-degree line
    plt.xlabel('Actual SoC Consumed')
    plt.ylabel('Predicted SoC Consumed')
    plt.title('Actual vs Predicted SoC Consumed')
    plt.legend()
    plt.show()

    # Plot residuals
    residuals = y - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, label='Residuals', color='green', s=10)
    plt.axhline(0, color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

if __name__ == "__main__":
    config_path = "D:/Research/Completed/EVresearch/config.json"
    main(config_path)