import pandas as pd
import numpy as np
import random
import json
import matplotlib.pyplot as plt
import optuna
import pickle
from data_processing.preprocessing import preprocessing
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor, AdaBoostRegressor, BaggingRegressor, VotingRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import ExtraTreeRegressor, DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
from sklearn.metrics import mean_squared_error

# Set random seed for reproducibility
random.seed(5)
np.random.seed(5)

def read_data(path):
    df = pd.read_csv(path)
    return df

def objective(trial, X_train, X_test, y_train, y_test, model_names):
    
    if len(model_names[0]) == 1 and model_names != "all":
        regressor_name = trial.suggest_categorical('model_name',[model_names])
    elif model_names == "all":
        regressor_name = trial.suggest_categorical('model_name',[
        'SVR', 'RandomForest', 'KNN', 'ExtraTree', 'DecisionTree',
        'LinearRegression', 'Ridge', 'Lasso', 'ElasticNet',
        'GradientBoosting', 'AdaBoost', 'BaggedKNN', 'MLPRegressor',
        'GaussianProcess', 'XGBoost', 'LightGBM', 'LinearSVR', 'VotingKNN'
        ])
    else:
        regressor_name = trial.suggest_categorical('model_name',model_names)
    
    if regressor_name == 'SVR':
        svr_c = trial.suggest_float('svr_c', 1e-10, 1e10, log=True)
        svr_kernel = trial.suggest_categorical('svr_kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
        regressor_obj = SVR(C=svr_c, kernel=svr_kernel, max_iter=10000)
    elif regressor_name == "LinearSVR":
        lsvr_c = trial.suggest_float('lsvr_c', 1e-3, 1e2, log=True)
        regressor_obj = LinearSVR(C=lsvr_c, max_iter=10000)
    elif regressor_name == "KNN":
        knn_k = trial.suggest_int('knn_k', 3, 15)
        knn_weights = trial.suggest_categorical('knn_weights',['uniform','distance'])
        knn_algo = trial.suggest_categorical('knn_algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
        regressor_obj = KNeighborsRegressor(weights=knn_weights, algorithm=knn_algo, n_neighbors=knn_k, n_jobs=8)
    elif regressor_name == "ExtraTree":
        etr_max_depth = trial.suggest_int('etr_max_depth', 2, 64)
        regressor_obj = ExtraTreeRegressor(max_depth=etr_max_depth)
    elif regressor_name == "DecisionTree":
        dtr_crit = trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error'])
        dtr_m_d = trial.suggest_int('dtr_max_depth', 2, 64)
        dtr_m_l = trial.suggest_int('min_samples_leaf',1,32)
        regressor_obj = DecisionTreeRegressor(criterion=dtr_crit, max_depth=dtr_m_d, min_samples_leaf=dtr_m_l)
    elif regressor_name == "RandomForest":
        rf_max_depth = trial.suggest_int('rf_max_depth', 2, 32)
        rf_n_estimators = trial.suggest_int('rf_n_estimators', 10, 1000)
        regressor_obj = RandomForestRegressor(max_depth=rf_max_depth, n_estimators=rf_n_estimators)
    elif regressor_name == "LinearRegression":
        regressor_obj = LinearRegression()
    elif regressor_name == "Ridge":
        ridge_alpha = trial.suggest_float('ridge_alpha', 1e-10, 1e2, log=True)
        regressor_obj = Ridge(alpha=ridge_alpha)
    elif regressor_name == "Lasso":
        lasso_alpha = trial.suggest_float('lasso_alpha', 1e-10, 1e2, log=True)
        regressor_obj = Lasso(alpha=lasso_alpha)
    elif regressor_name == "ElasticNet":
        en_alpha = trial.suggest_float('en_alpha', 1e-10, 1e2, log=True)
        en_l1_ratio = trial.suggest_float('en_l1_ratio', 0, 1)
        regressor_obj = ElasticNet(alpha=en_alpha, l1_ratio=en_l1_ratio)
    elif regressor_name == "GradientBoosting":
        gb_max_depth = trial.suggest_int('gb_max_depth', 2, 32)
        gb_n_estimators = trial.suggest_int('gb_n_estimators', 10, 1000)
        regressor_obj = GradientBoostingRegressor(max_depth=gb_max_depth, n_estimators=gb_n_estimators)
    elif regressor_name == "AdaBoost":
        ab_n_estimators = trial.suggest_int('ab_n_estimators', 10, 1000)
        ab_lr = trial.suggest_float('ab_lr',1e-4,1e4,log=True)
        regressor_obj = AdaBoostRegressor(n_estimators=ab_n_estimators,learning_rate=ab_lr)
    elif regressor_name == "BaggedKNN":
        knn_k = trial.suggest_int('knn_k', 3, 15)
        knn_n = trial.suggest_int('knn_n', 1, 100)
        knn_weights = trial.suggest_categorical('knn_weights',['uniform','distance'])
        knn_algo = trial.suggest_categorical('knn_algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
        regressor_obj = BaggingRegressor(estimator=KNeighborsRegressor(weights=knn_weights,algorithm=knn_algo, n_neighbors=knn_k, n_jobs = 8),n_estimators=knn_n)
    elif regressor_name == "MLPRegressor":
        mlp_hidden_layer_sizes = trial.suggest_int('mlp_hidden_layer_sizes', 10, 200)
        mlp_alpha = trial.suggest_float('mlp_alpha', 1e-10, 1e2, log=True)
        regressor_obj = MLPRegressor(hidden_layer_sizes=(mlp_hidden_layer_sizes,), alpha=mlp_alpha, max_iter=10000)
    elif regressor_name == "GaussianProcess":
        regressor_obj = GaussianProcessRegressor()
    elif regressor_name == "VotingKNN":
        knn_weights = trial.suggest_categorical('knn_weights',['uniform','distance'])
        knn_algo = trial.suggest_categorical('knn_algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
        regressor_obj = VotingRegressor(estimators=[("kn3",KNeighborsRegressor(weights=knn_weights, algorithm=knn_algo, n_neighbors=3)),
                                                    ("kn5",KNeighborsRegressor(weights=knn_weights, algorithm=knn_algo, n_neighbors=5)),
                                                    ("kn7",KNeighborsRegressor(weights=knn_weights, algorithm=knn_algo, n_neighbors=7))],n_jobs=8)
    elif regressor_name == "XGBoost":
        xgb_max_depth = trial.suggest_int('xgb_max_depth', 2, 32)
        xgb_n_estimators = trial.suggest_int('xgb_n_estimators', 10, 1000)
        xgb_eta = trial.suggest_float('xgb_eta', 0.01, 0.3)
        regressor_obj = xgb.XGBRegressor(max_depth=xgb_max_depth, n_estimators=xgb_n_estimators, eta=xgb_eta, use_label_encoder=False)
    elif regressor_name == "LightGBM":
        lgb_max_depth = trial.suggest_int('lgb_max_depth', 2, 32)
        lgb_n_estimators = trial.suggest_int('lgb_n_estimators', 10, 1000)
        lgb_learning_rate = trial.suggest_float('lgb_learning_rate', 0.01, 0.3)
        regressor_obj = lgb.LGBMRegressor(max_depth=lgb_max_depth, n_estimators=lgb_n_estimators, learning_rate=lgb_learning_rate)
    else:
        raise ValueError(f"Unknown regressor: {regressor_name}")    

    regressor_obj.fit(X_train, y_train)
    y_pred = regressor_obj.predict(X_test)

    error = mean_squared_error(y_test, y_pred)

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
    X_train, X_test, y_train, y_test = preprocessing(df, n_features=config['selected_features_count'])

    # Hyperparameter tuning with Optuna
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, X_train, X_test, y_train, y_test, config['model']), n_trials=config['n_trials'], show_progress_bar=True)

    # Get the best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    print("Best parameters:", best_params)

    # Instantiate the best model with the best hyperparameters
    if best_params['model_name'] == 'SVR':
        final_model = SVR(C=best_params['svr_c'], kernel=best_params['svr_kernel'], max_iter=10000)
    elif best_params['model_name'] == "LinearSVR":
        final_model = LinearSVR(C=best_params['lsvr_c'], max_iter=10000)
    elif best_params['model_name'] == "KNN":
        final_model = KNeighborsRegressor(weights=best_params['knn_weights'], algorithm=best_params['knn_algorithm'], n_neighbors=best_params['knn_k'], n_jobs=8)
    elif best_params['model_name'] == "ExtraTree":
        final_model = ExtraTreeRegressor(max_depth=best_params['etr_max_depth'])
    elif best_params['model_name'] == "DecisionTree":
        final_model = DecisionTreeRegressor(criterion=best_params['criterion'], max_depth=best_params['dtr_max_depth'], min_samples_leaf=best_params['min_samples_leaf'])
    elif best_params['model_name'] == "RandomForest":
        final_model = RandomForestRegressor(max_depth=best_params['rf_max_depth'], n_estimators=best_params['rf_n_estimators'])
    elif best_params['model_name'] == "LinearRegression":
        final_model = LinearRegression()
    elif best_params['model_name'] == "Ridge":
        final_model = Ridge(alpha=best_params['ridge_alpha'])
    elif best_params['model_name'] == "Lasso":
        final_model = Lasso(alpha=best_params['lasso_alpha'])
    elif best_params['model_name'] == "ElasticNet":
        final_model = ElasticNet(alpha=best_params['en_alpha'], l1_ratio=best_params['en_l1_ratio'])
    elif best_params['model_name'] == "GradientBoosting":
        final_model = GradientBoostingRegressor(max_depth=best_params['gb_max_depth'], n_estimators=best_params['gb_n_estimators'])
    elif best_params['model_name'] == "AdaBoost":
        final_model = AdaBoostRegressor(n_estimators=best_params['ab_n_estimators'], learning_rate=best_params['ab_lr'])
    elif best_params['model_name'] == "BaggedKNN":
        final_model = BaggingRegressor(KNeighborsRegressor(weights=best_params['knn_weights'], algorithm=best_params['knn_algorithm'], n_neighbors=best_params['knn_k'], n_jobs=8), n_estimators=best_params['knn_n'])
    elif best_params['model_name'] == "MLPRegressor":
        final_model = MLPRegressor(hidden_layer_sizes=(best_params['mlp_hidden_layer_sizes'],), alpha=best_params['mlp_alpha'], max_iter=10000)
    elif best_params['model_name'] == "GaussianProcess":
        final_model = GaussianProcessRegressor()
    elif best_params['model_name'] == "VotingKNN":
        final_model = VotingRegressor(estimators=[
            ("kn3", KNeighborsRegressor(weights=best_params['knn_weights'], algorithm=best_params['knn_algorithm'], n_neighbors=3)),
            ("kn5", KNeighborsRegressor(weights=best_params['knn_weights'], algorithm=best_params['knn_algorithm'], n_neighbors=5)),
            ("kn7", KNeighborsRegressor(weights=best_params['knn_weights'], algorithm=best_params['knn_algorithm'], n_neighbors=7))
        ], n_jobs=8)
    elif best_params['model_name'] == "XGBoost":
        final_model = xgb.XGBRegressor(max_depth=best_params['xgb_max_depth'], n_estimators=best_params['xgb_n_estimators'], eta=best_params['xgb_eta'], use_label_encoder=False)
    elif best_params['model_name'] == "LightGBM":
        final_model = lgb.LGBMRegressor(max_depth=best_params['lgb_max_depth'], n_estimators=best_params['lgb_n_estimators'], learning_rate=best_params['lgb_learning_rate'])
    else:
        raise ValueError(f"Unknown model: {best_params['model_name']}")

    # Train the final model
    final_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = final_model.predict(X_test)
    
    # Evaluate the model
    error = mean_squared_error(y_test, y_pred)
    print(f"Mean Squared Error on the test set: {error}")

    with open('trained_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)

    model_save_path = config['model_path'] + "trained_model.bin"
    joblib.dump(final_model, model_save_path)
    print(f"Model saved to {model_save_path}")

    # Plot actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(y_test)), y_test, label='Actual SoC Consumed', color='blue', s=10)  # Actual values as scatter plot
    plt.plot(y_pred, label='Predicted SoC Consumed', color='red', alpha=0.7)  # Predicted values as line plot
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('SoC Consumed')
    plt.title('Actual vs Predicted SoC Consumed')
    plt.show()

    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(residuals)), residuals, label='Residuals', color='green', s=10)
    plt.axhline(0, color='red', linestyle='--')
    plt.legend()
    plt.xlabel('Samples')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.show()

if __name__ == "__main__":
    config_path = r"C:\Users\get2b\Desktop\Arav\Research Papers\VIT Ather project\Code pushed to github\Generalized-EV-SoC-Consumption-Estimation-using-Machine-Learning\sample_json\config.json"  # Path to your config file
    main(config_path)
