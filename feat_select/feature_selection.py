import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression

def select_features(X, y, df, n_features, method='permutation_importance'):
    """
    Selects relevant features from input data using various methods.

    Args:
        X (numpy.ndarray or pandas.DataFrame): Input features (samples x features).
        y (numpy.ndarray or pandas.Series): Target variable (samples x 1).
        df (pandas.DataFrame): DataFrame containing feature names.
        n_features (int): Number of features to select.
        method (str): Feature selection method ('permutation_importance', 'f_regression', or 'mutual_info_regression'). Defaults to 'permutation_importance'.

    Raises:
        ValueError: If an unknown feature selection method is provided.

    Returns:
        numpy.ndarray: Selected features (samples x n_features).
        pandas.Index: Names of selected features.
    """
    
    # Ensure X is a NumPy array for correct slicing
    if isinstance(X, pd.DataFrame):
        X = X.values
    
    if method == 'permutation_importance':
        model = LinearRegression()
        model.fit(X, y)
        perm_importance = permutation_importance(model, X, y, scoring=None)
        # Get indices of the top n_features
        sorted_idx = perm_importance.importances_mean.argsort()[-n_features:][::-1]
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
    feature_names = df.columns[sorted_idx]
    print(f"Selected features: {feature_names}")
    
    return X_selected, feature_names