import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor

def data_cleaning(df:pd.DataFrame, remove_0:bool=True):
    '''
    Removal of NaN's and 0's from the dataset.
    
    Args:
        df (pd.DataFrame) : Input Dataframe
        remove_0 (bool, optional) : Whether you want to remove 0 or not. True by default.
        
    Ruturns:
        df (pd.DataFrame) : Dataset with 0 and NaN values removed
    '''
    
    df = df.replace(0, np.nan)
    df = df.dropna()
    
    return df

def remove_outliers(X,y):
    """
    Removal of Outliers using `LocalOutlierFactor` from sklearn.neighbors

    Args:
        X (pd.DataFrame): Independent Variables.
        y (pd.Series): Target.
        
    Returns:
        pd.DataFrame: Indepenent Variables with the outliers removed.
        pd.Series: Target with the outliers removed.
    """
    
    old_size = X.shape[0]
    lof = LocalOutlierFactor()
    yhat = lof.fit_predict(X)
    # select all rows that are not outliers
    mask = yhat != -1
    X, y = X[mask, :], y[mask]
    print("Data points removed: ", old_size-X.shape[0])
    
    return X, y