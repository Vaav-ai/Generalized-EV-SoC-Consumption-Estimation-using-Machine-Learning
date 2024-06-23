import pandas as pd
from sklearn.preprocessing import StandardScaler, normalize
from .clean import data_cleaning, remove_outliers
from sklearn.model_selection import train_test_split
from feat_select.feature_selection import select_features


def preprocessing(df:pd.DataFrame, test_size = 0.2, n_features = 10, use_StandardScaler=False, use_normalization=True, outlier_removal=True):
    """
    Preprocesses data by either normalizing or using Standard Scaler.

    Args:
        df (pd.DataFrame): Input DataFrame.
        test_size (float): Test size.
        n_features (int): Number of features to choose for feature selection.
        use_StandardScaler (bool, optional): Whether to use StandardScaler. Defaults to False.
        use_normalization (bool, optional): Whether to normalize. Defaults to True.
        outlier_removal (bool, optional): Whether to remove outlier. Defaults to True.

    Returns:
        pd.DataFrame: Processed features (X_train).
        pd.DataFrame: Processed features (X_test).
        pd.Series: Target variable (y_train).
        pd.Series: Target variable (y_test).
    """
    df = data_cleaning(df)
    
    X = df.drop('SoC Consumed', axis=1)
    y = df['SoC Consumed']
    
    X,_ = select_features(X,y,df,n_features)
    
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_size,random_state=42)
    
    if use_StandardScaler:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)
        
    if use_normalization:
        X_train = normalize(X_train, axis=0)
        X_test = normalize(X_test, axis=0)
        
    if outlier_removal:    
        X_train, y_train = remove_outliers(X_train, y_train)
        X_test, y_test = remove_outliers(X_test, y_test)
        
    return X_train, X_test, y_train, y_test