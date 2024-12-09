# src/data_preprocessing.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def load_data(file_path):
    """
    Load the dataset from a CSV file.

    Parameters:
        file_path (str or Path): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    data = pd.read_csv(file_path)
    return data

def clean_column_names(data):
    """
    Clean and standardize column names.

    Parameters:
        data (pd.DataFrame): The dataset with raw column names.

    Returns:
        pd.DataFrame: Dataset with cleaned column names.
    """
    data.columns = data.columns.str.strip().str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_').str.lower()
    return data

def handle_missing_values(data):
    """
    Handle missing values in the dataset.

    Parameters:
        data (pd.DataFrame): The dataset with potential missing values.

    Returns:
        pd.DataFrame: Dataset with missing values handled.
    """
    missing_indicators = ['NA', '...', '', 'nan', 'NaN', 'nan']
    data.replace(missing_indicators, np.nan, inplace=True)
    
    missing_summary = data.isnull().sum()
    print("Missing Values Summary:\n", missing_summary[missing_summary > 0])
    
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    for col in categorical_cols:
        if data[col].isnull().sum() > 0:
            mode = data[col].mode()[0]
            data[col].fillna(mode, inplace=True)
            print(f"Filled missing values in '{col}' with mode: {mode}")
    
    for col in numerical_cols:
        if data[col].isnull().sum() > 0:
            median = data[col].median()
            data[col].fillna(median, inplace=True)
            print(f"Filled missing values in '{col}' with median: {median}")
    
    return data

def preprocess_data(data, target_column='class'):
    """
    Preprocess the dataset by cleaning, encoding, scaling, and feature selection.

    Parameters:
        data (pd.DataFrame): Raw dataset.
        target_column (str): Name of the target variable.

    Returns:
        np.ndarray: Processed feature matrix.
        np.ndarray: Encoded target vector.
        ColumnTransformer: Preprocessing pipeline for future transformations.
        np.ndarray: Names of the selected features.
    """
    data = clean_column_names(data)
    data = handle_missing_values(data)
    
    X = data.drop(target_column, axis=1)
    y = data[target_column]
    
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Split combined columns if necessary
    if 'personal_status_and_sex' in categorical_cols:
        X[['personal_status', 'sex']] = X['personal_status_and_sex'].str.split(':', expand=True)
        X.drop('personal_status_and_sex', axis=1, inplace=True)
        categorical_cols.remove('personal_status_and_sex')
        categorical_cols.extend(['personal_status', 'sex'])
        print("Split 'personal_status_and_sex' into 'personal_status' and 'sex'")
    
    # Define preprocessing steps
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Apply preprocessing
    X_processed = preprocessor.fit_transform(X)
    
    # Feature Selection
    selector = SelectKBest(score_func=mutual_info_classif, k=30)
    X_selected = selector.fit_transform(X_processed, y)
    
    selected_feature_indices = selector.get_support(indices=True)
    feature_names = preprocessor.get_feature_names_out()
    selected_feature_names = feature_names[selected_feature_indices]
    print(f"\nSelected Top 30 Features:\n{selected_feature_names}")
    
    # Encode target variable
    y_encoded = y.map({'Good': 1, 'Bad': 0}).values
    
    return X_selected, y_encoded, preprocessor, selected_feature_names


def split_data(X, y, test_size=0.3, random_state=42):
    """
    Split the data into training, validation, and testing sets with stratification.

    Parameters:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        test_size (float): Proportion of data to allocate to validation and test sets.
        random_state (int): Seed for reproducibility.

    Returns:
        np.ndarray: Training feature matrix.
        np.ndarray: Validation feature matrix.
        np.ndarray: Testing feature matrix.
        np.ndarray: Training target vector.
        np.ndarray: Validation target vector.
        np.ndarray: Testing target vector.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
