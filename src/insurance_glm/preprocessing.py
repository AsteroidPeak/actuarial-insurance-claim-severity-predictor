import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

def clean_categorical(df: pd.DataFrame, cols):
    """Convert categorical columns to appropriate type and handle missing values.
    Missing values are filled with a placeholder string 'Missing'.
    """
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.add_categories('Missing')
        df[col] = df[col].fillna('Missing')
    return df

def log_transform_density(df: pd.DataFrame):
    df = df.copy()
    if 'Density' in df.columns:
        # avoid non-positive values
        df['Density'] = df['Density'].apply(lambda x: np.log(x) if x > 0 else np.nan)
    return df

def preprocess_frequency(df: pd.DataFrame):
    """Preprocess the frequency dataset.
    Returns the processed DataFrame and the ColumnTransformer for later use.
    """
    df = df.copy()
    # Verify required columns
    required = ['IDpol', 'Exposure', 'ClaimNb', 'VehBrand', 'Area', 'Region', 'VehGas', 'Density']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in frequency data: {missing}")
    # Remove records with zero exposure
    df = df[df['Exposure'] > 0]
    # Categorical handling
    cat_cols = ['VehBrand', 'Area', 'Region', 'VehGas']
    df = clean_categorical(df, cat_cols)
    # Log transform density
    df = log_transform_density(df)
    # Drop rows with any remaining NaNs (e.g., non-positive density)
    df = df.dropna()
    return df

def preprocess_severity(df: pd.DataFrame):
    """Preprocess the severity dataset.
    Keep only ClaimAmount > 0 and handle categoricals.
    """
    df = df.copy()
    required = ['IDpol', 'ClaimAmount', 'VehBrand', 'Area', 'Region', 'VehGas', 'Density']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in severity data: {missing}")
    df = df[df['ClaimAmount'] > 0]
    cat_cols = ['VehBrand', 'Area', 'Region', 'VehGas']
    df = clean_categorical(df, cat_cols)
    df = log_transform_density(df)
    df = df.dropna()
    return df

def build_preprocessor(df: pd.DataFrame):
    """Build a ColumnTransformer that one-hot encodes categorical columns.
    The transformer expects the same columns as used in preprocessing.
    """
    cat_cols = ['VehBrand', 'Area', 'Region', 'VehGas']
    # Use pandas category dtype, OneHotEncoder will handle unknowns.
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
        ], remainder='passthrough'
    )
    return preprocessor

