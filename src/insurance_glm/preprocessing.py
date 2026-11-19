import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


def clean_categorical(df: pd.DataFrame, cols):
    df = df.copy()
    for col in cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.add_categories('Missing')
        df[col] = df[col].fillna('Missing')
    return df


def log_transform_density(df: pd.DataFrame):
    df = df.copy()
    if 'Density' in df.columns:
        df['Density'] = df['Density'].apply(lambda x: np.log(x) if x > 0 else np.nan)
    return df


def preprocess_frequency(df: pd.DataFrame):
    df = df.copy()
    required = ['IDpol', 'Exposure', 'ClaimNb', 'VehBrand', 'Area', 'Region', 'VehGas', 'Density']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in frequency data: {missing}")

    df = df[df['Exposure'] > 0]

    cat_cols = ['VehBrand', 'Area', 'Region', 'VehGas']
    df = clean_categorical(df, cat_cols)
    df = log_transform_density(df)
    df = df.dropna()
    return df


def preprocess_severity(df: pd.DataFrame):
    df = df.copy()
    required = ['IDpol', 'ClaimAmount', 'VehBrand', 'Area', 'Region', 'VehGas', 'Density']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in severity data: {missing}")

    # only keep positive claim amounts -- zeros shouldn't be here but just in case
    df = df[df['ClaimAmount'] > 0]

    cat_cols = ['VehBrand', 'Area', 'Region', 'VehGas']
    df = clean_categorical(df, cat_cols)
    df = log_transform_density(df)
    df = df.dropna()
    return df


def build_preprocessor(df: pd.DataFrame):
    cat_cols = ['VehBrand', 'Area', 'Region', 'VehGas']
    # drop='first' to avoid the dummy variable trap with the intercept
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', drop='first', sparse_output=False), cat_cols)
        ],
        remainder='passthrough'
    )
    return preprocessor
