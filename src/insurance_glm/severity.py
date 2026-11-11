"""Gamma GLM for claim severity prediction."""
import pandas as pd
import numpy as np
import statsmodels.api as sm


def fit_severity_glm(df, preprocessor, train_idx, val_idx):
    """Fit Gamma GLM for claim severity.

    Parameters
    ----------
    df : DataFrame with columns IDpol, ClaimAmount, VehBrand, Area, Region, VehGas, Density
    preprocessor : fitted ColumnTransformer (will be refit on training data)
    train_idx : array of integer indices for training rows
    val_idx : array of integer indices for validation rows

    Returns
    -------
    dict with model, null_model, val_predictions, val_null_predictions, metrics,
         preprocessor, train_idx, val_idx
    """
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    drop_cols = ['IDpol', 'ClaimAmount']
    # Also drop Exposure, ClaimNb, set if present
    for c in ['Exposure', 'ClaimNb', 'set']:
        if c in train_df.columns:
            drop_cols.append(c)

    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['ClaimAmount'].values.astype(float)
    X_val = val_df.drop(columns=drop_cols, errors='ignore')
    y_val = val_df['ClaimAmount'].values.astype(float)

    # Fit preprocessor on training data
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc = preprocessor.transform(X_val)

    # Convert sparse to dense if needed
    if hasattr(X_train_enc, 'toarray'):
        X_train_enc = X_train_enc.toarray()
    if hasattr(X_val_enc, 'toarray'):
        X_val_enc = X_val_enc.toarray()

    # Add constant column for intercept
    X_train_enc = sm.add_constant(X_train_enc, has_constant='add')
    X_val_enc = sm.add_constant(X_val_enc, has_constant='add')

    # Gamma GLM with log link
    gamma_model = sm.GLM(y_train, X_train_enc,
                         family=sm.families.Gamma(link=sm.families.links.Log()))
    gamma_results = gamma_model.fit()

    # Null model (intercept only)
    null_X_train = np.ones((len(y_train), 1))
    null_X_val = np.ones((len(y_val), 1))
    null_model = sm.GLM(y_train, null_X_train,
                        family=sm.families.Gamma(link=sm.families.links.Log()))
    null_results = null_model.fit()

    # Predictions on validation
    pred_sev = gamma_results.predict(X_val_enc)
    null_pred = null_results.predict(null_X_val)

    def gamma_deviance(y, mu):
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        return 2.0 * np.sum(-np.log(y / mu) + (y - mu) / mu)

    dev_full = gamma_deviance(y_val, pred_sev)
    dev_null = gamma_deviance(y_val, null_pred)
    perc_reduction = (dev_null - dev_full) / dev_null * 100 if dev_null != 0 else float('nan')

    converged = gamma_results.converged if hasattr(gamma_results, 'converged') else True

    metrics = {
        'deviance_full': float(dev_full),
        'deviance_null': float(dev_null),
        'percent_deviance_reduction': float(perc_reduction),
        'converged': bool(converged),
    }

    return {
        'model': gamma_results,
        'null_model': null_results,
        'val_predictions': np.asarray(pred_sev),
        'val_null_predictions': np.asarray(null_pred),
        'metrics': metrics,
        'preprocessor': preprocessor,
        'train_idx': train_idx,
        'val_idx': val_idx,
    }

