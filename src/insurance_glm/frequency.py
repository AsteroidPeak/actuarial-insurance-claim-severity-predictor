"""Poisson GLM for claim frequency prediction."""
import pandas as pd
import numpy as np
import statsmodels.api as sm


def fit_frequency_glm(df, preprocessor, train_idx, val_idx):
    """Fit Poisson GLM for claim frequency with log(Exposure) offset.

    Parameters
    ----------
    df : DataFrame with columns IDpol, Exposure, ClaimNb, VehBrand, Area, Region, VehGas, Density
    preprocessor : ColumnTransformer (will be fit on training data)
    train_idx : array of integer indices for training rows
    val_idx : array of integer indices for validation rows

    Returns
    -------
    dict with model, null_model, val_predictions, val_null_predictions, metrics,
         preprocessor, train_idx, val_idx
    """
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    drop_cols = ['IDpol', 'ClaimNb', 'Exposure']
    # Also drop 'set' column if present
    if 'set' in train_df.columns:
        drop_cols.append('set')

    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df['ClaimNb'].values.astype(float)
    exposure_train = train_df['Exposure'].values.astype(float)

    X_val = val_df.drop(columns=drop_cols)
    y_val = val_df['ClaimNb'].values.astype(float)
    exposure_val = val_df['Exposure'].values.astype(float)

    # Fit preprocessor on training data only
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

    offset_train = np.log(exposure_train)
    offset_val = np.log(exposure_val)

    # Fit Poisson GLM
    poisson_model = sm.GLM(y_train, X_train_enc,
                           family=sm.families.Poisson(),
                           offset=offset_train)
    poisson_results = poisson_model.fit()

    # Null model (intercept only) with same offset
    null_X_train = np.ones((len(y_train), 1))
    null_X_val = np.ones((len(y_val), 1))
    null_model = sm.GLM(y_train, null_X_train,
                        family=sm.families.Poisson(),
                        offset=offset_train)
    null_results = null_model.fit()

    # Predictions on validation
    pred_counts = poisson_results.predict(X_val_enc, offset=offset_val)
    null_pred = null_results.predict(null_X_val, offset=offset_val)

    def poisson_deviance(y, mu):
        y = np.asarray(y, dtype=float)
        mu = np.asarray(mu, dtype=float)
        mu = np.maximum(mu, 1e-12)  # avoid division by zero
        term = np.where(y == 0, 0.0, y * np.log(np.maximum(y, 1e-12) / mu))
        return 2.0 * np.sum(term - (y - mu))

    dev_full = poisson_deviance(y_val, pred_counts)
    dev_null = poisson_deviance(y_val, null_pred)
    perc_reduction = (dev_null - dev_full) / dev_null * 100 if dev_null != 0 else float('nan')

    converged = poisson_results.converged if hasattr(poisson_results, 'converged') else True

    metrics = {
        'deviance_full': float(dev_full),
        'deviance_null': float(dev_null),
        'percent_deviance_reduction': float(perc_reduction),
        'converged': bool(converged),
    }

    return {
        'model': poisson_results,
        'null_model': null_results,
        'val_predictions': np.asarray(pred_counts),
        'val_null_predictions': np.asarray(null_pred),
        'metrics': metrics,
        'preprocessor': preprocessor,
        'train_idx': train_idx,
        'val_idx': val_idx,
    }
