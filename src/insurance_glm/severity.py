import numpy as np
import pandas as pd
import statsmodels.api as sm


def fit_severity_glm(df, preprocessor, train_idx, val_idx):
    train_df = df.iloc[train_idx].copy()
    val_df   = df.iloc[val_idx].copy()

    drop_cols = ['IDpol', 'ClaimAmount']
    for c in ['Exposure', 'ClaimNb', 'set']:
        if c in train_df.columns:
            drop_cols.append(c)

    X_train = train_df.drop(columns=drop_cols, errors='ignore')
    y_train = train_df['ClaimAmount'].values.astype(float)
    X_val   = val_df.drop(columns=drop_cols,   errors='ignore')
    y_val   = val_df['ClaimAmount'].values.astype(float)

    # fit only on training data
    X_train_enc = preprocessor.fit_transform(X_train)
    X_val_enc   = preprocessor.transform(X_val)

    if hasattr(X_train_enc, 'toarray'):
        X_train_enc = X_train_enc.toarray()
    if hasattr(X_val_enc, 'toarray'):
        X_val_enc = X_val_enc.toarray()

    X_train_enc = sm.add_constant(X_train_enc, has_constant='add')
    X_val_enc   = sm.add_constant(X_val_enc,   has_constant='add')

    # Gamma with log link keeps predictions positive and handles skewed claims
    gamma_model   = sm.GLM(y_train, X_train_enc,
                           family=sm.families.Gamma(link=sm.families.links.Log()))
    gamma_results = gamma_model.fit()

    # intercept-only null for deviance comparison
    null_X_train = np.ones((len(y_train), 1))
    null_X_val   = np.ones((len(y_val),   1))
    null_model   = sm.GLM(y_train, null_X_train,
                          family=sm.families.Gamma(link=sm.families.links.Log()))
    null_results = null_model.fit()

    pred_sev  = gamma_results.predict(X_val_enc)
    null_pred = null_results.predict(null_X_val)

    def gamma_deviance(y, mu):
        y  = np.asarray(y,  dtype=float)
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
