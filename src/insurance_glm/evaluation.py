"""Evaluation utilities: pure premium computation."""
import pandas as pd
import numpy as np


def compute_pure_premium(freq_results, sev_results, freq_df, sev_df,
                         exposure_col='Exposure'):
    """Compute per-policy pure premium on validation data.

    Pure Premium = Annualized Frequency x Predicted Severity

    Annualized Frequency = Predicted Claim Count / Exposure

    Parameters
    ----------
    freq_results : dict returned by fit_frequency_glm
    sev_results : dict returned by fit_severity_glm
    freq_df : preprocessed frequency DataFrame
    sev_df : preprocessed severity DataFrame
    exposure_col : name of exposure column

    Returns
    -------
    DataFrame with IDpol, pred_frequency, pred_severity, pure_premium,
    actual_claim_amount
    """
    # --- frequency side ---
    freq_val = freq_df.iloc[freq_results['val_idx']].copy()
    freq_val['pred_count'] = np.asarray(freq_results['val_predictions'])
    freq_val['pred_frequency'] = freq_val['pred_count'] / freq_val[exposure_col]

    # --- severity side ---
    sev_val = sev_df.iloc[sev_results['val_idx']].copy()
    sev_val['pred_severity'] = np.asarray(sev_results['val_predictions'])

    # Average predicted severity per policy
    sev_per_policy = sev_val.groupby('IDpol').agg(
        pred_severity=('pred_severity', 'mean'),
        actual_claim_total=('ClaimAmount', 'sum'),
    ).reset_index()

    # Merge frequency and severity predictions on IDpol
    merged = freq_val[['IDpol', exposure_col, 'ClaimNb', 'pred_frequency']].merge(
        sev_per_policy, on='IDpol', how='inner'
    )
    merged['pure_premium'] = merged['pred_frequency'] * merged['pred_severity']

    return merged

