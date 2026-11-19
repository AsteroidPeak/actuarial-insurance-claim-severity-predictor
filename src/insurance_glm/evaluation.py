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
    # Compute average predicted severity per policy (for policies with severity records)
    sev_per_policy = sev_val.groupby('IDpol').agg(
        pred_severity=('pred_severity', 'mean'),
        actual_claim_total=('ClaimAmount', 'sum'),
    ).reset_index()
    # Identify validation policies without any severity records
    freq_val_ids = set(freq_val['IDpol'])
    sev_policy_ids = set(sev_per_policy['IDpol'])
    missing_ids = freq_val_ids - sev_policy_ids
    if missing_ids:
        # Assign zero severity for policies with no severity records (no claim amount to predict)
        missing_df = pd.DataFrame({
            'IDpol': list(missing_ids),
            'pred_severity': 0.0,
            'actual_claim_total': 0.0,
        })
        sev_per_policy = pd.concat([sev_per_policy, missing_df], ignore_index=True)
    # Merge frequency and severity predictions on IDpol (left join to keep all freq policies)
    merged = freq_val[['IDpol', exposure_col, 'ClaimNb', 'pred_frequency']].merge(
        sev_per_policy, on='IDpol', how='left'
    )
    # Compute pure premium as predicted frequency * predicted severity
    merged['pure_premium'] = merged['pred_frequency'] * merged['pred_severity']
    # Ensure any missing values are filled (should not occur after missing handling)
    merged['pred_severity'].fillna(0.0, inplace=True)
    merged['actual_claim_total'].fillna(0.0, inplace=True)

    return merged

