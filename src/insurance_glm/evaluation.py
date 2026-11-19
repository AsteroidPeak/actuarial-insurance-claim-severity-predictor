import pandas as pd
import numpy as np
import statsmodels.api as sm


def compute_pure_premium(freq_results, sev_results, freq_df, sev_df,
                         exposure_col='Exposure',
                         freq_raw=None, train_ids=None, val_ids=None):
    # pull out the validation frequency rows and attach predictions
    freq_val = freq_df.iloc[freq_results['val_idx']].copy()
    freq_val['pred_count'] = np.asarray(freq_results['val_predictions'])
    # annualised rate = count / exposure
    freq_val['pred_frequency'] = freq_val['pred_count'] / freq_val[exposure_col]

    drop_cols_sev = ['IDpol', 'ClaimAmount', 'Exposure', 'ClaimNb', 'set']

    gamma_model = sev_results.get('model')
    sev_preprocessor = sev_results.get('preprocessor')

    if gamma_model is not None and sev_preprocessor is not None \
            and freq_raw is not None and val_ids is not None:
        # predict severity for every validation policy, not just those with claims
        from .preprocessing import clean_categorical, log_transform_density
        val_freq_raw = freq_raw[freq_raw['IDpol'].isin(val_ids)].copy()
        cat_cols = ['VehBrand', 'Area', 'Region', 'VehGas']
        val_freq_raw = clean_categorical(val_freq_raw, cat_cols)
        val_freq_raw = log_transform_density(val_freq_raw)
        val_freq_raw = val_freq_raw.dropna()
        feature_cols = [c for c in val_freq_raw.columns
                        if c not in drop_cols_sev + ['Exposure', 'ClaimNb']]
        X_all_val = val_freq_raw[feature_cols].copy()
        idpol_all = val_freq_raw['IDpol'].values

        X_all_enc = sev_preprocessor.transform(X_all_val)
        if hasattr(X_all_enc, 'toarray'):
            X_all_enc = X_all_enc.toarray()
        X_all_enc = sm.add_constant(X_all_enc, has_constant='add')
        pred_sev_all = np.asarray(gamma_model.predict(X_all_enc))

        sev_pred_df = pd.DataFrame({
            'IDpol': idpol_all,
            'pred_severity': pred_sev_all,
        }).groupby('IDpol', as_index=False)['pred_severity'].mean()

    else:
        # fallback for unit tests that don't pass a live model
        sev_val = sev_df.iloc[sev_results['val_idx']].copy()
        sev_val['pred_severity'] = np.asarray(sev_results['val_predictions'])
        sev_pred_df = sev_val.groupby('IDpol', as_index=False)['pred_severity'].mean()

    # sum up actual claim amounts per policy across the whole severity dataset
    actual_by_policy = (
        sev_df.groupby('IDpol', as_index=False)['ClaimAmount']
        .sum()
        .rename(columns={'ClaimAmount': 'actual_claim_total'})
    )

    # left join so every frequency-validation policy is kept
    # policies with no claim records get actual_claim_total = 0
    merged = freq_val[
        ['IDpol', exposure_col, 'ClaimNb', 'pred_count', 'pred_frequency']
    ].merge(sev_pred_df, on='IDpol', how='left')
    merged = merged.merge(actual_by_policy, on='IDpol', how='left')

    merged['pred_severity'] = merged['pred_severity'].fillna(0.0)
    merged['actual_claim_total'] = merged['actual_claim_total'].fillna(0.0)

    # pred_count already accounts for exposure via the log offset
    # so loss = count * severity, no need to multiply by exposure again
    merged['pred_loss'] = merged['pred_count'] * merged['pred_severity']
    merged['pure_premium'] = merged['pred_frequency'] * merged['pred_severity']

    return merged
