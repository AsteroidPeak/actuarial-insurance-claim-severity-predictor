import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .data import download_datasets, load_frequency, load_severity
from .preprocessing import preprocess_frequency, preprocess_severity, build_preprocessor
from .frequency import fit_frequency_glm
from .severity import fit_severity_glm
from .evaluation import compute_pure_premium


def run_pipeline(base_dir=None):
    if base_dir is None:
        base_dir = Path('.')
    base_dir = Path(base_dir)

    raw_dir = base_dir / 'data' / 'raw'
    freq_path = raw_dir / 'freMTPL2freq.csv'
    sev_path = raw_dir / 'freMTPL2sev.csv'

    if not freq_path.is_file() or not sev_path.is_file():
        print("[pipeline] Downloading datasets ...")
        freq_path, sev_path = download_datasets(raw_dir)
    else:
        print("[pipeline] Raw data already present, skipping download.")

    print("[pipeline] Loading raw data ...")
    freq_raw = load_frequency(freq_path)
    sev_raw = load_severity(sev_path)
    print(f"  Frequency shape: {freq_raw.shape}")
    print(f"  Severity  shape: {sev_raw.shape}")

    # basic sanity checks before doing anything
    print("[pipeline] Validating data ...")
    if not freq_raw['IDpol'].is_unique:
        print("  WARNING: IDpol is NOT unique in frequency data!")
    else:
        print("  IDpol is unique in frequency data.")

    zero_exp = (freq_raw['Exposure'] <= 0).sum()
    if zero_exp > 0:
        print(f"  WARNING: {zero_exp} records have Exposure <= 0 (will be removed).")
    else:
        print("  All Exposure values are positive.")

    missing_freq = freq_raw.isnull().sum()
    missing_sev = sev_raw.isnull().sum()
    if missing_freq.sum() > 0:
        print(f"  Missing values in frequency data:\n{missing_freq[missing_freq > 0]}")
    else:
        print("  No missing values in frequency data.")
    if missing_sev.sum() > 0:
        print(f"  Missing values in severity data:\n{missing_sev[missing_sev > 0]}")
    else:
        print("  No missing values in severity data.")

    # the two datasets don't perfectly match -- reporting the gap rather than hiding it
    total_claimnb = freq_raw['ClaimNb'].sum()
    num_sev_records = len(sev_raw)
    print(f"  Total ClaimNb in freq data: {total_claimnb}")
    print(f"  Number of severity records:  {num_sev_records}")
    if total_claimnb != num_sev_records:
        print(f"  DISCREPANCY: sum(ClaimNb)={total_claimnb} != severity records={num_sev_records}")

    print("[pipeline] Preprocessing ...")
    freq = preprocess_frequency(freq_raw)
    print(f"  Frequency after preprocessing: {freq.shape}")

    # attach vehicle/driver features to each claim row so the severity model can use them
    # drop ClaimNb and Exposure so they don't accidentally end up as features
    policy_features = freq_raw.drop(columns=['ClaimNb', 'Exposure'], errors='ignore')
    sev_joined = sev_raw.merge(policy_features, on='IDpol', how='left')
    sev = preprocess_severity(sev_joined)

    # reset so iloc indexing works correctly later
    freq = freq.reset_index(drop=True)
    sev = sev.reset_index(drop=True)
    print(f"  Severity after preprocessing:  {sev.shape}")

    # split on policy IDs so the same policy never appears in both train and val
    print("[pipeline] Splitting data (80/20 by IDpol) ...")
    unique_ids = freq['IDpol'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

    freq_train_mask = freq['IDpol'].isin(train_ids)
    freq_val_mask   = freq['IDpol'].isin(val_ids)
    sev_train_mask  = sev['IDpol'].isin(train_ids)
    sev_val_mask    = sev['IDpol'].isin(val_ids)

    freq_train_idx = np.where(freq_train_mask)[0]
    freq_val_idx   = np.where(freq_val_mask)[0]
    sev_train_idx  = np.where(sev_train_mask)[0]
    sev_val_idx    = np.where(sev_val_mask)[0]

    print(f"  Frequency train: {len(freq_train_idx)}, val: {len(freq_val_idx)}")
    print(f"  Severity  train: {len(sev_train_idx)}, val: {len(sev_val_idx)}")

    print("[pipeline] Fitting Poisson frequency model ...")
    freq_preprocessor = build_preprocessor(freq)
    freq_results = fit_frequency_glm(freq, freq_preprocessor, freq_train_idx, freq_val_idx)
    print(f"  Frequency converged: {freq_results['metrics']['converged']}")
    print(f"  Frequency deviance (full):  {freq_results['metrics']['deviance_full']:.2f}")
    print(f"  Frequency deviance (null):  {freq_results['metrics']['deviance_null']:.2f}")
    print(f"  Frequency deviance reduction: {freq_results['metrics']['percent_deviance_reduction']:.2f}%")

    # primary severity model -- validation targets are never touched/clipped
    print("[pipeline] Fitting Gamma severity model ...")
    sev_preprocessor = build_preprocessor(sev)
    sev_results = fit_severity_glm(sev, sev_preprocessor, sev_train_idx, sev_val_idx)
    print(f"  Severity converged: {sev_results['metrics']['converged']}")
    print(f"  Severity deviance (full):  {sev_results['metrics']['deviance_full']:.2f}")
    print(f"  Severity deviance (null):  {sev_results['metrics']['deviance_null']:.2f}")
    print(f"  Severity deviance reduction: {sev_results['metrics']['percent_deviance_reduction']:.2f}%")

    # winsorization sensitivity test: clip only training claims at 99th pct
    # computed from training data only so val targets stay original
    clip_val = float(sev.iloc[sev_train_idx]['ClaimAmount'].quantile(0.99))
    sev_wins = sev.copy()
    sev_wins.iloc[sev_train_idx, sev_wins.columns.get_loc('ClaimAmount')] = (
        sev_wins.iloc[sev_train_idx]['ClaimAmount'].clip(upper=clip_val).values
    )
    sev_wins_preprocessor = build_preprocessor(sev_wins)
    print(f"[pipeline] Winsorization check: 99th pct on training = {clip_val:.2f}")
    sev_wins_results = fit_severity_glm(
        sev_wins, sev_wins_preprocessor, sev_train_idx, sev_val_idx
    )
    print(f"  Winsorized model converged: {sev_wins_results['metrics']['converged']}")
    print(f"  Winsorized deviance (full):  {sev_wins_results['metrics']['deviance_full']:.2f}")
    print(f"  Winsorized deviance (null):  {sev_wins_results['metrics']['deviance_null']:.2f}")
    print(f"  Winsorized deviance reduction: {sev_wins_results['metrics']['percent_deviance_reduction']:.2f}%")

    print("[pipeline] Computing pure premium ...")
    premium_df = compute_pure_premium(
        freq_results, sev_results, freq, sev,
        freq_raw=freq_raw, train_ids=train_ids, val_ids=val_ids
    )
    print(f"  Pure premium computed for {len(premium_df)} policies.")

    agg_predicted = premium_df['pred_loss'].sum()
    agg_actual = premium_df['actual_claim_total'].sum()
    print(f"  Aggregate predicted losses: {agg_predicted:.2f}")
    print(f"  Aggregate actual losses:    {agg_actual:.2f}")

    reports_dir = base_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / 'figures').mkdir(exist_ok=True)

    metrics = {
        'frequency': freq_results['metrics'],
        'severity_original': sev_results['metrics'],
        'severity_winsorized_sensitivity': sev_wins_results['metrics'],
        'winsorization_clip_value': clip_val,
        'aggregate_predicted_losses': float(agg_predicted),
        'aggregate_actual_losses': float(agg_actual),
        'total_validation_policies': int(len(premium_df)),
        'policies_with_claims': int((premium_df['actual_claim_total'] > 0).sum()),
    }
    metrics_path = reports_dir / 'model_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved metrics to {metrics_path}")

    # save model coefficients
    freq_params = freq_results['model'].params
    if hasattr(freq_params, 'index'):
        freq_coef = pd.DataFrame({'feature': freq_params.index, 'coefficient': freq_params.values})
    else:
        freq_coef = pd.DataFrame({'feature': [f'x{i}' for i in range(len(freq_params))], 'coefficient': freq_params})
    freq_coef.to_csv(reports_dir / 'model_coefficients.csv', index=False)

    sev_params = sev_results['model'].params
    if hasattr(sev_params, 'index'):
        sev_coef = pd.DataFrame({'feature': sev_params.index, 'coefficient': sev_params.values})
    else:
        sev_coef = pd.DataFrame({'feature': [f'x{i}' for i in range(len(sev_params))], 'coefficient': sev_params})
    sev_coef.to_csv(reports_dir / 'model_coefficients_severity.csv', index=False)
    print(f"  Saved coefficients.")

    premium_df.to_csv(reports_dir / 'validation_predictions.csv', index=False)
    print(f"  Saved validation predictions to {reports_dir / 'validation_predictions.csv'}")

    print("\n[pipeline] DONE.")
    return {
        'frequency_results': freq_results,
        'severity_results': sev_results,
        'premium_predictions': premium_df,
        'metrics': metrics,
    }
