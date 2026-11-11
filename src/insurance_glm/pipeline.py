"""End-to-end insurance GLM pipeline."""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from .data import download_datasets, load_frequency, load_severity
from .preprocessing import (preprocess_frequency, preprocess_severity,
                             build_preprocessor)
from .frequency import fit_frequency_glm
from .severity import fit_severity_glm
from .evaluation import compute_pure_premium


def run_pipeline(base_dir=None):
    """Run the full pipeline: download, preprocess, train, evaluate, save.

    Parameters
    ----------
    base_dir : Path or str, default '.'
        Project root directory.

    Returns
    -------
    dict with frequency_results, severity_results, premium_predictions, metrics
    """
    if base_dir is None:
        base_dir = Path('.')
    base_dir = Path(base_dir)

    # ---- 1. Data acquisition ------------------------------------------------
    raw_dir = base_dir / 'data' / 'raw'
    freq_path = raw_dir / 'freMTPL2freq.csv'
    sev_path = raw_dir / 'freMTPL2sev.csv'

    if not freq_path.is_file() or not sev_path.is_file():
        print("[pipeline] Downloading datasets ...")
        freq_path, sev_path = download_datasets(raw_dir)
    else:
        print("[pipeline] Raw data already present, skipping download.")

    # ---- 2. Load raw data ----------------------------------------------------
    print("[pipeline] Loading raw data ...")
    freq_raw = load_frequency(freq_path)
    sev_raw = load_severity(sev_path)
    print(f"  Frequency shape: {freq_raw.shape}")
    print(f"  Severity  shape: {sev_raw.shape}")

    # ---- 3. Data validation --------------------------------------------------
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

    total_claimnb = freq_raw['ClaimNb'].sum()
    num_sev_records = len(sev_raw)
    print(f"  Total ClaimNb in freq data: {total_claimnb}")
    print(f"  Number of severity records:  {num_sev_records}")
    if total_claimnb != num_sev_records:
        print(f"  DISCREPANCY: sum(ClaimNb)={total_claimnb} != severity records={num_sev_records}")

    # ---- 4. Preprocessing ----------------------------------------------------
    print("[pipeline] Preprocessing ...")
    freq = preprocess_frequency(freq_raw)
    print(f"  Frequency after preprocessing: {freq.shape}")

    # For severity: join policy-level features, then preprocess
    # Keep only columns needed: IDpol, ClaimAmount from severity
    # Join policy features from freq_raw
    policy_features = freq_raw.drop(columns=['ClaimNb', 'Exposure'], errors='ignore')
    sev_joined = sev_raw.merge(policy_features, on='IDpol', how='left')
    sev = preprocess_severity(sev_joined)
    # Reset indices to make iloc-based indexing safe
    freq = freq.reset_index(drop=True)
    sev = sev.reset_index(drop=True)
    print(f"  Severity after preprocessing:  {sev.shape}")

    # ---- 5. Train / Validation split -----------------------------------------
    print("[pipeline] Splitting data (80/20 by IDpol) ...")
    unique_ids = freq['IDpol'].unique()
    train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)

    freq_train_mask = freq['IDpol'].isin(train_ids)
    freq_val_mask = freq['IDpol'].isin(val_ids)
    sev_train_mask = sev['IDpol'].isin(train_ids)
    sev_val_mask = sev['IDpol'].isin(val_ids)

    # Use positional integer indices (safe after reset_index)
    freq_train_idx = np.where(freq_train_mask)[0]
    freq_val_idx = np.where(freq_val_mask)[0]
    sev_train_idx = np.where(sev_train_mask)[0]
    sev_val_idx = np.where(sev_val_mask)[0]

    print(f"  Frequency train: {len(freq_train_idx)}, val: {len(freq_val_idx)}")
    print(f"  Severity  train: {len(sev_train_idx)}, val: {len(sev_val_idx)}")

    # ---- 6. Frequency model --------------------------------------------------
    print("[pipeline] Fitting Poisson frequency model ...")
    freq_preprocessor = build_preprocessor(freq)
    freq_results = fit_frequency_glm(freq, freq_preprocessor, freq_train_idx, freq_val_idx)
    print(f"  Frequency converged: {freq_results['metrics']['converged']}")
    print(f"  Frequency deviance (full):  {freq_results['metrics']['deviance_full']:.2f}")
    print(f"  Frequency deviance (null):  {freq_results['metrics']['deviance_null']:.2f}")
    print(f"  Frequency deviance reduction: {freq_results['metrics']['percent_deviance_reduction']:.2f}%")

    # ---- 7. Severity model ---------------------------------------------------
    print("[pipeline] Fitting Gamma severity model ...")
    sev_preprocessor = build_preprocessor(sev)
    sev_results = fit_severity_glm(sev, sev_preprocessor, sev_train_idx, sev_val_idx)
    print(f"  Severity converged: {sev_results['metrics']['converged']}")
    print(f"  Severity deviance (full):  {sev_results['metrics']['deviance_full']:.2f}")
    print(f"  Severity deviance (null):  {sev_results['metrics']['deviance_null']:.2f}")
    print(f"  Severity deviance reduction: {sev_results['metrics']['percent_deviance_reduction']:.2f}%")

    # ---- 8. Pure premium -----------------------------------------------------
    print("[pipeline] Computing pure premium ...")
    premium_df = compute_pure_premium(freq_results, sev_results, freq, sev)
    print(f"  Pure premium computed for {len(premium_df)} policies.")

    agg_predicted = (premium_df['pure_premium'] * premium_df['Exposure']).sum()
    agg_actual = premium_df['actual_claim_total'].sum()
    print(f"  Aggregate predicted losses: {agg_predicted:.2f}")
    print(f"  Aggregate actual losses:    {agg_actual:.2f}")

    # ---- 9. Save reports -----------------------------------------------------
    reports_dir = base_dir / 'reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / 'figures').mkdir(exist_ok=True)

    metrics = {
        'frequency': freq_results['metrics'],
        'severity': sev_results['metrics'],
        'aggregate_predicted_losses': float(agg_predicted),
        'aggregate_actual_losses': float(agg_actual),
    }
    metrics_path = reports_dir / 'model_metrics.json'
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"  Saved metrics to {metrics_path}")

    # Coefficients
    freq_params = freq_results['model'].params
    if hasattr(freq_params, 'index'):
        freq_coef = pd.DataFrame({
            'feature': freq_params.index,
            'coefficient': freq_params.values,
        })
    else:
        freq_coef = pd.DataFrame({
            'feature': [f'x{i}' for i in range(len(freq_params))],
            'coefficient': freq_params,
        })
    freq_coef_path = reports_dir / 'model_coefficients.csv'
    freq_coef.to_csv(freq_coef_path, index=False)

    sev_params = sev_results['model'].params
    if hasattr(sev_params, 'index'):
        sev_coef = pd.DataFrame({
            'feature': sev_params.index,
            'coefficient': sev_params.values,
        })
    else:
        sev_coef = pd.DataFrame({
            'feature': [f'x{i}' for i in range(len(sev_params))],
            'coefficient': sev_params,
        })
    sev_coef_path = reports_dir / 'model_coefficients_severity.csv'
    sev_coef.to_csv(sev_coef_path, index=False)
    print(f"  Saved coefficients.")

    # Validation predictions
    pred_path = reports_dir / 'validation_predictions.csv'
    premium_df.to_csv(pred_path, index=False)
    print(f"  Saved validation predictions to {pred_path}")

    print("\n[pipeline] DONE.")
    return {
        'frequency_results': freq_results,
        'severity_results': sev_results,
        'premium_predictions': premium_df,
        'metrics': metrics,
    }

