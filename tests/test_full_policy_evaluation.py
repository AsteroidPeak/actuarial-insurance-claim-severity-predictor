"""Additional evaluation tests: full-policy severity prediction, no double exposure, zero-claim aggregation."""
import numpy as np
import pandas as pd
import pytest


def _make_freq_df():
    """Minimal frequency DataFrame with 3 policies."""
    return pd.DataFrame({
        'IDpol':    [1, 2, 3],
        'Exposure': [1.0, 0.5, 2.0],
        'ClaimNb':  [1,   0,   2],
        'VehBrand': ['B1', 'B1', 'B2'],
        'Area':     ['A', 'B', 'A'],
        'Region':   ['R1', 'R2', 'R1'],
        'VehGas':   ['Regular', 'Diesel', 'Regular'],
        'Density':  [5.0, 6.0, 4.5],
    })


def _make_sev_df():
    """Severity DataFrame — only IDpol 1 and 3 have observed claims."""
    return pd.DataFrame({
        'IDpol':       [1,     1,     3,     3],
        'ClaimAmount': [500.0, 300.0, 200.0, 400.0],
        'VehBrand':    ['B1', 'B1', 'B2', 'B2'],
        'Area':        ['A',  'A',  'A',  'A'],
        'Region':      ['R1', 'R1', 'R1', 'R1'],
        'VehGas':      ['Regular', 'Regular', 'Regular', 'Regular'],
        'Density':     [5.0, 5.0, 4.5, 4.5],
    })


def test_pred_severity_covers_all_validation_policies():
    """Result must contain one row for every frequency-validation policy (no missing rows)."""
    from insurance_glm.evaluation import compute_pure_premium

    freq_df = _make_freq_df()
    sev_df = _make_sev_df()

    freq_results = {
        'val_predictions': np.array([0.1, 0.05, 0.3]),
        'val_idx': np.array([0, 1, 2]),
    }
    sev_results = {
        'val_predictions': np.array([450.0, 350.0, 250.0, 380.0]),
        'val_idx': np.array([0, 1, 2, 3]),
        'model': None,
        'preprocessor': None,
    }

    result = compute_pure_premium(freq_results, sev_results, freq_df, sev_df)

    # Every frequency-validation policy must appear in the result
    assert len(result) == len(freq_df), (
        f"Expected {len(freq_df)} rows, got {len(result)}"
    )
    # IDpol column must contain all three policy IDs
    assert set(result['IDpol'].tolist()) == {1, 2, 3}, (
        "Result must contain exactly the 3 validation policy IDs"
    )
    # pred_severity must have no NaN values
    assert result['pred_severity'].isna().sum() == 0, (
        "pred_severity must not contain NaN"
    )
    # actual_claim_total must have no NaN values
    assert result['actual_claim_total'].isna().sum() == 0, (
        "actual_claim_total must not contain NaN"
    )


def test_no_double_exposure_in_pred_loss():
    """pred_loss must equal pred_count * pred_severity (Exposure NOT multiplied again)."""
    from insurance_glm.evaluation import compute_pure_premium

    freq_df = pd.DataFrame({
        'IDpol':    [10],
        'Exposure': [2.0],   # 2-year policy
        'ClaimNb':  [1],
        'VehBrand': ['B1'], 'Area': ['A'], 'Region': ['R1'],
        'VehGas':   ['Regular'], 'Density': [5.0],
    })
    sev_df = pd.DataFrame({
        'IDpol':       [10],
        'ClaimAmount': [800.0],
        'VehBrand': ['B1'], 'Area': ['A'], 'Region': ['R1'],
        'VehGas':   ['Regular'], 'Density': [5.0],
    })

    pred_count = 0.4   # expected claims over the 2-year exposure
    pred_sev   = 600.0

    freq_results = {
        'val_predictions': np.array([pred_count]),
        'val_idx': np.array([0]),
    }
    sev_results = {
        'val_predictions': np.array([pred_sev]),
        'val_idx': np.array([0]),
        'model': None,
        'preprocessor': None,
    }

    result = compute_pure_premium(freq_results, sev_results, freq_df, sev_df)

    expected_pred_loss = pred_count * pred_sev   # 240.0
    np.testing.assert_allclose(
        result['pred_loss'].values[0], expected_pred_loss, rtol=1e-6,
        err_msg="pred_loss must be pred_count * pred_severity (no extra Exposure factor)"
    )
    # Also verify pure_premium (annualised) = pred_frequency * pred_severity
    expected_pp = (pred_count / 2.0) * pred_sev  # 120.0
    np.testing.assert_allclose(
        result['pure_premium'].values[0], expected_pp, rtol=1e-6,
        err_msg="pure_premium must be (pred_count/Exposure) * pred_severity"
    )


def test_zero_claim_policies_get_zero_actual_total():
    """Policies with no severity records must have actual_claim_total == 0."""
    from insurance_glm.evaluation import compute_pure_premium

    freq_df = pd.DataFrame({
        'IDpol':    [20, 21, 22],
        'Exposure': [1.0, 1.0, 1.0],
        'ClaimNb':  [0,   0,   1],
        'VehBrand': ['B1', 'B1', 'B2'],
        'Area':     ['A', 'A', 'B'],
        'Region':   ['R1', 'R1', 'R2'],
        'VehGas':   ['Regular', 'Regular', 'Diesel'],
        'Density':  [5.0, 5.0, 6.0],
    })
    # Only IDpol 22 has a severity record
    sev_df = pd.DataFrame({
        'IDpol':       [22],
        'ClaimAmount': [1200.0],
        'VehBrand': ['B2'], 'Area': ['B'], 'Region': ['R2'],
        'VehGas': ['Diesel'], 'Density': [6.0],
    })

    freq_results = {
        'val_predictions': np.array([0.02, 0.03, 0.15]),
        'val_idx': np.array([0, 1, 2]),
    }
    sev_results = {
        'val_predictions': np.array([1200.0]),
        'val_idx': np.array([0]),
        'model': None,
        'preprocessor': None,
    }

    result = compute_pure_premium(freq_results, sev_results, freq_df, sev_df)

    # IDpol 20 and 21 have no claims → actual_claim_total must be 0
    no_claim = result[result['IDpol'].isin([20, 21])]
    assert (no_claim['actual_claim_total'] == 0.0).all(), (
        "Policies without claim records must have actual_claim_total == 0"
    )
    # IDpol 22 has actual claims
    with_claim = result[result['IDpol'] == 22]
    assert with_claim['actual_claim_total'].values[0] == 1200.0
