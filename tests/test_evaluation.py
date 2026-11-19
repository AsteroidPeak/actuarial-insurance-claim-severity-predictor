import numpy as np
import pandas as pd
import pytest


def test_poisson_deviance():
    # checking the formula manually with small numbers
    y  = np.array([0, 1, 2, 3])
    mu = np.array([0.5, 1.0, 2.0, 2.5])
    # y=0: 0 - (0-0.5) = 0.5
    # y=1: 1*log(1) - 0 = 0
    # y=2: 2*log(1) - 0 = 0
    # y=3: 3*log(1.2) - 0.5 ≈ 0.047
    # total ≈ 2 * 0.547 = 1.094
    mu_clipped = np.maximum(mu, 1e-12)
    term = np.where(y == 0, 0.0, y * np.log(np.maximum(y, 1e-12) / mu_clipped))
    deviance = 2.0 * np.sum(term - (y - mu))
    assert deviance > 0
    np.testing.assert_allclose(deviance, 1.094, atol=0.01)


def test_gamma_deviance():
    y  = np.array([100.0, 200.0, 300.0])
    mu = np.array([150.0, 180.0, 350.0])
    dev = 2.0 * np.sum(-np.log(y / mu) + (y - mu) / mu)
    assert dev >= 0


def test_pure_premium_structure():
    from insurance_glm.evaluation import compute_pure_premium

    freq_df = pd.DataFrame({
        'IDpol':    [1, 2, 3],
        'Exposure': [1.0, 0.5, 1.0],
        'ClaimNb':  [1, 0, 2],
    })
    sev_df = pd.DataFrame({
        'IDpol':       [1, 1, 3, 3],
        'ClaimAmount': [500.0, 300.0, 200.0, 400.0],
        'Exposure':    [1.0, 1.0, 1.0, 1.0],
    })

    freq_results = {
        'val_predictions': np.array([0.8, 0.1, 1.5]),
        'val_idx': np.array([0, 1, 2]),
    }
    sev_results = {
        'val_predictions': np.array([450.0, 350.0, 250.0, 380.0]),
        'val_idx': np.array([0, 1, 2, 3]),
        'model': None,
        'preprocessor': None,
    }

    result = compute_pure_premium(freq_results, sev_results, freq_df, sev_df)
    assert 'IDpol' in result.columns
    assert 'pure_premium' in result.columns
    assert 'pred_frequency' in result.columns
    assert 'pred_severity' in result.columns
    assert len(result) > 0
    assert (result['pure_premium'] >= 0).all()


def test_pure_premium_values():
    from insurance_glm.evaluation import compute_pure_premium

    freq_df = pd.DataFrame({
        'IDpol':    [10],
        'Exposure': [1.0],
        'ClaimNb':  [1],
    })
    sev_df = pd.DataFrame({
        'IDpol':       [10],
        'ClaimAmount': [500.0],
        'Exposure':    [1.0],
    })

    pred_count = 0.5  # expected claims over the exposure period
    pred_sev   = 400.0

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
    # pure_premium = (count / exposure) * severity = 0.5 * 400 = 200
    expected_pp = (pred_count / 1.0) * pred_sev
    np.testing.assert_allclose(result['pure_premium'].values[0], expected_pp, rtol=1e-6)
