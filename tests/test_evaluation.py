"""Tests for evaluation / pure premium computation."""
import numpy as np
import pandas as pd
import pytest


def test_poisson_deviance():
    """Test the Poisson deviance formula directly."""
    from insurance_glm.frequency import fit_frequency_glm
    # Test the deviance formula with known values
    y = np.array([0, 1, 2, 3])
    mu = np.array([0.5, 1.0, 2.0, 2.5])
    # manual: 2 * sum(y*log(y/mu) - (y - mu))
    # y=0: 0 - (0-0.5) = 0.5
    # y=1: 1*log(1/1) - (1-1) = 0
    # y=2: 2*log(2/2) - (2-2) = 0
    # y=3: 3*log(3/2.5) - (3-2.5) = 3*0.1823 - 0.5 = 0.0470
    # total = 2*(0.5 + 0 + 0 + 0.0470) = 1.094
    mu_clipped = np.maximum(mu, 1e-12)
    term = np.where(y == 0, 0.0, y * np.log(np.maximum(y, 1e-12) / mu_clipped))
    deviance = 2.0 * np.sum(term - (y - mu))
    assert deviance > 0, "Deviance should be positive"
    np.testing.assert_allclose(deviance, 1.094, atol=0.01)


def test_gamma_deviance():
    """Test the Gamma deviance formula directly."""
    y = np.array([100.0, 200.0, 300.0])
    mu = np.array([150.0, 180.0, 350.0])
    # gamma deviance: 2 * sum(-log(y/mu) + (y-mu)/mu)
    dev = 2.0 * np.sum(-np.log(y / mu) + (y - mu) / mu)
    assert dev >= 0, "Gamma deviance should be non-negative"


def test_pure_premium_structure():
    """Test that pure premium output has the right columns."""
    from insurance_glm.evaluation import compute_pure_premium

    freq_df = pd.DataFrame({
        'IDpol': [1, 2, 3],
        'Exposure': [1.0, 0.5, 1.0],
        'ClaimNb': [1, 0, 2],
    })
    sev_df = pd.DataFrame({
        'IDpol': [1, 1, 3, 3],
        'ClaimAmount': [500.0, 300.0, 200.0, 400.0],
        'Exposure': [1.0, 1.0, 1.0, 1.0],
    })

    freq_results = {
        'val_predictions': np.array([0.8, 0.1, 1.5]),
        'val_idx': np.array([0, 1, 2]),
    }
    sev_results = {
        'val_predictions': np.array([450.0, 350.0, 250.0, 380.0]),
        'val_idx': np.array([0, 1, 2, 3]),
    }

    result = compute_pure_premium(freq_results, sev_results, freq_df, sev_df)
    assert 'IDpol' in result.columns
    assert 'pure_premium' in result.columns
    assert 'pred_frequency' in result.columns
    assert 'pred_severity' in result.columns
    assert len(result) > 0
    assert (result['pure_premium'] >= 0).all()


def test_pure_premium_values():
    """Test pure premium arithmetic on a simple example."""
    from insurance_glm.evaluation import compute_pure_premium

    freq_df = pd.DataFrame({
        'IDpol': [10],
        'Exposure': [1.0],
        'ClaimNb': [1],
    })
    sev_df = pd.DataFrame({
        'IDpol': [10],
        'ClaimAmount': [500.0],
        'Exposure': [1.0],
    })

    pred_count = 0.5  # predicted claims in exposure period
    pred_sev = 400.0  # predicted severity per claim

    freq_results = {
        'val_predictions': np.array([pred_count]),
        'val_idx': np.array([0]),
    }
    sev_results = {
        'val_predictions': np.array([pred_sev]),
        'val_idx': np.array([0]),
    }

    result = compute_pure_premium(freq_results, sev_results, freq_df, sev_df)
    expected_pp = (pred_count / 1.0) * pred_sev  # annualized freq * severity
    np.testing.assert_allclose(result['pure_premium'].values[0], expected_pp, rtol=1e-6)

