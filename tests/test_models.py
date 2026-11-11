"""Tests for model fitting using small synthetic data."""
import numpy as np
import pandas as pd
import pytest
from insurance_glm.frequency import fit_frequency_glm
from insurance_glm.severity import fit_severity_glm
from insurance_glm.preprocessing import build_preprocessor


@pytest.fixture
def synthetic_freq():
    """Small synthetic frequency dataset (100 rows) with enough events."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'IDpol': range(n),
        'Exposure': np.random.uniform(0.3, 1.0, n),
        'ClaimNb': np.random.poisson(0.3, n),
        'VehBrand': np.random.choice(['B1', 'B2'], n),
        'Area': np.random.choice(['A', 'B'], n),
        'Region': np.random.choice(['R1', 'R2'], n),
        'VehGas': np.random.choice(['Regular', 'Diesel'], n),
        'Density': np.log(np.random.uniform(10, 5000, n)),
    })
    return df


@pytest.fixture
def synthetic_sev():
    """Small synthetic severity dataset (15 rows)."""
    np.random.seed(42)
    n = 15
    df = pd.DataFrame({
        'IDpol': np.random.choice(range(20), n),
        'ClaimAmount': np.random.gamma(2, 500, n),
        'VehBrand': np.random.choice(['B1', 'B2', 'B3'], n),
        'Area': np.random.choice(['A', 'B', 'C'], n),
        'Region': np.random.choice(['R1', 'R2'], n),
        'VehGas': np.random.choice(['Regular', 'Diesel'], n),
        'Density': np.log(np.random.uniform(10, 5000, n)),
    })
    return df


def test_frequency_model_converges(synthetic_freq):
    preprocessor = build_preprocessor(synthetic_freq)
    train_idx = np.arange(70)
    val_idx = np.arange(70, 100)
    result = fit_frequency_glm(synthetic_freq, preprocessor, train_idx, val_idx)
    assert result['metrics']['converged'] is True, "Frequency model should converge"


def test_frequency_predictions_positive(synthetic_freq):
    preprocessor = build_preprocessor(synthetic_freq)
    train_idx = np.arange(70)
    val_idx = np.arange(70, 100)
    result = fit_frequency_glm(synthetic_freq, preprocessor, train_idx, val_idx)
    preds = result['val_predictions']
    assert (preds >= 0).all(), "Predicted counts must be non-negative"


def test_frequency_deviance_nonnegative(synthetic_freq):
    preprocessor = build_preprocessor(synthetic_freq)
    train_idx = np.arange(70)
    val_idx = np.arange(70, 100)
    result = fit_frequency_glm(synthetic_freq, preprocessor, train_idx, val_idx)
    assert result['metrics']['deviance_full'] >= 0, "Deviance must be non-negative"
    assert result['metrics']['deviance_null'] >= 0, "Null deviance must be non-negative"


def test_severity_model_converges(synthetic_sev):
    preprocessor = build_preprocessor(synthetic_sev)
    train_idx = np.arange(10)
    val_idx = np.arange(10, 15)
    result = fit_severity_glm(synthetic_sev, preprocessor, train_idx, val_idx)
    assert result['metrics']['converged'] is True, "Severity model should converge"


def test_severity_predictions_positive(synthetic_sev):
    preprocessor = build_preprocessor(synthetic_sev)
    train_idx = np.arange(10)
    val_idx = np.arange(10, 15)
    result = fit_severity_glm(synthetic_sev, preprocessor, train_idx, val_idx)
    preds = result['val_predictions']
    assert (preds > 0).all(), "Predicted severities must be positive"


def test_severity_deviance_nonnegative(synthetic_sev):
    preprocessor = build_preprocessor(synthetic_sev)
    train_idx = np.arange(10)
    val_idx = np.arange(10, 15)
    result = fit_severity_glm(synthetic_sev, preprocessor, train_idx, val_idx)
    assert result['metrics']['deviance_full'] >= 0, "Deviance must be non-negative"
    assert result['metrics']['deviance_null'] >= 0, "Null deviance must be non-negative"
