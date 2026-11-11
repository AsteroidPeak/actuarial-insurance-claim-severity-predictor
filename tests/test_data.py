"""Tests for data loading and validation."""
import pathlib
import pytest
import pandas as pd
from insurance_glm.data import load_frequency, load_severity


@pytest.fixture
def freq_path():
    p = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'freMTPL2freq.csv'
    if not p.exists():
        pytest.skip("Raw data not downloaded yet")
    return p


@pytest.fixture
def sev_path():
    p = pathlib.Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'freMTPL2sev.csv'
    if not p.exists():
        pytest.skip("Raw data not downloaded yet")
    return p


def test_frequency_columns(freq_path):
    df = load_frequency(freq_path)
    for col in ['IDpol', 'Exposure', 'ClaimNb', 'VehBrand', 'Area', 'Region', 'VehGas', 'Density']:
        assert col in df.columns, f"Missing column: {col}"


def test_frequency_idpol_unique(freq_path):
    df = load_frequency(freq_path)
    assert df['IDpol'].is_unique, "IDpol should be unique in frequency data"


def test_frequency_positive_exposure(freq_path):
    df = load_frequency(freq_path)
    assert (df['Exposure'] > 0).all(), "All exposures should be positive"


def test_severity_columns(sev_path):
    df = load_severity(sev_path)
    for col in ['IDpol', 'ClaimAmount']:
        assert col in df.columns, f"Missing column: {col}"


def test_severity_positive_amounts(sev_path):
    df = load_severity(sev_path)
    assert (df['ClaimAmount'] > 0).all(), "All claim amounts should be positive"


def test_frequency_shape(freq_path):
    df = load_frequency(freq_path)
    assert df.shape[0] > 600000, f"Expected >600k rows, got {df.shape[0]}"
    assert df.shape[1] == 12, f"Expected 12 columns, got {df.shape[1]}"

