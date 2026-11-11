import os
import pandas as pd
import openml
from pathlib import Path

def download_datasets(raw_dir: Path = Path('data/raw')):
    """Download the frequency and severity datasets from OpenML.
    The function creates the `raw_dir` if it does not exist and saves the
    datasets as CSV files named `freMTPL2freq.csv` and `freMTPL2sev.csv`.
    Returns the paths to the saved CSV files.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Frequency dataset (ID 41214)
    freq = openml.datasets.get_dataset(41214, download_all_files=True)
    freq_df = freq.get_data()[0]
    freq_path = raw_dir / 'freMTPL2freq.csv'
    freq_df.to_csv(freq_path, index=False)

    # Severity dataset (ID 41215)
    sev = openml.datasets.get_dataset(41215, download_all_files=True)
    sev_df = sev.get_data()[0]
    sev_path = raw_dir / 'freMTPL2sev.csv'
    sev_df.to_csv(sev_path, index=False)

    return freq_path, sev_path

def load_frequency(path: Path = Path('data/raw/freMTPL2freq.csv')) -> pd.DataFrame:
    return pd.read_csv(path)

def load_severity(path: Path = Path('data/raw/freMTPL2sev.csv')) -> pd.DataFrame:
    return pd.read_csv(path)

