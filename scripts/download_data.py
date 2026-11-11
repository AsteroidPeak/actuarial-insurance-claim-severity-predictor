#!/usr/bin/env python
"""Download OpenML datasets and store them under data/raw/"""
from pathlib import Path
from insurance_glm.data import download_datasets

def main():
    raw_dir = Path(__file__).resolve().parents[1] / 'data' / 'raw'
    freq_path, sev_path = download_datasets(raw_dir)
    print(f"Downloaded frequency to {freq_path}\nDownloaded severity to {sev_path}")

if __name__ == '__main__':
    main()

