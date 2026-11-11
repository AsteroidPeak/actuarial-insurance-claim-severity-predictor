#!/usr/bin/env python
"""Execute the full insurance GLM pipeline.
It downloads data, fits models, evaluates, and saves reports.
"""
import pathlib
from insurance_glm.pipeline import run_pipeline

def main():
    base_dir = pathlib.Path(__file__).resolve().parents[1]
    results = run_pipeline(base_dir)
    print("--- Pipeline completed ---")
    print("Metrics:")
    for model, metric in results['metrics'].items():
        print(f"{model}: {metric}")

if __name__ == "__main__":
    main()

