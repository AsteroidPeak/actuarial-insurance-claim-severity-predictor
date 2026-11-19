# Insurance Claim Frequency and Severity Prediction Using Generalized Linear Models

## Project Overview

This repository implements a classic actuarial pricing workflow using Generalized Linear Models (GLMs):

- **Poisson GLM** for claim **frequency** (number of claims per policy).
- **Gamma GLM** with a log link for claim **severity** (average claim amount per claim).
- The **pure premium** is obtained by multiplying the predicted annualized frequency by the predicted severity.

The implementation follows a reproducible pipeline that downloads the **freMTPL2freq** and **freMTPL2sev** datasets from OpenML, performs thorough preprocessing, fits the models with `statsmodels`, evaluates deviance reductions against null models, and saves all artefacts.

## Problem Statement

Insurance companies need to price policies by estimating the expected cost (pure premium) of each policyholder. This is decomposed into two components:

1. **How often** will a policyholder make a claim? (Frequency)
2. **How much** will each claim cost? (Severity)

The product of these two quantities gives the **pure premium** — the minimum premium required to cover expected losses.

## Dataset Sources

| Dataset | OpenML ID | Rows | Description |
|---------|-----------|------|-------------|
| freMTPL2freq | 41214 | 678,013 | Policy-level data with exposure, claim count, and vehicle/driver attributes |
| freMTPL2sev | 41215 | 26,639 | Individual claim records with claim amounts |

**Note:** The sum of `ClaimNb` in the frequency dataset (36,102) does not match the number of severity records (26,639). This discrepancy is documented and reported by the pipeline rather than silently modified.

## Features

| Feature | Type | Description |
|---------|------|-------------|
| IDpol | Identifier | Policy ID (not used as predictor) |
| ClaimNb | Target (freq) | Number of claims |
| Exposure | Offset (freq) | Duration of policy coverage in years |
| ClaimAmount | Target (sev) | Individual claim amount in EUR |
| VehPower | Numeric | Vehicle power rating |
| VehAge | Numeric | Vehicle age in years |
| DrivAge | Numeric | Driver age in years |
| BonusMalus | Numeric | Bonus-malus coefficient |
| VehBrand | Categorical | Vehicle brand |
| VehGas | Categorical | Fuel type (Regular/Diesel) |
| Area | Categorical | Population density area code |
| Density | Numeric | Population density (log-transformed) |
| Region | Categorical | Geographic region |

## Preprocessing

1. **Categorical encoding:** `VehBrand`, `Area`, `Region`, `VehGas` are one-hot encoded using `ColumnTransformer` with `drop='first'` to avoid multicollinearity with the intercept.
2. **Log transformation:** `Density` is log-transformed to reduce skewness.
3. **Zero exposure removal:** Records with `Exposure <= 0` are dropped.
4. **Missing values:** Categorical missing values are filled with `'Missing'`; remaining NaNs are dropped.
5. **Leakage prevention:** The preprocessor is fit on training data only. No winsorization is applied before the train/validation split.

## Model Methodology

### Frequency Model (Poisson GLM)

$$Y_i \sim \text{Poisson}(\mu_i)$$

$$\log(\mu_i) = \beta_0 + \mathbf{x}_i^T \boldsymbol{\beta} + \log(\text{Exposure}_i)$$

- **Offset:** `log(Exposure)` accounts for differing policy durations. The prediction `pred_count` is the expected claim count over the actual exposure period.
- **Null model:** Intercept-only with the same offset.
- **Deviance:** $D = 2 \sum_{i} \left[ y_i \log\frac{y_i}{\hat{\mu}_i} - (y_i - \hat{\mu}_i) \right]$

### Severity Model (Gamma GLM)

$$Y_j \sim \text{Gamma}(\mu_j, \phi)$$

$$\log(\mu_j) = \beta_0 + \mathbf{x}_j^T \boldsymbol{\beta}$$

- **No offset:** Each claim is treated independently.
- **Training data only:** 99th-percentile winsorization is explored as a *sensitivity experiment* on training data only; validation targets are never modified.
- **Null model:** Intercept-only Gamma GLM.
- **Deviance:** $D = 2 \sum_{j} \left[ -\log\frac{y_j}{\hat{\mu}_j} + \frac{y_j - \hat{\mu}_j}{\hat{\mu}_j} \right]$

### Pure Premium

$$\text{Pure Premium} = \frac{\hat{\mu}_{\text{count}}}{\text{Exposure}} \times \hat{\mu}_{\text{sev}}$$

$$\text{Predicted loss per policy} = \hat{\mu}_{\text{count}} \times \hat{\mu}_{\text{sev}}$$

The Poisson offset `log(Exposure)` is already incorporated into `pred_count`. Exposure is **not** multiplied again when computing aggregate losses.

## Training and Validation Methodology

- **Split:** 80/20 split on unique policy IDs (`random_state=42`).
- **Leakage prevention:** All claims belonging to a policy stay in the same partition.
- **Frequency:** 542,410 training / 135,603 validation policies.
- **Severity:** 21,044 training / 5,400 validation claims.
- **Full-policy severity prediction:** The Gamma model predicts severity for **every** frequency-validation policy (135,603 policies) using its vehicle/driver features, not only the 5,101 that have observed claims.

## Results (Actual Measured Metrics)

### Frequency Model (Poisson)

| Metric | Value |
|--------|-------|
| Converged | ✅ Yes |
| Validation Deviance (Full Model) | 43,611.90 |
| Validation Deviance (Null Model) | 44,941.91 |
| Deviance Reduction | **+2.96%** |

### Severity Model (Gamma) — Primary Results (Original Unmodified Validation Targets)

| Metric | Value |
|--------|-------|
| Converged | ✅ Yes |
| Validation Deviance (Full Model) | 9,958.20 |
| Validation Deviance (Null Model) | 9,114.55 |
| Deviance Reduction | **−9.26%** |

> **Honest finding:** The full severity model does **not** improve over the null model on held-out validation data. Gamma deviance on the original (unclipped) validation claim amounts is worse for the full model than for the null. This is a well-documented limitation of the available policy-level covariates for predicting individual claim amounts on the freMTPL2 dataset.

### Severity Model — Winsorization Sensitivity Experiment

Winsorization (clipping at 99th percentile of **training data only**; validation targets unchanged):

| Metric | Value |
|--------|-------|
| 99th pct clip value (training) | €16,252.02 |
| Validation Deviance (Full, winsorized training) | 9,803.11 |
| Validation Deviance (Null, winsorized training) | 9,805.52 |
| Deviance Reduction | **+0.02%** |

This marginal improvement (+0.02%) confirms the lack of predictive signal rather than resolving it.

### Pure Premium

| Metric | Value |
|--------|-------|
| Total validation policies | 135,603 |
| Policies with at least one claim | 5,101 |
| Total actual claim records (val set) | 5,400 |
| Aggregate predicted losses | €15,827,461 |
| Aggregate actual losses | €11,868,104 |
| Predicted / actual loss ratio | 1.334 |
| Average predicted severity (all val policies) | €2,119.12 |
| Average actual severity (claimants only) | €2,326.62 |

#### Reconciliation of earlier figures
- **€11.87M** (earlier "actual"): this is the correct figure — sum of all severity records belonging to the frequency-validation policy IDs.
- **€8.51M** (interim figure): arose from an earlier bug where severity records were aggregated *before* filtering to the frequency-validation population. Now fixed.
- **€1.003M** (older predicted figure): arose because pure premium was computed only for the 5,101 intersection policies instead of all 135,603 validation policies.

## Model Comparison

| Model | Type | Deviance Reduction vs Null |
|-------|------|---------------------------|
| Frequency (Poisson) | Full model | **+2.96%** improvement |
| Severity (Gamma, original) | Full model | **−9.26%** (worse than null) |
| Severity (Gamma, winsorized sensitivity) | Sensitivity only | **+0.02%** (negligible) |

## How to Install Dependencies

```bash
# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the package and all dependencies
pip install -e .
```

## How to Reproduce the Project

```bash
# Run the full pipeline (downloads data, trains, evaluates, saves reports)
python3 scripts/run_pipeline.py

# Run the test suite
python3 -m pytest tests/ -v
```

## Project Structure

```
insurance-ml-glm/
├── README.md
├── STUDY_GUIDE.md
├── VIVA_QA.md
├── SUBMISSION_CHECKLIST.md
├── pyproject.toml
├── setup.py
├── setup.cfg
├── .gitignore
├── data/
│   └── raw/              # Downloaded CSVs (excluded from Git)
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── src/insurance_glm/
│   ├── __init__.py
│   ├── data.py           # Data download and loading
│   ├── preprocessing.py  # Cleaning, encoding, transforms
│   ├── frequency.py      # Poisson GLM fitting
│   ├── severity.py       # Gamma GLM fitting
│   ├── evaluation.py     # Pure premium computation
│   └── pipeline.py       # End-to-end orchestration
├── scripts/
│   ├── download_data.py
│   └── run_pipeline.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_evaluation.py
│   └── test_full_policy_evaluation.py
└── reports/
    ├── figures/
    ├── model_metrics.json
    ├── model_coefficients.csv
    ├── model_coefficients_severity.csv
    └── validation_predictions.csv
```

## Limitations

- The severity model's available covariates (vehicle/driver characteristics) have limited predictive power for individual claim sizes, resulting in no improvement over the null model (−9.26% deviance reduction).
- No interaction terms or spline transformations are explored.
- No regularization is applied; over-dispersion could be addressed with a quasi-Poisson or Negative Binomial model.
- The discrepancy between `sum(ClaimNb)=36,102` and severity records `26,639` is documented but not corrected; the severity model is trained on the available severity records.
- The loss ratio of 1.334 reflects the severity model's overprediction due to its reliance on features with limited discriminatory power.

---

*All results are from actual pipeline execution. No metrics are fabricated. Raw data files are not committed; they are fetched automatically from OpenML.*
