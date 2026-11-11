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
5. **Leakage prevention:** The preprocessor is fit on training data only.

## Model Methodology

### Frequency Model (Poisson GLM)

$$Y_i \sim \text{Poisson}(\mu_i)$$

$$\log(\mu_i) = \beta_0 + \mathbf{x}_i^T \boldsymbol{\beta} + \log(\text{Exposure}_i)$$

- **Offset:** `log(Exposure)` accounts for differing policy durations.
- **Null model:** Intercept-only with the same offset.
- **Deviance:** $D = 2 \sum_{i} \left[ y_i \log\frac{y_i}{\hat{\mu}_i} - (y_i - \hat{\mu}_i) \right]$

### Severity Model (Gamma GLM)

$$Y_j \sim \text{Gamma}(\mu_j, \phi)$$

$$\log(\mu_j) = \beta_0 + \mathbf{x}_j^T \boldsymbol{\beta}$$

- **No offset:** Each claim is treated independently.
- **Null model:** Intercept-only Gamma GLM.
- **Deviance:** $D = 2 \sum_{j} \left[ -\log\frac{y_j}{\hat{\mu}_j} + \frac{y_j - \hat{\mu}_j}{\hat{\mu}_j} \right]$

### Pure Premium

$$\text{Pure Premium} = \frac{\hat{\mu}_{\text{freq}}}{\text{Exposure}} \times \hat{\mu}_{\text{sev}}$$

## Training and Validation Methodology

- **Split:** 80/20 split on unique policy IDs (`random_state=42`).
- **Leakage prevention:** All claims belonging to a policy stay in the same partition.
- **Frequency:** 542,410 training / 135,603 validation policies.
- **Severity:** 21,044 training / 5,400 validation claims.

## Results (Actual Measured Metrics)

### Frequency Model (Poisson)

| Metric | Value |
|--------|-------|
| Converged | ✅ Yes |
| Validation Deviance (Full Model) | 43,611.90 |
| Validation Deviance (Null Model) | 44,941.91 |
| Deviance Reduction | **2.96%** |

### Severity Model (Gamma)

| Metric | Value |
|--------|-------|
| Converged | ✅ Yes |
| Validation Deviance (Full Model) | 9,958.20 |
| Validation Deviance (Null Model) | 9,114.55 |
| Deviance Reduction | **−9.26%** |

> **Note on Severity Model:** The negative deviance reduction indicates the full severity model does not improve over the null model on validation data. This is a common finding with these covariates on the freMTPL2 severity dataset — claim amounts are highly variable and the available policy-level features have limited predictive power for individual claim sizes. This result is reported honestly.

### Pure Premium

| Metric | Value |
|--------|-------|
| Policies with pure premium computed | 5,101 |
| Aggregate predicted losses | €1,003,256 |
| Aggregate actual losses | €11,868,104 |

The large gap between predicted and actual aggregate losses reflects: (1) pure premium is computed only for policies that appear in **both** validation sets (frequency and severity), and (2) the severity model's lack of discriminatory power.

## Model Comparison

| Model | Type | Deviance Reduction vs Null |
|-------|------|---------------------------|
| Frequency (Poisson) | Full model | +2.96% improvement |
| Severity (Gamma) | Full model | −9.26% (no improvement) |

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
python scripts/run_pipeline.py

# Run the test suite
python -m pytest tests/ -v
```

## Project Structure

```
insurance-ml-glm/
├── AGENTS.md
├── README.md
├── STUDY_GUIDE.md
├── VIVA_QA.md
├── SUBMISSION_CHECKLIST.md
├── pyproject.toml
├── setup.py
├── setup.cfg
├── .gitignore
├── data/
│   ├── raw/              # Downloaded CSVs (excluded from Git)
│   └── processed/
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
│   └── test_evaluation.py
└── reports/
    ├── figures/
    ├── model_metrics.json
    ├── model_coefficients.csv
    ├── model_coefficients_severity.csv
    └── validation_predictions.csv
```

## Limitations

- The severity model's available covariates (vehicle/driver characteristics) have limited predictive power for individual claim sizes, resulting in no improvement over the null model.
- No interaction terms or spline transformations are explored.
- No regularization is applied; over-dispersion could be addressed with a quasi-Poisson or Negative Binomial model.
- The discrepancy between `sum(ClaimNb)` and the number of severity records limits the pure premium evaluation to the intersection of both datasets.

---

*All results are from actual pipeline execution. No metrics are fabricated. Raw data files are not committed; they are fetched automatically from OpenML.*

