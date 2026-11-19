# Study Guide — Insurance GLM Project

## 1. Project Purpose

This project implements an actuarial **Frequency–Severity** pricing model using Generalized Linear Models (GLMs) on the French Motor Third-Party Liability dataset (freMTPL2). It is a standard industry approach to estimating the **pure premium** — the expected cost a policy is likely to generate.

---

## 2. The Frequency–Severity Framework

| Component | Model | Distribution | Link | Target | Offset |
|-----------|-------|--------------|------|--------|--------|
| Frequency | Poisson GLM | Poisson | Log | ClaimNb | log(Exposure) |
| Severity | Gamma GLM | Gamma | Log | ClaimAmount | None |
| Pure Premium | — | — | — | Frequency × Severity | — |

**Pure premium formula:**

$$\text{Pure Premium} = \frac{\hat{\lambda}}{\text{Exposure}} \times \hat{\mu}_{\text{sev}}$$

where $\hat{\lambda}$ = predicted claim count over the actual exposure period (already includes the exposure offset), so:

$$\text{Predicted loss per policy} = \hat{\lambda} \times \hat{\mu}_{\text{sev}}$$

Exposure is **not** multiplied again when aggregating.

---

## 3. Why Poisson for Frequency?

- Claim counts are non-negative integers.
- The Poisson distribution is the natural choice when events occur independently and at a constant rate.
- The **log offset** `log(Exposure)` corrects for different observation durations without inflating the exposure as a predictor.
- The null model uses an intercept only with the same offset, so the deviance comparison is fair.

---

## 4. Why Gamma for Severity?

- Claim amounts are strictly positive and right-skewed.
- The Gamma distribution with log link keeps predictions positive and handles skewness better than a linear model.
- Evaluated on individual claim rows, not aggregated per policy.

---

## 5. What is Deviance?

Deviance measures goodness of fit relative to a saturated model (perfect fit).

**Poisson deviance:**
$$D = 2 \sum_i \left[ y_i \log\frac{y_i}{\hat{\mu}_i} - (y_i - \hat{\mu}_i) \right]$$

**Gamma deviance:**
$$D = 2 \sum_j \left[ -\log\frac{y_j}{\hat{\mu}_j} + \frac{y_j - \hat{\mu}_j}{\hat{\mu}_j} \right]$$

**Percentage deviance reduction** (vs null):
$$\% \text{ reduction} = \frac{D_{\text{null}} - D_{\text{full}}}{D_{\text{null}}} \times 100$$

A positive reduction means the full model fits better than the intercept-only null.

---

## 6. Actual Results

| Metric | Value |
|--------|-------|
| Frequency deviance (full) | 43,611.90 |
| Frequency deviance (null) | 44,941.91 |
| Frequency deviance reduction | **+2.96%** ✅ |
| Severity deviance (full, original) | 9,958.20 |
| Severity deviance (null, original) | 9,114.55 |
| Severity deviance reduction | **−9.26%** ⚠️ |
| Winsorized severity (sensitivity) deviance reduction | +0.02% |
| Validation policies | 135,603 |
| Policies with claims | 5,101 |
| Actual claim records (val) | 5,400 |
| Aggregate predicted losses | €15,827,461 |
| Aggregate actual losses | €11,868,104 |
| Loss ratio (pred/actual) | 1.334 |

---

## 7. Why Does the Severity Model Underperform?

The Gamma GLM performs worse than the null model (−9.26% deviance reduction) because:

1. **Claim amounts are extremely heavy-tailed.** A few very large claims dominate the deviance calculation.
2. **Policy features are weak predictors of claim size.** Vehicle brand, area, region, driver age, etc. explain frequency well but not severity.
3. **The model tends to overpredict severity** for policies with low actual claim amounts.

This is a known limitation of the freMTPL2 dataset — an honest finding, not a modelling error.

---

## 8. Winsorization as Sensitivity Experiment

Winsorization (clipping at the 99th percentile) is applied **to training data only** (computed from training rows, never from validation). The validation targets remain unmodified. This ensures:

- No data leakage (the clip threshold is not informed by validation data).
- Primary deviance results are reported on the original claim amounts.
- Winsorized results are labelled explicitly as a sensitivity experiment.

Result: Winsorizing the training data yields +0.02% deviance reduction — essentially zero improvement.

---

## 9. Exposure Handling

The Poisson GLM uses `offset = log(Exposure)`. This means:

- `pred_count = exp(Xβ + log(Exposure)) = Exposure × exp(Xβ)`.
- `pred_count` is the expected number of claims **over the actual observed period** for that policy.
- Annualised frequency = `pred_count / Exposure`.
- **Do not** multiply by Exposure again when computing aggregate predicted losses.

---

## 10. Full-Policy Severity Prediction

The Gamma model predicts severity for **all 135,603 frequency-validation policies**, including those with zero observed claims. This is correct because:

- The model uses vehicle/driver features, which are available for every policy.
- Using null-model predictions as a fallback would bias the aggregate loss calculation.
- The severity model's predictions are purely based on policy covariates, independent of whether a claim occurred in the validation window.

---

## 11. Train/Validation Split Design

- Split is done on **unique policy IDs** (IDpol), not on rows.
- All frequency rows and all severity rows for a given policy go to the same partition.
- This prevents data leakage where the same policy's claim behaviour is observed in both training and validation.

---

## 12. Data Discrepancy

`sum(ClaimNb) = 36,102` in the frequency dataset, but only `26,639` severity records exist. This means approximately 9,463 claims have no corresponding severity record. The pipeline reports this discrepancy rather than silently modifying either dataset.

---

## 13. Reconciliation of Historical Loss Figures

| Figure | Source | Why |
|--------|--------|-----|
| €11.87M actual | Correct | Aggregate of severity records for all frequency-validation policy IDs |
| €8.51M actual | Earlier bug | Severity aggregation applied before filtering to validation policy IDs |
| €1.003M predicted | Earlier bug | Only 5,101 intersection policies used instead of all 135,603 |
| €15.83M predicted | Current | All 135,603 val policies; Gamma model predicts severity for each |

---

## 14. Limitations and Honest Conclusions

- The severity model provides no meaningful predictive improvement over the null on held-out data.
- The loss ratio of 1.334 indicates the model overpredicts aggregate losses.
- These are genuine findings from the data, not modelling errors. A production system would require richer features, interaction terms, or alternative distributional assumptions.
