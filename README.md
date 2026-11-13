<img width="1917" height="1013" alt="Screenshot 2026-09-14 013407" src="https://github.com/user-attachments/assets/21542807-b0f3-4de9-a0ae-ec56b5bcb3d7" />


# Loan Default Prediction

An end-to-end machine learning project that predicts whether a borrower is
likely to default on a loan, built on a LendingClub-style historical loan
dataset. The project follows a full Kaggle-style workflow — data
understanding → cleaning → EDA → feature engineering → preprocessing →
model training/comparison → hyperparameter tuning → threshold optimization →
final predictions — and ships with a Streamlit app for interactive risk
scoring.

## Overview

Lenders need to estimate the probability that a loan applicant will default
before extending credit. Getting this wrong in either direction is costly:
approving high-risk borrowers leads to losses, while rejecting good
borrowers means lost revenue and unhappy customers. This project builds a
classification model that estimates default probability from information
available **at the time of loan application** — deliberately excluding any
data that would only be known after the loan was issued.

## Problem Statement

Predict whether a borrower will default on a loan (`loan_status = 1`) or
repay it (`loan_status = 0`), based on borrower and loan-application
attributes, and produce well-calibrated default probabilities that can
support a risk-based lending decision.

## Objectives

- Analyze real loan application and performance data
- Identify and eliminate data leakage before modeling
- Engineer meaningful, non-leaky risk features
- Train and compare multiple classification algorithms
- Select the best model using metrics appropriate for imbalanced data
- Tune the strongest model and choose a business-appropriate decision threshold
- Generate Kaggle-style test predictions (`submission.csv`)
- Provide a simple, professional prediction interface

## Dataset

The dataset is a LendingClub-style loan-level export with **27,003 training
rows** and **11,574 test rows**, each with 46–47 raw columns covering loan
terms, borrower financial attributes, and (in the training data only)
post-origination loan performance.

- **Target column:** `loan_status` — binary, `0` = No Default, `1` = Default
- **ID column:** `id`
- **Class balance:** 23,026 non-default (85.3%) vs. 3,977 default (14.7%) — a meaningfully imbalanced problem
- **Train/test schema:** identical except that `loan_status` is (correctly) absent from the test set

### Data Leakage — the most important finding

This dataset bundles loan-performance fields alongside application-time
fields. Several of these only exist **after** a loan has matured or
defaulted, and would leak the answer directly into the model:

| Column(s) | Why it's leakage |
|---|---|
| `recoveries`, `collection_recovery_fee` | Money recovered after a charge-off. `recoveries > 0` for ~75% of defaults vs. 0% of non-defaults — this alone nearly gives away the label. |
| `total_pymnt`, `total_pymnt_inv`, `total_rec_prncp`, `total_rec_int`, `total_rec_late_fee` | Cumulative repayment figures that directly reflect how the loan actually performed. |
| `out_prncp`, `out_prncp_inv` | Remaining principal owed; also constant (always 0) in this extract. |
| `last_pymnt_d`, `last_pymnt_amnt` | Date/amount of the most recent payment — only known as the loan matures. |
| `last_credit_pull_d` | Date of a credit check performed periodically *during* the loan's life, not at application. |

All of these are excluded from modeling, with the full reasoning documented
in `src/feature_engineering.py`. Non-predictive identifiers and free text
(`id`, `member_id`, `url`, `zip_code`, `title`, `emp_title`, raw `desc`) are
also excluded or reduced to a simple derived signal (`has_desc`).

## Technologies

- Python, Pandas, NumPy
- Scikit-learn (pipelines, ColumnTransformer, model selection)
- XGBoost, LightGBM, Gradient Boosting, Random Forest, Decision Tree, Logistic Regression
- Matplotlib, Seaborn (EDA & evaluation plots)
- SHAP-compatible feature importance (permutation/tree importances)
- Joblib (model persistence)
- Streamlit (prediction app)

## ML Workflow

```text
Data Collection
    ↓
Data Cleaning (type parsing, constant-column removal, missing-value analysis)
    ↓
EDA (target distribution, numeric/categorical analysis, correlation, leakage audit)
    ↓
Feature Engineering (ratios, credit history length, missingness flags)
    ↓
Preprocessing (median/most-frequent imputation, one-hot encoding, scaling)
    ↓
Model Training (6 candidate algorithms)
    ↓
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
    ↓
Hyperparameter Tuning (RandomizedSearchCV on the best model)
    ↓
Threshold Optimization (recall-prioritized, precision-floored)
    ↓
Final Prediction & submission.csv
```

## Models

Six models were trained and compared under an identical preprocessing
pipeline: **Logistic Regression, Decision Tree, Random Forest, Gradient
Boosting, XGBoost, LightGBM**.

## Results

Validation set (20% stratified hold-out), default 0.5 threshold:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| **XGBoost** | 0.7217 | 0.2782 | 0.5585 | 0.3714 | **0.7184** | 0.3277 |
| Gradient Boosting | 0.8532 | 0.5278 | 0.0239 | 0.0457 | 0.7155 | 0.3147 |
| Logistic Regression | 0.6673 | 0.2505 | 0.6327 | 0.3589 | 0.7140 | 0.3154 |
| LightGBM | 0.7389 | 0.2857 | 0.5157 | 0.3677 | 0.7126 | 0.3123 |
| Random Forest | 0.8530 | 0.5455 | 0.0075 | 0.0149 | 0.7093 | 0.3001 |
| Decision Tree | 0.6212 | 0.2206 | 0.6214 | 0.3256 | 0.6507 | 0.2404 |

Note how misleading raw accuracy is here: Random Forest and Gradient
Boosting reach 85%+ accuracy largely by predicting "no default" almost
every time (Recall ≈ 1–2%), which is useless for actually catching
defaulters. This is exactly why ROC-AUC/PR-AUC/Recall are used as the
primary selection criteria (Section 14 of the project brief) rather than
accuracy.

After tuning, XGBoost's validation ROC-AUC was **0.7201** (vs. 0.7184
baseline). At the chosen decision threshold of **0.20** (rather than the
default 0.5), the model achieves:

| Metric | Value |
|---|---|
| Accuracy | 0.766 |
| Precision | 0.307 |
| Recall | 0.468 |
| F1 | 0.371 |

## Best Model

**XGBoost** was selected as the final model: it had the highest validation
ROC-AUC among all six candidates, improved further under a bounded
`RandomizedSearchCV`, and — unlike Random Forest/Gradient Boosting at the
default threshold — maintains a usable recall without collapsing to
near-zero. The full trained pipeline (preprocessing + model) is saved to
`models/best_model.pkl`, with training details in
`models/model_metadata.json`.

**Threshold:** The default 0.5 cutoff is not used. Because failing to flag
an actual defaulter (false negative) is typically costlier for a lender than
a false alarm, the threshold was chosen by sweeping 0.05–0.95 and selecting
the value that maximizes recall subject to a minimum 30% precision floor,
landing on **0.20**.

## Feature Importance

Top contributing features (XGBoost):

1. `grade_A` (LendingClub's own credit grade)
2. `term` (loan length in months)
3. `int_rate`
4. `loan_amnt_to_installment` (engineered)
5. `purpose_small_business`
6. `annual_inc`
7. `loan_to_income` (engineered)
8. `mths_since_last_record` / its missingness flag
9. `funded_amnt`, `loan_amnt`
10. `inq_last_6mths`

Full top-20 table: `outputs/metrics/feature_importance.csv`. Chart:
`outputs/figures/feature_importance.png`.

Feature importance shows association with the model's predictions, **not
causation** — `grade`/`sub_grade`, for instance, is itself derived from
LendingClub's own underwriting model.

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

`main.py` runs the complete pipeline: loads and inspects the data, engineers
features, trains and compares all six models, tunes the best one, optimizes
the decision threshold, computes feature importance, refits on full
training data, and writes `models/best_model.pkl`, `outputs/submission.csv`,
and `outputs/test_predictions.csv`.

For the interactive risk-scoring app:

```bash
streamlit run app/app.py
```

For the full narrative analysis with inline charts:

```bash
jupyter notebook notebooks/loan_default_analysis.ipynb
```

## Project Structure

```text
loan-default-prediction/
│
├── data/
│   ├── raw/
│   │   ├── loan_train.csv
│   │   ├── loan_test.csv
│   │   └── loan_sample_submission.csv
│   └── processed/
│
├── notebooks/
│   └── loan_default_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── best_model.pkl
│   └── model_metadata.json
│
├── outputs/
│   ├── figures/
│   ├── metrics/
│   ├── submission.csv
│   └── test_predictions.csv
│
├── app/
│   └── app.py
│
├── requirements.txt
├── README.md
├── .gitignore
└── main.py
```

## Future Improvements

- Better threshold optimization using an explicit cost matrix (expected loss per false negative vs. false positive)
- Explainable AI via full SHAP value analysis per prediction, surfaced in the app
- Model monitoring for feature/label drift as new loan vintages arrive
- A richer credit-risk dashboard (portfolio-level risk aggregation, cohort analysis)
- API deployment (FastAPI wrapper around the saved pipeline)
- Cloud deployment (containerized Streamlit app + model registry)
