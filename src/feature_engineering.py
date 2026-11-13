"""
feature_engineering.py
-----------------------
Leakage exclusion, raw type cleaning, and derived feature creation for the
Loan Default Prediction project.

IMPORTANT - DATA LEAKAGE (see project brief, Section 7)
---------------------------------------------------------
This dataset is a LendingClub-style loan performance export. Several columns
only become known AFTER a loan has been issued and, in some cases, only
AFTER it has already defaulted. These are documented below and excluded from
the feature set. Removing them is not optional -- keeping them would let the
model "cheat" by looking at the outcome of the loan to predict the outcome
of the loan.

Excluded (post-origination / outcome-leakage) columns and why:
    out_prncp, out_prncp_inv        - remaining principal owed; only exists
                                       while/after the loan is being repaid,
                                       and is constant (0) in this extract,
                                       so it is also a constant column.
    total_pymnt, total_pymnt_inv    - total amount repaid to date; directly
                                       reflects loan performance.
    total_rec_prncp                 - principal repaid to date; performance.
    total_rec_int                   - interest repaid to date; performance.
    total_rec_late_fee              - late fees only accrue after payment
                                       problems begin.
    recoveries                      - money recovered after a charge-off /
                                       default. Directly encodes the target
                                       (verified: 0% of non-defaults vs 75%
                                       of defaults have recoveries > 0).
    collection_recovery_fee         - fee tied to the recovery process above.
    last_pymnt_d, last_pymnt_amnt   - date/amount of the most recent payment;
                                       only known as the loan matures/fails.
    last_credit_pull_d              - date of the most recent credit report
                                       pull, performed periodically during
                                       the life of the loan (post-origination
                                       monitoring), not at application time.

Excluded (identifier / non-predictive) columns and why:
    id, member_id   - unique identifiers, no predictive signal, id is kept
                      aside for submission alignment only.
    url             - unique per-loan URL that embeds the loan id; no signal.
    zip_code        - masked to 3 digits (e.g. "330xx") with 800+ distinct
                      values; addr_state already captures geography at a
                      modeling-appropriate granularity.
    title           - free-text loan nickname chosen by the borrower;
                      redundant with the clean categorical `purpose` field.
    emp_title       - free-text job title with >20,000 unique values; too
                      high-cardinality to encode safely and prone to noise.
    desc            - free-text borrower description. Kept only indirectly
                      via an engineered `has_desc` flag (see below); the raw
                      text is dropped.
"""

import numpy as np
import pandas as pd

from src.utils import safe_divide

LEAKAGE_COLUMNS = [
    "out_prncp",
    "out_prncp_inv",
    "total_pymnt",
    "total_pymnt_inv",
    "total_rec_prncp",
    "total_rec_int",
    "total_rec_late_fee",
    "recoveries",
    "collection_recovery_fee",
    "last_pymnt_d",
    "last_pymnt_amnt",
    "last_credit_pull_d",
]

NON_PREDICTIVE_COLUMNS = [
    "member_id",
    "url",
    "zip_code",
    "title",
    "emp_title",
    "desc",
]

EMP_LENGTH_MAP = {
    "< 1 year": 0,
    "1 year": 1,
    "2 years": 2,
    "3 years": 3,
    "4 years": 4,
    "5 years": 5,
    "6 years": 6,
    "7 years": 7,
    "8 years": 8,
    "9 years": 9,
    "10+ years": 10,
}


def _clean_percent_string(series: pd.Series) -> pd.Series:
    """Convert strings like '13.23%' -> 13.23 (float)."""
    return pd.to_numeric(series.astype(str).str.replace("%", "", regex=False).str.strip(), errors="coerce")


def _clean_term_string(series: pd.Series) -> pd.Series:
    """Convert strings like ' 36 months' -> 36 (int)."""
    return pd.to_numeric(series.astype(str).str.replace("months", "", regex=False).str.strip(), errors="coerce")


def clean_raw_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix columns that are numeric in nature but stored as text due to
    formatting (percent signs, "months" suffix, employment-length buckets,
    date strings).
    """
    df = df.copy()

    if "term" in df.columns:
        df["term"] = _clean_term_string(df["term"])

    if "int_rate" in df.columns:
        df["int_rate"] = _clean_percent_string(df["int_rate"])

    if "revol_util" in df.columns:
        df["revol_util"] = _clean_percent_string(df["revol_util"])

    if "emp_length" in df.columns:
        df["emp_length"] = df["emp_length"].map(EMP_LENGTH_MAP)

    # Dates are stored as "Mon-YY" (e.g. "Jul-10"). Parse to datetime.
    for date_col in ["issue_d", "earliest_cr_line"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], format="%b-%y", errors="coerce")

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features from the cleaned dataframe. Every feature here
    only uses information available at loan application / origination time.
    """
    df = df.copy()

    # --- Free-text presence flag (keeps a weak signal from `desc` without
    #     using its raw high-cardinality text content) ---
    if "desc" in df.columns:
        df["has_desc"] = df["desc"].notna().astype(int)

    # --- Ratio features (safe division handles zero/NaN denominators) ---
    if {"loan_amnt", "annual_inc"}.issubset(df.columns):
        df["loan_to_income"] = safe_divide(df["loan_amnt"].values, df["annual_inc"].values)

    if {"installment", "annual_inc"}.issubset(df.columns):
        df["installment_to_income"] = safe_divide(df["installment"].values, df["annual_inc"].values)

    if {"revol_bal", "annual_inc"}.issubset(df.columns):
        df["revol_bal_to_income"] = safe_divide(df["revol_bal"].values, df["annual_inc"].values)

    if {"loan_amnt", "installment"}.issubset(df.columns):
        # Rough measure of remaining term implied by installment size.
        df["loan_amnt_to_installment"] = safe_divide(df["loan_amnt"].values, df["installment"].values)

    # --- Credit history length in years, derived from two dates ---
    if {"issue_d", "earliest_cr_line"}.issubset(df.columns):
        days = (df["issue_d"] - df["earliest_cr_line"]).dt.days
        df["credit_history_length_years"] = (days / 365.25).clip(lower=0)

    # --- Date part features (year/month of loan issuance) ---
    if "issue_d" in df.columns:
        df["issue_year"] = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month

    # --- Missingness flags for sparse "since last X" columns, then encode
    #     the missing value as "never happened" (a large sentinel) rather
    #     than imputing with a misleading median. ---
    for col in ["mths_since_last_delinq", "mths_since_last_record"]:
        if col in df.columns:
            flag_col = f"{col}_missing"
            df[flag_col] = df[col].isna().astype(int)
            df[col] = df[col].fillna(-1)

    # Drop the now-redundant raw date columns (their signal is captured in
    # the engineered features above); keep issue_d out of the model matrix.
    for col in ["issue_d", "earliest_cr_line"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df


def drop_leakage_and_non_predictive(df: pd.DataFrame, id_col: str, target_col: str = None) -> pd.DataFrame:
    """
    Drop columns identified as leakage or non-predictive. `id_col` and
    `target_col` (if present) are preserved.
    """
    cols_to_drop = [
        c for c in (LEAKAGE_COLUMNS + NON_PREDICTIVE_COLUMNS)
        if c in df.columns and c not in (id_col, target_col)
    ]
    return df.drop(columns=cols_to_drop)


def full_feature_pipeline(df: pd.DataFrame, id_col: str, target_col: str = None) -> pd.DataFrame:
    """Run the full sequence: type cleaning -> feature engineering -> leakage removal."""
    df = clean_raw_types(df)
    df = engineer_features(df)
    df = drop_leakage_and_non_predictive(df, id_col=id_col, target_col=target_col)
    return df
