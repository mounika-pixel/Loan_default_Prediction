"""
data_loader.py
--------------
Handles loading the raw CSV files and performing the initial, dataset-agnostic
inspection required before any modeling assumptions are made (see project
brief, Section 2: never assume column names / target / schema in advance).
"""

import os
import pandas as pd

from src.utils import path_from_root, print_section

TRAIN_FILENAME = "loan_train.csv"
TEST_FILENAME = "loan_test.csv"
SUBMISSION_FILENAME = "loan_sample_submission.csv"


def load_raw_data(data_dir: str = None):
    """
    Load train, test, and sample submission CSVs from data/raw.

    Returns
    -------
    train, test, sample_submission : pd.DataFrame
    """
    if data_dir is None:
        data_dir = path_from_root("data", "raw")

    available_files = os.listdir(data_dir)
    print("Files found in data/raw:", available_files)

    train_path = os.path.join(data_dir, TRAIN_FILENAME)
    test_path = os.path.join(data_dir, TEST_FILENAME)
    sub_path = os.path.join(data_dir, SUBMISSION_FILENAME)

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    sample_submission = pd.read_csv(sub_path) if os.path.exists(sub_path) else None

    return train, test, sample_submission


def identify_target_and_id(train: pd.DataFrame, test: pd.DataFrame, sample_submission: pd.DataFrame):
    """
    Identify the target column and the ID column by comparing train/test
    schemas and inspecting the sample submission file, instead of assuming
    fixed names.

    Returns
    -------
    target_col : str
    id_col : str
    """
    train_only_cols = set(train.columns) - set(test.columns)

    id_col = None
    target_col = None

    # ID column: present in both train and test, and present in sample submission.
    if sample_submission is not None:
        common_with_sub = [c for c in train.columns if c in sample_submission.columns and c in test.columns]
        if common_with_sub:
            id_col = common_with_sub[0]

        # Target column: appears in sample submission but is NOT the id column.
        sub_target_candidates = [c for c in sample_submission.columns if c != id_col]
        if sub_target_candidates:
            candidate = sub_target_candidates[0]
            if candidate in train_only_cols:
                target_col = candidate

    # Fallbacks if sample submission was not conclusive.
    if id_col is None:
        id_candidates = [c for c in train.columns if c.lower() in ("id", "loan_id", "member_id")]
        id_col = id_candidates[0] if id_candidates else train.columns[0]

    if target_col is None:
        if len(train_only_cols) == 1:
            target_col = list(train_only_cols)[0]
        else:
            raise ValueError(
                f"Could not confidently identify the target column. "
                f"Columns present only in train: {train_only_cols}. "
                f"Refusing to fabricate a target column — please verify manually."
            )

    return target_col, id_col


def basic_inspection_report(train: pd.DataFrame, test: pd.DataFrame, target_col: str, id_col: str) -> dict:
    """
    Print and return a dictionary summarizing the dataset understanding step
    (shape, dtypes, missing values, duplicates, schema comparison, target
    distribution, etc.) as required by the project brief, Section 2 and 5.
    """
    print_section("DATASET SHAPES")
    print("Train shape:", train.shape)
    print("Test shape :", test.shape)

    print_section("TRAIN / TEST SCHEMA COMPARISON")
    train_only = set(train.columns) - set(test.columns)
    test_only = set(test.columns) - set(train.columns)
    print("Columns only in train:", train_only)
    print("Columns only in test :", test_only)

    print_section("TARGET / ID")
    print("Target column:", target_col)
    print("ID column    :", id_col)

    print_section("TARGET DISTRIBUTION (train)")
    dist = train[target_col].value_counts(dropna=False)
    pct = train[target_col].value_counts(normalize=True, dropna=False) * 100
    print(pd.DataFrame({"count": dist, "percent": pct.round(2)}))

    print_section("DATA TYPES")
    print(train.dtypes)

    print_section("MISSING VALUES (train, % of rows)")
    miss = train.isnull().mean().sort_values(ascending=False) * 100
    print(miss[miss > 0].round(2))

    print_section("DUPLICATE ROWS")
    print("Fully duplicated rows in train:", train.duplicated().sum())
    if id_col in train.columns:
        print(f"Duplicated '{id_col}' values in train:", train[id_col].duplicated().sum())

    numeric_cols = train.select_dtypes(include="number").columns.tolist()
    categorical_cols = train.select_dtypes(exclude="number").columns.tolist()

    print_section("NUMERICAL COLUMNS")
    print(numeric_cols)

    print_section("CATEGORICAL COLUMNS")
    print(categorical_cols)

    constant_cols = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
    print_section("CONSTANT COLUMNS")
    print(constant_cols)

    report = {
        "train_shape": train.shape,
        "test_shape": test.shape,
        "train_only_cols": train_only,
        "test_only_cols": test_only,
        "target_col": target_col,
        "id_col": id_col,
        "target_distribution": dist.to_dict(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "constant_cols": constant_cols,
        "missing_pct": miss.to_dict(),
    }
    return report
