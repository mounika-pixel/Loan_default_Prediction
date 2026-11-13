"""
preprocessing.py
-----------------
Builds a reusable, leakage-safe sklearn preprocessing pipeline
(SimpleImputer + OneHotEncoder + StandardScaler wrapped in a
ColumnTransformer) that is fit only on the training fold and then applied
identically to validation/test data.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_types(df: pd.DataFrame, id_col: str, target_col: str = None):
    """
    Split feature columns (i.e. everything except id/target) into numerical
    and categorical groups based on dtype.
    """
    feature_cols = [c for c in df.columns if c not in (id_col, target_col)]
    numeric_cols = df[feature_cols].select_dtypes(include="number").columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_preprocessor(numeric_cols, categorical_cols, scale_numeric: bool = True) -> ColumnTransformer:
    """
    Build a ColumnTransformer:
        numeric:     median imputation (+ optional scaling)
        categorical: most-frequent imputation + one-hot encoding

    scale_numeric=False is useful for tree-based models, which do not
    require feature scaling; scale_numeric=True is used for linear models.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipeline = Pipeline(numeric_steps)

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor
