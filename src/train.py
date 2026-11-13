"""
train.py
--------
Defines the candidate models and the training / hyperparameter-tuning logic
for the Loan Default Prediction project.
"""

import warnings
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor

RANDOM_STATE = 42

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False


def get_candidate_models(scale_pos_weight: float = 1.0) -> dict:
    """
    Return a dict of {model_name: (estimator, needs_scaling)}.

    scale_pos_weight is used by XGBoost/LightGBM to handle class imbalance;
    other models use class_weight="balanced" instead.
    """
    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            True,
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE, max_depth=8),
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            False,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = (
            XGBClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
            False,
        )
    else:
        warnings.warn("XGBoost not available - skipping this model.")

    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = (
            LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=-1,
                subsample=0.8,
                colsample_bytree=0.8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            ),
            False,
        )
    else:
        warnings.warn("LightGBM not available - skipping this model.")

    return models


def build_full_pipeline(estimator, numeric_cols, categorical_cols, scale_numeric: bool) -> Pipeline:
    """Wrap a preprocessing ColumnTransformer + estimator into a single sklearn Pipeline."""
    preprocessor = build_preprocessor(numeric_cols, categorical_cols, scale_numeric=scale_numeric)
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def train_all_models(X_train, y_train, numeric_cols, categorical_cols, scale_pos_weight: float = 1.0) -> dict:
    """
    Fit every candidate model on the training fold.

    Returns
    -------
    dict of {model_name: fitted Pipeline}
    """
    candidates = get_candidate_models(scale_pos_weight=scale_pos_weight)
    fitted = {}
    for name, (estimator, needs_scaling) in candidates.items():
        pipeline = build_full_pipeline(estimator, numeric_cols, categorical_cols, scale_numeric=needs_scaling)
        pipeline.fit(X_train, y_train)
        fitted[name] = pipeline
    return fitted


def tune_best_model(best_name, X_train, y_train, numeric_cols, categorical_cols, n_iter: int = 15):
    """
    Run a bounded RandomizedSearchCV over the strongest model family found in
    the baseline comparison. Keeps the search small/fast per the project
    brief (Section 15: "do not perform extremely large searches").
    """
    param_grids = {
        "Random Forest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [6, 10, 14, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2"],
        },
        "XGBoost": {
            "model__n_estimators": [200, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        },
        "LightGBM": {
            "model__n_estimators": [200, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__subsample": [0.7, 0.8, 1.0],
            "model__colsample_bytree": [0.7, 0.8, 1.0],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
        "Logistic Regression": {
            "model__C": [0.01, 0.1, 1, 10],
        },
        "Decision Tree": {
            "model__max_depth": [4, 6, 8, 12, None],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },
    }

    candidates = get_candidate_models()
    estimator, needs_scaling = candidates[best_name]
    pipeline = build_full_pipeline(estimator, numeric_cols, categorical_cols, scale_numeric=needs_scaling)

    grid = param_grids.get(best_name, {})
    if not grid:
        pipeline.fit(X_train, y_train)
        return pipeline

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=grid,
        n_iter=min(n_iter, int(np.prod([len(v) for v in grid.values()]))),
        scoring="roc_auc",
        cv=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_
