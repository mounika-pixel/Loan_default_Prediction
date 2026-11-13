"""
predict.py
----------
Apply a trained pipeline to new (test) data and produce:
  - outputs/submission.csv          (matches loan_sample_submission.csv format)
  - outputs/test_predictions.csv    (user-friendly: ID, class, probability, risk level)
"""

import pandas as pd


def risk_level_from_probability(prob: float) -> str:
    if prob < 0.30:
        return "Low"
    elif prob < 0.60:
        return "Medium"
    else:
        return "High"


def generate_predictions(pipeline, X_test: pd.DataFrame, ids: pd.Series, threshold: float = 0.5):
    """
    Run the fitted pipeline on X_test and return a DataFrame with id,
    predicted probability, predicted class, and risk level.
    """
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    results = pd.DataFrame({
        "id": ids.values,
        "Default_Probability": probabilities,
        "Predicted_Class": predictions,
    })
    results["Risk_Level"] = results["Default_Probability"].apply(risk_level_from_probability)
    return results


def build_submission_file(results: pd.DataFrame, sample_submission: pd.DataFrame, id_col: str, target_col: str) -> pd.DataFrame:
    """
    Build the Kaggle-style submission file, reproducing the exact column
    names and row order of the sample submission file.
    """
    pred_map = dict(zip(results["id"], results["Predicted_Class"]))

    submission = sample_submission.copy()
    submission[target_col] = submission[id_col].map(pred_map)

    # Validation checks (Section 18)
    assert submission.shape[0] == sample_submission.shape[0], "Row count mismatch with sample submission!"
    assert list(submission.columns) == list(sample_submission.columns), "Column mismatch with sample submission!"
    assert submission[target_col].isnull().sum() == 0, "Missing predictions found in submission!"
    assert set(submission[target_col].unique()).issubset({0, 1}), "Unexpected prediction values found!"

    return submission


def build_user_friendly_predictions(results: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Build the human-readable test_predictions.csv file."""
    out = results.rename(columns={"id": id_col}).copy()
    out["Predicted_Class"] = out["Predicted_Class"].map({0: "Unlikely to Default", 1: "Likely to Default"})
    out["Default_Probability"] = (out["Default_Probability"] * 100).round(2)
    return out[[id_col, "Predicted_Class", "Default_Probability", "Risk_Level"]]
