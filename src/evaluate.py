"""
evaluate.py
-----------
Evaluation utilities: classification metrics, confusion matrix, ROC / PR
curves, model comparison table, and probability-threshold optimization.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Compute the standard classification metrics for one model."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_proba),
        "PR-AUC": average_precision_score(y_true, y_proba),
    }


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    results: {model_name: metrics_dict}
    Returns a DataFrame sorted by ROC-AUC (primary metric, see Section 14).
    """
    table = pd.DataFrame(results).T
    table = table.sort_values("ROC-AUC", ascending=False)
    return table


def plot_confusion_matrix(y_true, y_pred, model_name: str, out_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Default", "Default"])
    ax.set_yticklabels(["No Default", "Default"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return cm


def plot_roc_curves(y_true_dict: dict, y_proba_dict: dict, out_path: str):
    """y_true_dict / y_proba_dict: {model_name: array}"""
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in y_proba_dict:
        fpr, tpr, _ = roc_curve(y_true_dict[name], y_proba_dict[name])
        auc = roc_auc_score(y_true_dict[name], y_proba_dict[name])
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pr_curves(y_true_dict: dict, y_proba_dict: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in y_proba_dict:
        precision, recall, _ = precision_recall_curve(y_true_dict[name], y_proba_dict[name])
        ap = average_precision_score(y_true_dict[name], y_proba_dict[name])
        ax.plot(recall, precision, label=f"{name} (AP={ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve Comparison")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def optimize_threshold(y_true, y_proba, priority: str = "recall", min_precision: float = 0.30) -> dict:
    """
    Sweep thresholds from 0.05 to 0.95 and select the one that best matches
    the stated objective: for loan default prediction, missing a defaulter
    (false negative) is typically costlier than a false alarm, so we
    prioritize recall while keeping precision above a reasonable floor.

    Returns a dict with the chosen threshold and the metrics at every
    threshold tested (for transparency / documentation).
    """
    thresholds = np.arange(0.05, 0.96, 0.01)
    rows = []
    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        rows.append({
            "threshold": round(t, 2),
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "f1": f1_score(y_true, preds, zero_division=0),
        })
    sweep = pd.DataFrame(rows)

    if priority == "recall":
        candidates = sweep[sweep["precision"] >= min_precision]
        if candidates.empty:
            candidates = sweep
        best_row = candidates.sort_values("recall", ascending=False).iloc[0]
    elif priority == "f1":
        best_row = sweep.sort_values("f1", ascending=False).iloc[0]
    else:  # default to 0.5
        best_row = sweep.iloc[(sweep["threshold"] - 0.5).abs().argmin()]

    return {
        "chosen_threshold": float(best_row["threshold"]),
        "precision_at_threshold": float(best_row["precision"]),
        "recall_at_threshold": float(best_row["recall"]),
        "f1_at_threshold": float(best_row["f1"]),
        "sweep_table": sweep,
    }


def get_feature_importance(pipeline, top_n: int = 20) -> pd.DataFrame:
    """
    Extract feature importance from a fitted Pipeline (preprocessor + model),
    mapping one-hot-encoded feature names back to readable names.
    """
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"feature_{i}" for i in range(model.n_features_in_)]

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).ravel()
    else:
        return pd.DataFrame(columns=["feature", "importance"])

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    df = df.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
    return df


def plot_feature_importance(importance_df: pd.DataFrame, model_name: str, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(importance_df["feature"][::-1], importance_df["importance"][::-1], color="#4C72B0")
    ax.set_title(f"Top Feature Importances - {model_name}")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
