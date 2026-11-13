"""
main.py
-------
End-to-end execution script for the Loan Default Prediction project.

Run with:
    python main.py

Steps:
    1. Load raw data
    2. Inspect data (target/id identification, schema checks)
    3. Feature engineering + leakage removal
    4. Train / validation split
    5. Train baseline models
    6. Evaluate & compare models
    7. Tune the best model
    8. Optimize decision threshold
    9. Feature importance
    10. Refit final pipeline on full training data
    11. Save model + metadata
    12. Generate test predictions + submission.csv
"""

import json
import warnings
from datetime import datetime

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.utils import set_seed, path_from_root, print_section, RANDOM_STATE
from src.data_loader import load_raw_data, identify_target_and_id, basic_inspection_report
from src.feature_engineering import full_feature_pipeline
from src.preprocessing import get_feature_types
from src.train import train_all_models, tune_best_model, build_full_pipeline, get_candidate_models
from src.evaluate import (
    compute_metrics,
    build_comparison_table,
    plot_confusion_matrix,
    plot_roc_curves,
    plot_pr_curves,
    optimize_threshold,
    get_feature_importance,
    plot_feature_importance,
)
from src.predict import generate_predictions, build_submission_file, build_user_friendly_predictions

warnings.filterwarnings("ignore")


def main():
    set_seed(RANDOM_STATE)

    # ------------------------------------------------------------------
    # 1-2. Load + inspect data
    # ------------------------------------------------------------------
    print_section("STEP 1-2: LOAD & INSPECT DATA")
    train_raw, test_raw, sample_submission = load_raw_data()
    target_col, id_col = identify_target_and_id(train_raw, test_raw, sample_submission)
    report = basic_inspection_report(train_raw, test_raw, target_col, id_col)

    # ------------------------------------------------------------------
    # 3. Feature engineering + leakage removal
    # ------------------------------------------------------------------
    print_section("STEP 3: FEATURE ENGINEERING & LEAKAGE REMOVAL")
    train_fe = full_feature_pipeline(train_raw, id_col=id_col, target_col=target_col)
    test_fe = full_feature_pipeline(test_raw, id_col=id_col, target_col=None)
    print("Train shape after feature engineering:", train_fe.shape)
    print("Test shape after feature engineering :", test_fe.shape)

    numeric_cols, categorical_cols = get_feature_types(train_fe, id_col=id_col, target_col=target_col)
    print(f"Numeric features ({len(numeric_cols)}):", numeric_cols)
    print(f"Categorical features ({len(categorical_cols)}):", categorical_cols)

    # ------------------------------------------------------------------
    # 4. Train / validation split (stratified, before any fitting)
    # ------------------------------------------------------------------
    print_section("STEP 4: TRAIN / VALIDATION SPLIT")
    feature_cols = numeric_cols + categorical_cols
    X = train_fe[feature_cols]
    y = train_fe[target_col].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print("X_train:", X_train.shape, " X_val:", X_val.shape)

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"Class imbalance ratio (neg/pos) used for scale_pos_weight: {scale_pos_weight:.3f}")

    # ------------------------------------------------------------------
    # 5-6. Train baseline models & evaluate
    # ------------------------------------------------------------------
    print_section("STEP 5-6: TRAIN & EVALUATE BASELINE MODELS")
    fitted_models = train_all_models(X_train, y_train, numeric_cols, categorical_cols, scale_pos_weight)

    metrics_results = {}
    proba_val = {}
    pred_val = {}
    for name, pipeline in fitted_models.items():
        proba = pipeline.predict_proba(X_val)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics_results[name] = compute_metrics(y_val, preds, proba)
        proba_val[name] = proba
        pred_val[name] = preds
        print(f"{name:20s} ROC-AUC={metrics_results[name]['ROC-AUC']:.4f}  "
              f"F1={metrics_results[name]['F1']:.4f}  Recall={metrics_results[name]['Recall']:.4f}")

    comparison_table = build_comparison_table(metrics_results)
    print_section("MODEL COMPARISON TABLE (sorted by ROC-AUC)")
    print(comparison_table.round(4))
    comparison_table.round(4).to_csv(path_from_root("outputs", "metrics", "model_comparison.csv"))

    best_model_name = comparison_table.index[0]
    print(f"\nBest baseline model: {best_model_name}")

    # ROC / PR curve plots across all models
    y_true_dict = {name: y_val for name in fitted_models}
    plot_roc_curves(y_true_dict, proba_val, path_from_root("outputs", "figures", "roc_curve_comparison.png"))
    plot_pr_curves(y_true_dict, proba_val, path_from_root("outputs", "figures", "pr_curve_comparison.png"))
    plot_confusion_matrix(y_val, pred_val[best_model_name], best_model_name,
                           path_from_root("outputs", "figures", "confusion_matrix_best_model.png"))

    # ------------------------------------------------------------------
    # 7. Hyperparameter tuning of the best model
    # ------------------------------------------------------------------
    print_section(f"STEP 7: HYPERPARAMETER TUNING - {best_model_name}")
    tuned_pipeline = tune_best_model(best_model_name, X_train, y_train, numeric_cols, categorical_cols)
    tuned_proba_val = tuned_pipeline.predict_proba(X_val)[:, 1]
    tuned_preds_val = (tuned_proba_val >= 0.5).astype(int)
    tuned_metrics = compute_metrics(y_val, tuned_preds_val, tuned_proba_val)
    print("Tuned model validation metrics:", {k: round(v, 4) for k, v in tuned_metrics.items()})

    # Keep whichever is better: baseline or tuned, on ROC-AUC.
    if tuned_metrics["ROC-AUC"] >= metrics_results[best_model_name]["ROC-AUC"]:
        final_pipeline_template = tuned_pipeline
        final_val_proba = tuned_proba_val
        print("Using TUNED model as final model.")
    else:
        final_pipeline_template = fitted_models[best_model_name]
        final_val_proba = proba_val[best_model_name]
        print("Tuning did not improve ROC-AUC; using BASELINE model as final model.")

    # ------------------------------------------------------------------
    # 8. Threshold optimization
    # ------------------------------------------------------------------
    print_section("STEP 8: THRESHOLD OPTIMIZATION")
    threshold_result = optimize_threshold(y_val, final_val_proba, priority="recall", min_precision=0.30)
    print(f"Chosen threshold: {threshold_result['chosen_threshold']}")
    print(f"  Precision @ threshold: {threshold_result['precision_at_threshold']:.4f}")
    print(f"  Recall    @ threshold: {threshold_result['recall_at_threshold']:.4f}")
    print(f"  F1        @ threshold: {threshold_result['f1_at_threshold']:.4f}")
    threshold_result["sweep_table"].to_csv(path_from_root("outputs", "metrics", "threshold_sweep.csv"), index=False)

    final_threshold = threshold_result["chosen_threshold"]
    final_preds_val = (final_val_proba >= final_threshold).astype(int)
    final_metrics_at_threshold = compute_metrics(y_val, final_preds_val, final_val_proba)

    # ------------------------------------------------------------------
    # 9. Feature importance
    # ------------------------------------------------------------------
    print_section("STEP 9: FEATURE IMPORTANCE")
    importance_df = get_feature_importance(final_pipeline_template, top_n=20)
    if not importance_df.empty:
        print(importance_df.to_string(index=False))
        importance_df.to_csv(path_from_root("outputs", "metrics", "feature_importance.csv"), index=False)
        plot_feature_importance(importance_df, best_model_name,
                                 path_from_root("outputs", "figures", "feature_importance.png"))

    # ------------------------------------------------------------------
    # 10. Refit final pipeline on FULL training data
    # ------------------------------------------------------------------
    print_section("STEP 10: REFIT FINAL MODEL ON FULL TRAINING DATA")
    candidates = get_candidate_models(scale_pos_weight=scale_pos_weight)
    best_estimator_params = final_pipeline_template.named_steps["model"].get_params()
    fresh_estimator, needs_scaling = candidates.get(
        best_model_name, (final_pipeline_template.named_steps["model"], False)
    )
    fresh_estimator = fresh_estimator.__class__(**{
        k: v for k, v in best_estimator_params.items() if k in fresh_estimator.get_params()
    })
    final_pipeline = build_full_pipeline(fresh_estimator, numeric_cols, categorical_cols, scale_numeric=needs_scaling)
    final_pipeline.fit(X, y)
    print("Final pipeline refit on full training data:", X.shape)

    # ------------------------------------------------------------------
    # 11. Save model + metadata
    # ------------------------------------------------------------------
    print_section("STEP 11: SAVE MODEL")
    model_path = path_from_root("models", "best_model.pkl")
    joblib.dump(final_pipeline, model_path)
    print("Saved model to:", model_path)

    metadata = {
        "model_name": best_model_name,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features_used": numeric_cols + categorical_cols,
        "target_column": target_col,
        "id_column": id_col,
        "decision_threshold": final_threshold,
        "validation_metrics_default_threshold": {k: round(v, 4) for k, v in metrics_results[best_model_name].items()},
        "validation_metrics_at_chosen_threshold": {k: round(v, 4) for k, v in final_metrics_at_threshold.items()},
    }
    with open(path_from_root("models", "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Saved metadata to models/model_metadata.json")

    # ------------------------------------------------------------------
    # 12. Test predictions + submission
    # ------------------------------------------------------------------
    print_section("STEP 12: TEST PREDICTIONS & SUBMISSION")
    X_test = test_fe[feature_cols]
    test_ids = test_fe[id_col]

    results = generate_predictions(final_pipeline, X_test, test_ids, threshold=final_threshold)

    submission = build_submission_file(results, sample_submission, id_col=id_col, target_col=target_col)
    submission_path = path_from_root("outputs", "submission.csv")
    submission.to_csv(submission_path, index=False)
    print("Saved submission to:", submission_path)
    print("Submission preview:\n", submission.head())

    user_friendly = build_user_friendly_predictions(results, id_col=id_col)
    predictions_path = path_from_root("outputs", "test_predictions.csv")
    user_friendly.to_csv(predictions_path, index=False)
    print("Saved user-friendly predictions to:", predictions_path)

    print_section("PIPELINE COMPLETE")
    print(f"Best model: {best_model_name}")
    print(f"Validation ROC-AUC: {metrics_results[best_model_name]['ROC-AUC']:.4f}")
    print(f"Chosen decision threshold: {final_threshold}")


if __name__ == "__main__":
    main()
