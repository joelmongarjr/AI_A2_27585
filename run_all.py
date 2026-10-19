import argparse
import os
import pandas as pd
import joblib

from src.generate_data import generate_student_dataset
from src.pipeline import validate_and_clean_data
from src.regression import run_regression_pipeline
from src.classification import run_classification_pipeline
from src.clustering import run_clustering_pipeline


def main():
    parser = argparse.ArgumentParser(description="Musanze Harvest Intelligence Service Pipeline")
    parser.add_argument("--student-id", required=True, help="Institutional student email/id")
    parser.add_argument("--seed", type=int, required=True, help="Student seed (e.g., 7585)")

    args = parser.parse_args()

    if args.seed != 7585:
        raise ValueError("Invalid seed provided! Must match student registration digits: 7585.")

    print(f"=== Starting Pipeline for AI-A2-7585 (Student ID: {args.student_id}) ===")

    csv_path = "data/AI_A2_2527585.csv"
    artifacts_dir = "artifacts"
    models_dir = "models"

    os.makedirs("data", exist_ok=True)
    os.makedirs(artifacts_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    df, fingerprint = generate_student_dataset(seed=args.seed, output_csv=csv_path)
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor, df_clean = validate_and_clean_data(
        csv_path, seed=args.seed
    )

    joblib.dump(preprocessor, os.path.join(models_dir, "preprocessor.pkl"))

    reg_model, reg_preds = run_regression_pipeline(
        X_train, X_test, y_reg_train, y_reg_test, preprocessor, artifacts_dir, models_dir
    )
    clf_model, clf_preds = run_classification_pipeline(
        X_train, X_test, y_clf_train, y_clf_test, preprocessor, artifacts_dir, models_dir
    )
    best_k, cluster_labels = run_clustering_pipeline(
        df_clean, preprocessor, artifacts_dir, models_dir
    )

    test_results_df = pd.DataFrame({
        "actual_yield": y_reg_test.values,
        "predicted_yield": reg_preds,
        "actual_risk": y_clf_test.values,
        "predicted_risk": clf_preds,
    })
    test_results_df.to_csv(os.path.join(artifacts_dir, "test_predictions.csv"), index=False)

    print("=== Pipeline Execution Complete Successfully. ===")


if __name__ == "__main__":
    main()
