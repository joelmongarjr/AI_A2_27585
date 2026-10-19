import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import joblib

def run_classification_pipeline(X_train, X_test, y_train, y_test, preprocessor, artifacts_dir, models_dir):
    rf_pipe = Pipeline([('preprocessor', preprocessor), ('model', RandomForestClassifier(n_estimators=100, random_state=7585))])
    rf_pipe.fit(X_train, y_train.values.ravel() if hasattr(y_train, 'values') else y_train)
    y_preds = rf_pipe.predict(X_test)
    
    metrics = {
        "precision": float(precision_score(y_test, y_preds, zero_division=0)),
        "recall": float(recall_score(y_test, y_preds, zero_division=0)),
        "f1": float(f1_score(y_test, y_preds, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_preds).tolist()
    }
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "classification_metrics.json"), "w") as f:
        json.dump({"selected_model": "RandomForest", "metrics": metrics}, f, indent=4)
    joblib.dump(rf_pipe, os.path.join(models_dir, "best_classifier.pkl"))
    return rf_pipe, y_preds
