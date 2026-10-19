import os
import json
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import joblib

def run_regression_pipeline(X_train, X_test, y_train, y_test, preprocessor, artifacts_dir, models_dir):
    rf_pipe = Pipeline([('preprocessor', preprocessor), ('model', RandomForestRegressor(n_estimators=100, random_state=7585))])
    rf_pipe.fit(X_train, y_train.values.ravel() if hasattr(y_train, 'values') else y_train)
    y_preds = rf_pipe.predict(X_test)
    
    metrics = {
        "mae": float(mean_absolute_error(y_test, y_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_preds))),
        "r2": float(r2_score(y_test, y_preds))
    }
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "regression_metrics.json"), "w") as f:
        json.dump({"selected_model": "RandomForest", "metrics": metrics}, f, indent=4)
    joblib.dump(rf_pipe, os.path.join(models_dir, "best_regressor.pkl"))
    return rf_pipe, y_preds
