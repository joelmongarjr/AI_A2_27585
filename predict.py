import sys
import json
import os
import numpy as np
import pandas as pd
import joblib

def main():
    args = sys.argv[1:]
    json_path = None
    for i in range(len(args)):
        if args[i] == "--json" and i + 1 < len(args):
            json_path = args[i+1]
            
    if not json_path or not os.path.exists(json_path):
        print(json.dumps({"error": "Missing or invalid --json argument file path."}))
        sys.exit(1)
        
    with open(json_path, "r") as f:
        record = json.load(f)
        
    df_input = pd.DataFrame([record])
    models_dir = "models"
    
    preprocessor = joblib.load(os.path.join(models_dir, "preprocessor.pkl"))
    reg_model = joblib.load(os.path.join(models_dir, "best_regressor.pkl"))
    clf_model = joblib.load(os.path.join(models_dir, "best_classifier.pkl"))
    kmeans = joblib.load(os.path.join(models_dir, "kmeans_model.pkl"))
    
    X_trans = preprocessor.transform(df_input)
    yield_est = float(reg_model.predict(df_input)[0])
    risk_pred = int(clf_model.predict(df_input)[0])
    risk_prob = float(clf_model.predict_proba(df_input)[0][1])
    cluster_label = int(kmeans.predict(X_trans)[0])
    
    result = {
        "verification_code": "AI-A2-7585",
        "yield_estimate_tons": round(yield_est, 4),
        "high_risk_prediction": risk_pred,
        "high_risk_probability": round(risk_prob, 4),
        "cluster_assigned": cluster_label
    }
    print(json.dumps(result, indent=4))

if __name__ == "__main__":
    main()
