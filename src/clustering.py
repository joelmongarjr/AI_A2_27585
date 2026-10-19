import os
import json
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

def run_clustering_pipeline(df, preprocessor, artifacts_dir, models_dir):
    features = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "cooperative_support"]
    X_trans = preprocessor.fit_transform(df[features])
    
    kmeans = KMeans(n_clusters=3, random_state=7585, n_init=10)
    labels = kmeans.fit_predict(X_trans)
    score = float(silhouette_score(X_trans, labels))
    
    os.makedirs(artifacts_dir, exist_ok=True)
    with open(os.path.join(artifacts_dir, "clustering_report.json"), "w") as f:
        json.dump({"selected_k": 3, "silhouette_score": score}, f, indent=4)
    joblib.dump(kmeans, os.path.join(models_dir, "kmeans_model.pkl"))
    return 3, labels
