from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import os
import pandas as pd

from src.generate_data import generate_student_dataset
from src.pipeline import validate_and_clean_data
from src.regression import run_regression_pipeline
from src.classification import run_classification_pipeline
from src.clustering import run_clustering_pipeline

app = FastAPI(title="Musanze Harvest Intelligence Service", version="1.0.0")

MODELS_DIR = "models"
DATA_FILE = "data/AI_A2_2527585.csv"

preprocessor = reg_model = clf_model = kmeans = None


def load_models():
    global preprocessor, reg_model, clf_model, kmeans

    required_files = [
        os.path.join(MODELS_DIR, "preprocessor.pkl"),
        os.path.join(MODELS_DIR, "best_regressor.pkl"),
        os.path.join(MODELS_DIR, "best_classifier.pkl"),
    ]

    if not all(os.path.exists(path) for path in required_files):
        os.makedirs(MODELS_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)

        if not os.path.exists(DATA_FILE):
            generate_student_dataset(seed=7585, output_csv=DATA_FILE)

        X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor, df_clean = validate_and_clean_data(
            DATA_FILE, seed=7585
        )
        reg_model, _ = run_regression_pipeline(X_train, X_test, y_reg_train, y_reg_test, preprocessor, "artifacts", MODELS_DIR)
        clf_model, _ = run_classification_pipeline(X_train, X_test, y_clf_train, y_clf_test, preprocessor, "artifacts", MODELS_DIR)
        _, _ = run_clustering_pipeline(df_clean, preprocessor, "artifacts", MODELS_DIR)

    preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
    reg_model = joblib.load(os.path.join(MODELS_DIR, "best_regressor.pkl"))
    clf_model = joblib.load(os.path.join(MODELS_DIR, "best_classifier.pkl"))

    kmeans_path = os.path.join(MODELS_DIR, "kmeans_model.pkl")
    if os.path.exists(kmeans_path):
        kmeans = joblib.load(kmeans_path)
    else:
        kmeans = None


load_models()


class FarmRecord(BaseModel):
    rainfall_mm: float = Field(..., ge=600, le=1400)
    avg_temp_c: float = Field(..., ge=12, le=26)
    soil_moisture_pct: float = Field(..., ge=20, le=80)
    fertilizer_kg: float = Field(..., ge=0, le=250)
    field_size_ha: float = Field(..., ge=0.2, le=4.0)
    season: str
    cooperative_support: int = Field(..., ge=0, le=1)


@app.get("/health")
def health_check():
    return {"status": "healthy", "verification_code": "AI-A2-7585"}


@app.get("/model-info")
def model_info():
    return {"verification_code": "AI-A2-7585", "student_name": "Joel Eliezer Youto Mongar Jr", "registration_number": "25/27585"}


@app.post("/predict")
def predict_endpoint(record: FarmRecord):
    if not all([preprocessor, reg_model, clf_model]):
        raise HTTPException(status_code=500, detail="Models not loaded.")

    df_input = pd.DataFrame([record.dict()])
    payload = {"verification_code": "AI-A2-7585", "yield_estimate_tons": round(float(reg_model.predict(df_input)[0]), 4), "high_risk_prediction": int(clf_model.predict(df_input)[0])}

    if kmeans is not None:
        X_trans = preprocessor.transform(df_input)
        payload["cluster_assigned"] = int(kmeans.predict(X_trans)[0])

    return payload
