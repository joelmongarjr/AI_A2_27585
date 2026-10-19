import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def validate_and_clean_data(csv_path, seed=7585):
    df = pd.read_csv(csv_path).drop_duplicates()
    features = ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "season", "cooperative_support"]
    X = df[features]
    y_reg = df["yield_tons"]
    y_clf = df["high_risk"]
    
    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X, y_reg, y_clf, test_size=0.2, random_state=seed, stratify=y_clf
    )
    
    preprocessor = ColumnTransformer(transformers=[
        ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), 
         ["rainfall_mm", "avg_temp_c", "soil_moisture_pct", "fertilizer_kg", "field_size_ha", "cooperative_support"]),
        ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), 
         ["season"])
    ])
    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test, preprocessor, df
