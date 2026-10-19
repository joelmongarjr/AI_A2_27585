import os
import hashlib
import numpy as np
import pandas as pd

def generate_student_dataset(seed=7585, output_csv="data/AI_A2_2527585.csv", fingerprint_txt="artifacts/data_fingerprint.txt"):
    rng = np.random.default_rng(seed)
    base_rows = 240 + (seed % 41)
    
    rainfall = rng.uniform(600, 1400, base_rows)
    avg_temp = rng.uniform(12, 26, base_rows)
    soil_moisture = rng.uniform(20, 80, base_rows)
    fertilizer = rng.uniform(0, 250, base_rows)
    field_size = rng.uniform(0.2, 4.0, base_rows)
    season = rng.choice(["A", "B"], size=base_rows)
    cooperative_support = rng.choice([0, 1], size=base_rows)
    
    season_effect = np.where(season == "A", 0.50, -0.20)
    noise = rng.normal(0, 0.80, base_rows)
    
    yield_tons = (
        0.004 * rainfall + 0.06 * soil_moisture + 0.02 * fertilizer +
        1.30 * field_size - 0.25 * np.abs(avg_temp - 18) +
        0.70 * cooperative_support + season_effect + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)
    yield_per_ha = yield_tons / field_size
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)
    
    df = pd.DataFrame({
        "rainfall_mm": rainfall, "avg_temp_c": avg_temp, "soil_moisture_pct": soil_moisture,
        "fertilizer_kg": fertilizer, "field_size_ha": field_size, "season": season,
        "cooperative_support": cooperative_support, "yield_tons": yield_tons, "high_risk": high_risk
    })
    
    m = round(0.05 * base_rows)
    df.loc[rng.choice(base_rows, m, replace=False), "soil_moisture_pct"] = np.nan
    df.loc[rng.choice(base_rows, m, replace=False), "fertilizer_kg"] = np.nan
    df = pd.concat([df, df.tail(3).copy()], ignore_index=True)
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, float_format="%.4f")
    
    with open(output_csv, "rb") as f:
        file_hash = hashlib.sha256(f.read()).hexdigest()
        
    os.makedirs(os.path.dirname(fingerprint_txt), exist_ok=True)
    with open(fingerprint_txt, "w") as f:
        f.write(file_hash)
    return df, file_hash
