"""
tests/smoke_test_model.py
Proves train_model.py works on data shaped exactly like build_features output,
without needing the real GRID3/WorldPop/DHS files. Run: python tests/smoke_test_model.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C
from src import train_model

rng = np.random.default_rng(0)

# 30 synthetic states, ~10 LGAs each
states = [f"State_{i:02d}" for i in range(30)]
rows = []
for s in states:
    # each state has an underlying access level that drives utilisation
    base_ppf = rng.uniform(3000, 18000)
    fac_delivery = max(5, min(95, 80 - base_ppf / 300 + rng.normal(0, 5)))  # access->utilisation
    for _ in range(10):
        ppf = max(800, base_ppf + rng.normal(0, 2000))
        fc = int(rng.integers(2, 40))
        risk = ("critical" if ppf > 10000 else "high" if ppf > 7500
                else "moderate" if ppf > 5000 else "functional")
        rows.append({
            "state_name": s,
            "pop_per_facility": ppf,
            "facility_density": rng.uniform(0.1, 5),
            "risk_level": risk,
            "facility_delivery": fac_delivery,   # DHS target (same for all LGAs in state)
            "geometry": Point(rng.uniform(3, 14), rng.uniform(4, 13)),
        })

gdf = gpd.GeoDataFrame(rows, crs=4326)
gdf.to_file(C.FEATURES_GEOJSON, driver="GeoJSON")
print(f"wrote {len(gdf)} synthetic LGAs across {len(states)} states\n")

metrics = train_model.train()
print("\nSMOKE TEST PASSED:")
print(f"  target={metrics['target']}  n_states={metrics['n_states']}")
print(f"  LOO MAE={metrics['loo_mae']:.2f}  R2={metrics['loo_r2']:.2f}")
assert C.MODEL_PATH.exists() and C.MODEL_METRICS_PATH.exists()
print("  model + metrics files written OK")