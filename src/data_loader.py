import geopandas as gpd
import pandas as pd
import numpy as np
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

BOUNDARIES_PATH = "data/raw/nga_admin_boundaries.geojson/nga_admin2.geojson"
FACILITIES_PATH = "data/raw/GRID3_NGA_health_facilities_v2_0_-5661871903075391498.geojson"
WORLDPOP_PATH   = "data/raw/nga_ppp_2020_UNadj_constrained.tif"

def get_population_per_lga(lgas, raster_path):
    print("  Processing LGAs (774 total, ~3-5 mins)...")
    populations = []
    with rasterio.open(raster_path) as src:
        for i, (_, row) in enumerate(lgas.iterrows()):
            if i % 100 == 0:
                print(f"  ...{i}/774 LGAs processed")
            try:
                geom = [mapping(row.geometry)]
                out_image, _ = rio_mask(src, geom, crop=True, nodata=-99999.0)
                data = out_image[0]
                total = float(data[data > 0].sum())
                populations.append(round(total))
            except Exception:
                populations.append(0)
    return populations

def classify_risk(row):
    if row["facility_count"] == 0:
        return "critical"
    ppf = row["pop_per_facility"]
    if ppf == np.inf or ppf > 10000:
        return "critical"
    elif ppf > 7500:
        return "high"
    elif ppf > 5922:
        return "moderate"
    else:
        return "functional"

def load_all_data():
    # ── 1. LGA BOUNDARIES ──────────────────────────────────────────────────
    print("Loading LGA boundaries...")
    lgas = gpd.read_file(BOUNDARIES_PATH)
    lgas = lgas[[
        "adm2_name", "adm1_name", "adm2_pcode",
        "area_sqkm", "center_lat", "center_lon", "geometry"
    ]].copy()
    lgas = lgas.rename(columns={
        "adm2_name": "lga_name",
        "adm1_name": "state_name",
        "adm2_pcode": "lga_pcode"
    })
    print(f"  {len(lgas)} LGAs loaded")

    # ── 2. WORLDPOP POPULATION PER LGA ─────────────────────────────────────
    print("\nComputing population per LGA...")
    lgas["population"] = get_population_per_lga(lgas, WORLDPOP_PATH)
    total_pop = lgas["population"].sum()
    print(f"  Nigeria population estimate: {total_pop:,.0f}")
    print(f"  Highest population LGA: {lgas.loc[lgas['population'].idxmax(), 'lga_name']} "
          f"({lgas['population'].max():,.0f})")

    # ── 3. HEALTH FACILITIES ────────────────────────────────────────────────
    print("\nLoading health facilities...")
    facilities = gpd.read_file(FACILITIES_PATH)
    facilities = facilities[[
        "facility_name", "facility_level", "ownership",
        "ownership_type", "state", "lga", "geometry"
    ]].copy()
    print(f"  {len(facilities)} facilities loaded")

    # ── 4. SPATIAL JOIN: COUNT FACILITIES PER LGA ──────────────────────────
    print("\nSpatial join: assigning facilities to LGAs...")
    joined = gpd.sjoin(
        facilities,
        lgas[["lga_name", "lga_pcode", "geometry"]],
        how="left",
        predicate="within"
    )

    facility_counts = (
        joined.groupby("lga_pcode")
        .size()
        .reset_index(name="facility_count")
    )

    level_counts = (
        joined.groupby(["lga_pcode", "facility_level"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    # ── 5. MERGE ────────────────────────────────────────────────────────────
    print("\nMerging datasets...")
    lgas = lgas.merge(facility_counts, on="lga_pcode", how="left")
    lgas = lgas.merge(level_counts,    on="lga_pcode", how="left")
    lgas["facility_count"] = lgas["facility_count"].fillna(0).astype(int)

    # ── 6. CORE FEATURES ────────────────────────────────────────────────────
    lgas["pop_per_facility"] = lgas.apply(
        lambda r: r["population"] / r["facility_count"]
        if r["facility_count"] > 0 else np.inf,
        axis=1
    )

    lgas["facility_density"] = lgas.apply(
        lambda r: (r["facility_count"] / r["area_sqkm"]) * 100
        if r["area_sqkm"] and r["area_sqkm"] > 0 else 0,
        axis=1
    )

    # ── 7. RISK CLASSIFICATION ──────────────────────────────────────────────
    lgas["risk_level"] = lgas.apply(classify_risk, axis=1)

    # ── 8. SUMMARY ──────────────────────────────────────────────────────────
    print("\n── Risk Distribution ──────────────────────────────")
    print(lgas["risk_level"].value_counts())
    print("\n── Top 10 Highest Risk LGAs ───────────────────────")
    print(lgas[[
        "lga_name", "state_name", "population",
        "facility_count", "pop_per_facility", "risk_level"
    ]].sort_values("pop_per_facility", ascending=False).head(10))

    return lgas, facilities

if __name__ == "__main__":
    lgas, facilities = load_all_data()
    lgas.to_file("data/processed/lgas_with_risk.geojson", driver="GeoJSON")
    print("\nSaved to data/processed/lgas_with_risk.geojson")