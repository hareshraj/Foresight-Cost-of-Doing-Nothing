"""
build_features.py
=================
STEP 2. The corrected, hardened replacement for your original data_loader.py.

What changed vs your version (and why it matters):
  * CRS SAFETY: facilities are reprojected to the boundary CRS before the
    spatial join. A silent CRS mismatch was the single biggest latent bug --
    it would misassign facilities and quietly corrupt every downstream number.
  * THRESHOLDS come from config (one source of truth) -- fixes the 5922-vs-5000
    discrepancy between your loader and your dashboard.
  * HONEST CONFIDENCE: instead of a circular ML model that just memorises your
    own threshold rule, "confidence" is a transparent data-robustness score
    (how borderline the LGA is + how complete its inputs are). This is
    defensible to a judge; a fake ML probability is not.
  * DHS OUTCOMES merged on at state level, so the simulator has something real
    to project.

Run (after fetch_dhs.py):
    python src/build_features.py
"""

import sys
import re
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask as rio_mask
from shapely.geometry import mapping

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C


# ──────────────────────────────────────────────────────────────────────────
# State-name normalisation: DHS region labels vs HDX adm1 names rarely match
# character-for-character. Normalise both sides before merging.
# ──────────────────────────────────────────────────────────────────────────
def normalise_state(name):
    if not isinstance(name, str):
        return name
    # DHS prefixes subnational labels with indentation dots, e.g. "..Abia" or
    # "..FCT Abuja". Strip any leading non-letter characters before matching.
    n = re.sub(r"^[^A-Za-z]+", "", name).strip().lower()
    aliases = {
        "fct abuja": "federal capital territory",
        "abuja": "federal capital territory",
        "fct": "federal capital territory",
        "nassarawa": "nasarawa",
        "akwa-ibom": "akwa ibom",
        "cross-river": "cross river",
    }
    return aliases.get(n, n)


# ──────────────────────────────────────────────────────────────────────────
# Population per LGA via WorldPop zonal sum (your logic, with nodata guard)
# ──────────────────────────────────────────────────────────────────────────
def population_per_lga(lgas, raster_path):
    print(f"  Zonal population for {len(lgas)} LGAs (a few minutes)...")
    pops = []
    with rasterio.open(raster_path) as src:
        for i, (_, row) in enumerate(lgas.iterrows()):
            if i % 100 == 0:
                print(f"    {i}/{len(lgas)}")
            try:
                out, _ = rio_mask(src, [mapping(row.geometry)],
                                  crop=True, nodata=-99999.0)
                band = out[0]
                pops.append(round(float(band[band > 0].sum())))
            except Exception:
                pops.append(0)
    return pops


def classify_risk(pop_per_facility, facility_count):
    if facility_count == 0:
        return "critical"
    ppf = pop_per_facility
    if ppf == np.inf or ppf > C.RISK_THRESHOLDS["critical"]:
        return "critical"
    if ppf > C.RISK_THRESHOLDS["high"]:
        return "high"
    if ppf > C.RISK_THRESHOLDS["moderate"]:
        return "moderate"
    return "functional"


def robustness_confidence(row):
    """
    Transparent, non-circular confidence in [0,1]. Lower = needs human review.
    Two honest drivers:
      1. BORDERLINE: an LGA whose people-per-facility sits right on a threshold
         could flip tiers with a tiny data error -> low confidence.
      2. COMPLETENESS: missing population, zero geocoded facilities, or tiny
         denominators make the estimate fragile -> low confidence.
    """
    conf = 1.0

    # (1) distance to nearest threshold boundary, as a fraction of the band width
    ppf = row["pop_per_facility"]
    if ppf == np.inf or pd.isna(ppf):
        conf -= 0.25  # no facilities at all: real signal but inherently uncertain
    else:
        edges = [C.RISK_THRESHOLDS["moderate"],
                 C.RISK_THRESHOLDS["high"],
                 C.RISK_THRESHOLDS["critical"]]
        nearest = min(abs(ppf - e) for e in edges)
        # within 10% of a boundary -> shave up to 0.3
        rel = nearest / max(ppf, 1)
        if rel < 0.10:
            conf -= 0.30 * (1 - rel / 0.10)

    # (2) completeness
    if row.get("population", 0) <= 0:
        conf -= 0.40
    if row.get("facility_count", 0) <= 1:
        conf -= 0.15

    return float(max(0.05, min(1.0, conf)))


def build():
    # ── 1. boundaries ──────────────────────────────────────────────────────
    print("Loading LGA boundaries...")
    lgas = gpd.read_file(C.BOUNDARIES_PATH)
    lgas = lgas[["adm2_name", "adm1_name", "adm2_pcode",
                 "area_sqkm", "center_lat", "center_lon", "geometry"]].copy()
    lgas = lgas.rename(columns={"adm2_name": "lga_name",
                                "adm1_name": "state_name",
                                "adm2_pcode": "lga_pcode"})
    if lgas.crs is None:
        lgas = lgas.set_crs(4326)
    print(f"  {len(lgas)} LGAs (CRS={lgas.crs})")

    # ── 2. population ──────────────────────────────────────────────────────
    print("\nComputing population per LGA...")
    lgas["population"] = population_per_lga(lgas, C.WORLDPOP_PATH)
    print(f"  national estimate: {lgas['population'].sum():,.0f}")

    # ── 3. facilities (CRS-SAFE join) ──────────────────────────────────────
    print("\nLoading & joining facilities...")
    fac = gpd.read_file(C.FACILITIES_PATH)
    if fac.crs is None:
        fac = fac.set_crs(4326)
    if fac.crs != lgas.crs:                       # <-- the critical fix
        print(f"  reprojecting facilities {fac.crs} -> {lgas.crs}")
        fac = fac.to_crs(lgas.crs)

    joined = gpd.sjoin(fac, lgas[["lga_pcode", "geometry"]],
                       how="left", predicate="within")
    counts = (joined.groupby("lga_pcode").size()
              .reset_index(name="facility_count"))
    lgas = lgas.merge(counts, on="lga_pcode", how="left")
    lgas["facility_count"] = lgas["facility_count"].fillna(0).astype(int)

    # ── 4. core access features ────────────────────────────────────────────
    lgas["pop_per_facility"] = np.where(
        lgas["facility_count"] > 0,
        lgas["population"] / lgas["facility_count"].replace(0, np.nan),
        np.inf)
    lgas["facility_density"] = np.where(
        lgas["area_sqkm"] > 0,
        (lgas["facility_count"] / lgas["area_sqkm"]) * 100, 0.0)

    # ── 5. rule-based risk + honest confidence ─────────────────────────────
    lgas["risk_level"] = lgas.apply(
        lambda r: classify_risk(r["pop_per_facility"], r["facility_count"]),
        axis=1)
    lgas["prediction_confidence"] = lgas.apply(robustness_confidence, axis=1)
    lgas["needs_human_review"] = (
        lgas["prediction_confidence"] < C.HUMAN_REVIEW_CONFIDENCE_THRESHOLD)

    # ── 6. merge DHS state outcomes ────────────────────────────────────────
    if C.DHS_CLEAN_CSV.exists():
        print("\nMerging DHS state-level outcomes...")
        dhs = pd.read_csv(C.DHS_CLEAN_CSV)
        dhs["join_key"] = dhs["state_raw"].apply(normalise_state)
        lgas["join_key"] = lgas["state_name"].apply(normalise_state)
        dhs_cols = [c for c in dhs.columns if c not in ("state_raw",)]
        lgas = lgas.merge(dhs[["join_key"] + [c for c in dhs_cols if c != "join_key"]],
                          on="join_key", how="left")
        matched = lgas["facility_delivery"].notna().sum() if "facility_delivery" in lgas else 0
        print(f"  states matched onto LGAs: {matched}/{len(lgas)} LGAs have DHS outcomes")
        lgas = lgas.drop(columns=["join_key"])
    else:
        print("\n[!] DHS CSV not found -- run fetch_dhs.py first. "
              "Continuing without outcomes (simulator will use national baseline).")

    # ── 7. save ────────────────────────────────────────────────────────────
    lgas.to_file(C.FEATURES_GEOJSON, driver="GeoJSON")
    print(f"\nSaved -> {C.FEATURES_GEOJSON}")
    print("\nRisk distribution:")
    print(lgas["risk_level"].value_counts())
    return lgas


if __name__ == "__main__":
    build()