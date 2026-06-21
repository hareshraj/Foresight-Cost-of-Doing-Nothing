from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

for _d in (RAW, PROCESSED, MODELS):
    _d.mkdir(parents=True, exist_ok=True)

# Existing source files
BOUNDARIES_PATH = RAW / "nga_admin_boundaries.geojson" / "nga_admin2.geojson"
FACILITIES_PATH = RAW / "GRID3_NGA_health_facilities_v2_0_-5661871903075391498.geojson"
WORLDPOP_PATH = RAW / "nga_ppp_2020_UNadj_constrained.tif"

# Outputs produced by the pipeline
DHS_RAW_JSON = RAW / "dhs_nigeria_subnational_raw.json"
DHS_CLEAN_CSV = RAW / "dhs_nigeria_state_indicators.csv"
FEATURES_GEOJSON = PROCESSED / "lgas_with_features.geojson"
PREDICTIONS_GEOJSON = PROCESSED / "lgas_with_predictions.geojson"
MODEL_PATH = MODELS / "access_to_care_model.joblib"
MODEL_METRICS_PATH = MODELS / "model_metrics.json"

# ──────────────────────────────────────────────────────────────────────────
# DHS API
# ──────────────────────────────────────────────────────────────────────────
DHS_API_BASE = "https://api.dhsprogram.com/rest/dhs/data"
DHS_COUNTRY_CODE = "NG"
DHS_SURVEY_YEAR_START = 2018

# Indicators to pull. Keys are our internal short names; values are DHS IDs.
# Mortality IDs are verified. The state-level utilisation indicators are the
# ones we actually FIT the model on (larger samples -> available per state).
# fetch_dhs.py will report which IDs returned data; any that fail are skipped,
# not fatal.
DHS_INDICATORS = {
    "u5_mortality":        "CM_ECMR_C_U5M",
    "infant_mortality":    "CM_ECMR_C_IMR",
    "full_vaccination":    "CH_VACC_C_BAS",
    "facility_delivery":   "RH_DELP_C_DHF",
    "anc_4plus":           "RH_ANCN_W_N4P",
    "women_secondary_edu": "ED_EDUC_W_SEH", 
}

# Which indicator the model TRIES to predict, in order of preference.
# The first one that has enough state-level coverage wins.
MODEL_TARGET_PREFERENCE = ["facility_delivery", "full_vaccination", "anc_4plus"]

# Minimum number of states with a usable target value to fit a model at all.
MIN_STATES_FOR_MODEL = 15

# ──────────────────────────────────────────────────────────────────────────
# RISK CLASSIFICATION THRESHOLDS  (people per functional facility)
# Basis: WHO benchmark of ~1 primary-care facility per 10,000 people is the
# "critical" red line. The intermediate bands are the 80th/90th percentiles of
# population-per-facility across all 774 LGAs.
# ──────────────────────────────────────────────────────────────────────────
RISK_THRESHOLDS = {
    "critical": 10000,   # > 10,000 people/facility
    "high": 7500,        # 7,500 - 10,000
    "moderate": 5000,    # 5,000 - 7,500
    # functional: < 5,000
}
RISK_ORDER = ["critical", "high", "moderate", "functional"]
RISK_COLORS = {
    "critical": "#d62728",
    "high": "#ff7f0e",
    "moderate": "#e6b800",
    "functional": "#2ca02c",
}

# Confidence below this routes an LGA to mandatory human review.
HUMAN_REVIEW_CONFIDENCE_THRESHOLD = 0.70

# ──────────────────────────────────────────────────────────────────────────
# EPIDEMIOLOGICAL & ECONOMIC ASSUMPTIONS  (the "cost of doing nothing" engine)
# Every value here is an ASSUMPTION with a source. The simulator samples around
# these (Monte Carlo) so outputs are ranges, never single points.
# ──────────────────────────────────────────────────────────────────────────

# Baseline under-5 mortality, NDHS 2023-24: 110 per 1,000 live births.
BASELINE_U5MR_PER_1000 = 110.0
U5MR_CI = (103.0, 117.0)

# Crude birth rate, Nigeria ~ 37 per 1,000 population per year (World Bank).
# Used to convert population -> annual live births.
CRUDE_BIRTH_RATE_PER_1000 = 37.0

# Share of under-5 deaths that are vaccine-preventable: ~41% (IHME GBD 2019).
# This is the lever through which improving facility access / vaccination
# coverage translates into deaths averted.
VACCINE_PREVENTABLE_FRACTION = 0.41
VACCINE_PREVENTABLE_FRACTION_RANGE = (0.35, 0.47)

# How strongly a 1-percentage-point rise in care utilisation (facility delivery
# / full vaccination) reduces the *preventable* share of U5 mortality.
UTILISATION_TO_MORTALITY_ELASTICITY = 0.006
UTILISATION_TO_MORTALITY_ELASTICITY_RANGE = (0.003, 0.009)

# Economic valuation. We report BOTH a health metric (deaths/DALYs averted) and
# a monetary one (cost of inaction). DALYs per under-5 death and value per DALY
# are explicit, editable assumptions.
DALYS_PER_U5_DEATH = 30.0
VALUE_PER_DALY_USD = 1600.0
VALUE_PER_DALY_USD_RANGE = (800.0, 3200.0)

# Rough unit costs for intervention scenarios (USD).
COST_PER_NEW_PHC_FACILITY_USD = 120000.0
COST_PER_HEALTH_WORKER_YEAR_USD = 6000.0

# ──────────────────────────────────────────────────────────────────────────
# SIMULATION
# ──────────────────────────────────────────────────────────────────────────
HORIZONS_YEARS = [1, 3, 5]
ANNUAL_POP_GROWTH_RATE = 0.025            # Nigeria ~2.5%/yr (World Bank)
MONTE_CARLO_RUNS = 2000
DISCOUNT_RATE = 0.03                      # standard 3% social discount rate

# ──────────────────────────────────────────────────────────────────────────
# GOVERNANCE / BYPASS CONDITIONS  (surfaced in the dashboard)
# The system must refuse to advise when its assumptions break down.
# ──────────────────────────────────────────────────────────────────────────
BYPASS_CONDITIONS = {
    "min_data_completeness": 0.60,     # < 60% of inputs present -> do not advise
    "min_prediction_confidence": 0.60, # model confidence below this -> human-led only
    "max_population_data_age_years": 6,# WorldPop 2020 is fine in 2026; flag if older basis used
}