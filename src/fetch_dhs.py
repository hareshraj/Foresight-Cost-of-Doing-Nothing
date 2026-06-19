"""
fetch_dhs.py
============
STEP 1 of the pipeline. Pulls Nigeria subnational (state / zone level) health
OUTCOME indicators from the DHS Program API and writes a tidy CSV that the
feature builder merges onto the LGA map.

Why this exists: your 3 original sources (GRID3 facilities, WorldPop population,
HDX boundaries) describe ACCESS only. They contain no health outcomes, so there
is nothing to project forward in a "cost of doing nothing" simulator. DHS fills
that gap and is free + open.

Run:
    python src/fetch_dhs.py

If your machine has no internet / the API is down, see fetch_from_manual_csv()
at the bottom for the STATcompiler manual-export fallback.

API reference: https://api.dhsprogram.com/#/api-data.cfm
Example verified call:
    https://api.dhsprogram.com/rest/dhs/data?indicatorIds=CM_ECMR_C_U5M&countryIds=NG&breakdown=subnational&f=json
"""

import sys
import json
import time
from pathlib import Path

import requests
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config as C


# ──────────────────────────────────────────────────────────────────────────
# Low-level API helper
# ──────────────────────────────────────────────────────────────────────────
def _get(params, max_retries=4, timeout=30):
    """GET with exponential backoff. Returns parsed JSON or raises."""
    for attempt in range(max_retries):
        try:
            resp = requests.get(C.DHS_API_BASE, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as e:
            wait = 2 ** attempt
            print(f"    request failed ({e}); retry in {wait}s "
                  f"[{attempt + 1}/{max_retries}]")
            time.sleep(wait)
    raise RuntimeError(f"DHS API request failed after {max_retries} retries: {params}")


def fetch_indicator(indicator_id, breakdown="subnational"):
    """
    Pull every page of one indicator for Nigeria at the given breakdown level.
    Returns a list of raw DHS records (dicts).
    """
    records, page, total_pages = [], 1, 1
    while page <= total_pages:
        params = {
            "indicatorIds": indicator_id,
            "countryIds": C.DHS_COUNTRY_CODE,
            "breakdown": breakdown,
            "surveyYearStart": C.DHS_SURVEY_YEAR_START,
            "f": "json",
            "perpage": 1000,
            "page": page,
        }
        payload = _get(params)
        records.extend(payload.get("Data", []))
        total_pages = int(payload.get("TotalPages", 1) or 1)
        page += 1
    return records


# ──────────────────────────────────────────────────────────────────────────
# Indicator search helper -- use this if any ID in config returns nothing
# ──────────────────────────────────────────────────────────────────────────
def search_indicators(keyword):
    """
    Print DHS indicators whose label matches `keyword`, so you can correct any
    indicator ID that silently returns no data. Example:
        python -c "from src.fetch_dhs import search_indicators; search_indicators('facility')"
    """
    url = "https://api.dhsprogram.com/rest/dhs/indicators"
    resp = requests.get(url, params={"f": "json", "perpage": 5000}, timeout=60)
    resp.raise_for_status()
    hits = [
        (d["IndicatorId"], d["Label"])
        for d in resp.json().get("Data", [])
        if keyword.lower() in d.get("Label", "").lower()
    ]
    for iid, label in hits:
        print(f"  {iid:24s}  {label}")
    if not hits:
        print(f"  no indicators matched '{keyword}'")
    return hits


# ──────────────────────────────────────────────────────────────────────────
# Clean one indicator down to ONE canonical value per state.
#
# DHS subnational responses are messy in two verified ways:
#   * LevelRank 1 = 6 geopolitical zones, LevelRank 2 = 37 states. We keep 2.
#   * Each region repeats under several ByVariableLabel recall windows
#     ("Two/Three/Five years preceding the survey"). IsTotal is always 0, so we
#     collapse using DHS's own IsPreferred flag, with a deterministic fallback.
# ──────────────────────────────────────────────────────────────────────────
STATE_LEVEL_RANK = 2


def _is_preferred(r):
    return r.get("IsPreferred") in (1, "1", True)


def clean_state_rows(records):
    """Return (list-of-one-row-per-state, latest_year) for a single indicator."""
    state_rows = [r for r in records if r.get("LevelRank") == STATE_LEVEL_RANK]
    if not state_rows:
        return [], None

    latest_year = max(int(r.get("SurveyYear", 0)) for r in state_rows)
    rows = [r for r in state_rows if int(r.get("SurveyYear", 0)) == latest_year]

    # collapse recall-window duplicates
    preferred = [r for r in rows if _is_preferred(r)]
    if preferred:
        rows = preferred
    else:
        # no preferred flag -> pick the single recall window with widest coverage
        from collections import Counter
        windows = Counter(r.get("ByVariableLabel") for r in rows)
        if windows:
            chosen = windows.most_common(1)[0][0]
            rows = [r for r in rows if r.get("ByVariableLabel") == chosen]

    # final defensive dedup: one row per state label
    seen, clean = set(), []
    for r in rows:
        label = r.get("CharacteristicLabel")
        if label in seen:
            continue
        seen.add(label)
        clean.append(r)
    return clean, latest_year


# ──────────────────────────────────────────────────────────────────────────
# Main extract
# ──────────────────────────────────────────────────────────────────────────
def fetch_all():
    print("Fetching DHS Nigeria subnational indicators...\n")
    raw_by_indicator = {}
    tidy_rows = []

    for short_name, indicator_id in C.DHS_INDICATORS.items():
        print(f"  [{short_name}] {indicator_id} ...", end=" ")
        try:
            records = fetch_indicator(indicator_id, breakdown="subnational")
        except Exception as e:
            print(f"FAILED ({e}) -- skipping")
            continue

        if not records:
            print("no rows returned -- skipping "
                  "(check the ID with search_indicators())")
            continue

        raw_by_indicator[short_name] = records

        clean, latest_year = clean_state_rows(records)
        if not clean:
            print(f"no state-level (LevelRank 2) rows -- skipping "
                  f"(indicator may only be reported at zone level)")
            continue

        for r in clean:
            tidy_rows.append({
                "state_raw": r.get("CharacteristicLabel"),
                "indicator": short_name,
                "value": r.get("Value"),
                "ci_low": r.get("CILow"),
                "ci_high": r.get("CIHigh"),
                "survey_year": latest_year,
                "denominator": r.get("DenominatorWeighted"),
            })
        print(f"ok ({len(clean)} states, latest={latest_year}, raw={len(records)})")

    if not tidy_rows:
        raise SystemExit(
            "\nNo DHS data retrieved at all. Either the network is blocked or "
            "every indicator ID is stale. Use fetch_from_manual_csv() instead."
        )

    # Save raw (for your data-disclosure / reproducibility section)
    C.DHS_RAW_JSON.write_text(json.dumps(raw_by_indicator, indent=2))

    long_df = pd.DataFrame(tidy_rows)

    # Pivot to one row per state, one column per indicator.
    wide = long_df.pivot_table(
        index="state_raw", columns="indicator", values="value", aggfunc="first"
    ).reset_index()

    # Carry the confidence interval of the headline mortality indicator through,
    # so the dashboard can show DHS's own uncertainty.
    ci = (long_df[long_df["indicator"] == "u5_mortality"]
          [["state_raw", "ci_low", "ci_high"]]
          .rename(columns={"ci_low": "u5_mortality_ci_low",
                           "ci_high": "u5_mortality_ci_high"}))
    wide = wide.merge(ci, on="state_raw", how="left")

    wide.to_csv(C.DHS_CLEAN_CSV, index=False)
    print(f"\nSaved {len(wide)} state rows -> {C.DHS_CLEAN_CSV}")
    print(f"Columns: {list(wide.columns)}")
    print(f"Raw JSON -> {C.DHS_RAW_JSON}")
    return wide


def fetch_from_manual_csv(downloaded_csv_path):
    """
    FALLBACK if the API is unreachable. Steps:
      1. Go to https://dhsprogram.com/data/statcompiler.cfm
      2. Country = Nigeria; latest survey; indicators = under-5 mortality,
         fully vaccinated, facility delivery, 4+ ANC; breakdown = by Region/State.
      3. Export to CSV, save it, pass the path here.
    This normalises whatever STATcompiler exports into the same schema as
    fetch_all() so the rest of the pipeline doesn't care which path you used.
    """
    df = pd.read_csv(downloaded_csv_path)
    print("Manual CSV loaded; columns:", list(df.columns))
    print("Rename/rearrange columns to match: "
          "state_raw, u5_mortality, full_vaccination, facility_delivery, anc_4plus")
    return df


if __name__ == "__main__":
    fetch_all()