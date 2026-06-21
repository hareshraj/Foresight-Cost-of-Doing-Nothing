# Foresight
### The Cost of Doing Nothing

**USAII Global AI Hackathon 2026 · Graduate Track · Challenge 6 (Direction A: The Cost of Doing Nothing Simulator)**

An AI decision-support system for Nigerian state primary-healthcare directors. It identifies underserved communities ("healthcare deserts"), then **models the long-term human and economic costs of delaying intervention** so that allocation decisions are proactive and evidence-based instead of reactive.

> It **informs** allocation decisions. It does **not** make them. (See *Human-in-the-loop*.)

---

## What it does

1. **Finds the deserts.** Combines facility locations, population, and boundaries to score all 774 Local Government Areas (LGAs) by people-per-functional-facility.
2. **Simulates the cost of waiting.** For any community, project deaths, DALYs, and economic value over 1, 3, 5 years under different interventions (building clinics, deploying staff, fixing supply chains) vs doing nothing, with every figure presented as an uncertainty range, never a single point. 
3. **Reaches the communities the data misses.** An offline field-report tool lets workers log settlements with no network signal by hand; zero-signal zones are escalated to critical priority. This closes the exclusion-bias gap, where areas with no digital footprint would otherwise be invisible to the model.
4. **Stays accountable.** Surfaces model evaluation, bypass conditions (when *not* to trust it), a two-stage human review path, and drift monitoring.

## Architecture

```
GRID3 facilities ─┐
WorldPop pop ─────┤→ build_features → lgas_with_features.geojson ─┐
HDX boundaries ───┘                                               │
                                                                  ├→ train_model → access_to_care_model + metrics
DHS API outcomes → fetch_dhs → dhs_state_indicators.csv ──────────┘                        │
                                                                                           ▼
                                                       simulator (scenarios + Monte-Carlo + cost of inaction)
                                                                                           ▼
                                                                                    app.py (dashboard)
```

The map is the **entry point**; the simulator is the product.

## Quickstart (run the dashboard)

The processed data and trained model are committed, so the app runs immediately:

```bash
git clone <your-repo-url> && cd <repo>
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Rebuild the pipeline from scratch (optional)

Needed only if you want to re-run the data pipeline. First download the raw sources into `data/raw/`:

- **GRID3 Nigeria Health Facilities v2.0** : https://grid3.org / https://data.grid3.org
- **WorldPop 2020 UN-adjusted constrained (Nigeria)** : https://hub.worldpop.org
- **HDX / OCHA admin boundaries (admin2)** : https://data.humdata.org

Then:

```bash
python src/fetch_dhs.py        # pulls DHS state outcomes (needs internet)
python src/build_features.py   # joins everything, ~3 min (WorldPop zonal sum)
python src/train_model.py      # fits + cross-validates the access→care model
streamlit run app.py
```

## The model (in one paragraph)

Risk tiers are **rule-based** and transparent (people-per-facility vs WHO-aligned thresholds). A separate **learned model** (Ridge regression, regularisation chosen by nested leave-one-state-out CV) predicts care utilisation (facility delivery) from access structure **controlling for women's secondary education**, the dominant driver. Adding that confounder lifted leave-one-state-out R² from −0.62 to **+0.68** (MAE 12pp vs 23pp naive). The key insight: building clinics is a **real but bounded** lever, demand-side factors carry most of the weight, so the simulator uses only the *partial* access effect and never over-credits construction.

## Data sources

| Source | Use | Level |
|---|---|---|
| GRID3 Health Facilities v2.0 | facility counts | point → LGA |
| WorldPop 2020 UN-adj. constrained | population | raster → LGA |
| HDX/OCHA admin2 boundaries | geometry | LGA (774) |
| DHS Program API (NDHS 2023–24) | health outcomes + education | state (37) |

All data is aggregated and open. No individual-level or personal data is used.

## Responsible AI

Uncertainty ranges on every estimate; explicit **bypass conditions** (when the system should *not* be trusted); **two-cycle** human review before any resource moves; drift monitoring on each DHS release; and an offline field-report path so communities with no network signal aren't excluded by the data itself.

## Repo structure

```
config.py                  single source of truth (paths, thresholds, assumptions)
app.py                     Streamlit dashboard
.streamlit/config.toml     theme
src/
  fetch_dhs.py             DHS API extraction
  build_features.py        feature pipeline (CRS-safe join, risk, confidence, DHS merge)
  train_model.py           access→care model + nested LOO CV
  simulator.py             cost-of-inaction Monte-Carlo engine
```

## Team

**The Exceptions**

Haresh Raj,
Muhammad Saad Umar,
Jesutomiwo Sapphire
