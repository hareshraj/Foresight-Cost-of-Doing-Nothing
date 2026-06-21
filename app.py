import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import altair as alt
import folium
from streamlit_folium import st_folium
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent))
import config as C
from src.simulator import compare_scenarios, explain_chain

# ──────────────────────────────────────────────────────────────────────────
# PAGE + DESIGN SYSTEM
# Palette (named):  ink #0F1B2D · slate #3A4A5E · teal #0E7C7B · mist #F4F6F9
# Risk ramp kept from config for continuity with the map.
# ──────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Healthcare Cost-of-Inaction System",
                   page_icon="🩺", layout="wide")

# "Nigerian atlas" palette — luminous green + Benin-bronze gold on forest ink.
INK      = "#08130E"   # deep forest-black: app base + signature panel
PANEL    = "#0E1C15"   # dark forest panel / card surface
LINE     = "#203A2C"   # green-tinted borders
TEXT     = "#DBE7DE"   # soft warm-white text
GRAPHITE = "#85A091"   # dim sage (secondary text)
TEAL     = "#2FC279"   # luminous Nigerian green (brand / signal)
AMBER    = "#E6B547"   # Benin-bronze gold (warm secondary accent)
ALARM    = "#FF6B5A"   # critical / cost emphasis (luminous on dark)
MIST     = INK         # legacy alias → base
PAPER    = PANEL       # legacy alias → panel surface
SLATE    = GRAPHITE    # legacy alias

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Public+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@500;600&display=swap');

html, body, [class*="css"], .stMarkdown, p, span, label {{ font-family:'Public Sans', system-ui, sans-serif; }}
h1,h2,h3,h4 {{ font-family:'Space Grotesk', system-ui, sans-serif; color:{INK}; letter-spacing:-0.015em; }}
.stApp {{
  background-color:#E7ECEE;
  background-image:
    radial-gradient(1100px 480px at 84% -8%, rgba(14,124,107,.13), transparent 60%),
    radial-gradient(820px 440px at -6% -4%, rgba(16,36,43,.06), transparent 55%),
    radial-gradient(rgba(16,36,43,.05) 1px, transparent 1px);
  background-size:auto, auto, 26px 26px;
  background-attachment:fixed;
}}
header[data-testid="stHeader"] {{ display:none; }}
.block-container {{ padding-top:1.5rem; padding-bottom:3rem; max-width:1480px; }}

/* ── masthead ───────────────────────────────────────────── */
.coord {{ font-family:'IBM Plex Mono',monospace; font-size:0.7rem; letter-spacing:0.22em;
          text-transform:uppercase; color:{TEAL}; }}
.mast-title {{ font-family:'Space Grotesk'; font-weight:700; font-size:2.6rem; line-height:1.0;
              color:{INK}; margin:5px 0 7px; letter-spacing:-0.025em; }}
.mast-thesis {{ font-family:'Space Grotesk'; font-weight:700; font-size:2rem;
   line-height:1.05; color:{TEAL}; letter-spacing:-0.02em; margin:0 0 15px;
   text-shadow:0 0 26px rgba(47,194,121,.18); }}
.mast-sub {{ color:{GRAPHITE}; font-size:1.12rem; line-height:1.5; max-width:860px; }}
.tickrule {{ height:10px; margin-top:14px; border-top:1.5px solid {INK}; opacity:0.85;
   background:repeating-linear-gradient(90deg, {INK} 0 1px, transparent 1px 38px) top/100% 6px no-repeat; }}

/* ── instrument-readout KPI cards ───────────────────────── */
.kpi-row {{ display:flex; gap:12px; flex-wrap:wrap; margin:20px 0 4px; }}
.kpi {{ flex:1; min-width:152px; background:{PAPER}; border:1px solid {LINE}; border-radius:12px;
        padding:14px 16px 15px; position:relative; overflow:hidden;
        transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }}
.kpi:hover {{ transform:translateY(-3px); box-shadow:0 10px 24px rgba(16,36,43,.10); border-color:#C4D0D2; }}
.kpi::before {{ content:''; position:absolute; top:0; left:18px; width:30px; height:3px;
                background:linear-gradient(90deg, {TEAL}, #1E9E5C); transition:width .22s ease; }}
.kpi:hover::before {{ width:54px; }}
.kpi.alarm::before {{ background:linear-gradient(90deg, {ALARM}, #D9613F); }}
.kpi .lab {{ font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.13em;
             text-transform:uppercase; color:{GRAPHITE}; }}
.kpi .val {{ font-family:'Space Grotesk'; font-weight:700; font-size:1.85rem; color:{INK};
             line-height:1.1; margin-top:7px; }}
.kpi.alarm .val {{ color:{ALARM}; }}
.kpi .sub {{ font-size:0.7rem; color:{GRAPHITE}; margin-top:2px; }}

/* ── shared ─────────────────────────────────────────────── */
.eyebrow {{ font-family:'IBM Plex Mono',monospace; font-size:0.66rem; letter-spacing:0.18em;
            text-transform:uppercase; color:{TEAL}; margin-bottom:3px; }}
.card {{ background:{PAPER}; border:1px solid {LINE}; border-radius:14px; padding:18px 20px;
         color:{INK}; box-shadow:0 1px 2px rgba(16,36,43,.04);
         transition:transform .18s ease, box-shadow .18s ease, border-color .18s ease; }}
.card:hover {{ transform:translateY(-2px); box-shadow:0 10px 26px rgba(16,36,43,.09); border-color:#C4D0D2; }}
.card small {{ color:{GRAPHITE}; }}
.num {{ font-family:'IBM Plex Mono',monospace; }}
.pill {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:0.7rem;
         font-weight:600; font-family:'IBM Plex Mono'; }}

/* ── signature panel: cost of doing nothing ─────────────── */
.hero {{ position:relative; color:#fff; border-radius:16px; padding:28px 30px; overflow:hidden;
   background:
     repeating-linear-gradient(0deg, rgba(255,255,255,.05) 0 1px, transparent 1px 30px),
     repeating-linear-gradient(90deg, rgba(255,255,255,.05) 0 1px, transparent 1px 30px),
     linear-gradient(135deg, {INK} 0%, #0b3236 100%);
   background-size:30px 30px, 30px 30px, 100% 100%;
   animation:heroDrift 34s linear infinite;
   transition:box-shadow .25s ease; }}
@keyframes heroDrift {{ to {{ background-position:30px 30px, -30px 30px, 0 0; }} }}
.hero:hover {{ box-shadow:0 16px 44px rgba(11,50,54,.32); }}
.hero::after {{ content:''; position:absolute; top:0; left:0; width:88px; height:4px; background:{ALARM}; }}
.hero .eyebrow {{ color:#79D0C4; }}
.hero .big {{ font-family:'Space Grotesk'; font-weight:700; font-size:2.6rem; line-height:1.04; }}
.hero .sub {{ color:#B4C7D2; font-size:0.92rem; }}

/* ── Streamlit chrome ───────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{ gap:2px; border-bottom:1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-family:'Space Grotesk'; font-weight:600; font-size:0.9rem;
                                color:{GRAPHITE}; padding:10px 14px; }}
.stTabs [aria-selected="true"] {{ color:{INK}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background:{TEAL}; height:2px; }}
[data-baseweb="select"] > div {{ border-radius:9px; border-color:{LINE}; }}
[data-testid="stExpander"] {{ border:1px solid {LINE}; border-radius:12px; background:{PAPER}; }}
[data-testid="stExpander"] summary {{ font-family:'Space Grotesk'; font-weight:600; color:{INK}; }}
.stButton>button, [data-testid="stFormSubmitButton"]>button {{
    font-family:'Space Grotesk'; font-weight:600; border-radius:9px; }}
[data-testid="stMetricValue"] {{ font-family:'Space Grotesk'; color:{INK}; }}
[data-testid="stMetricLabel"] p {{ font-family:'IBM Plex Mono'; font-size:0.72rem;
    letter-spacing:0.06em; color:{GRAPHITE}; }}
[data-testid="stDataFrame"] {{ font-family:'Public Sans'; }}
.stTabs [data-baseweb="tab"]:hover {{ color:{INK}; }}
.stButton>button, [data-testid="stFormSubmitButton"]>button {{ transition:transform .15s ease, box-shadow .15s ease; }}
.stButton>button:hover, [data-testid="stFormSubmitButton"]>button:hover {{
    transform:translateY(-1px); box-shadow:0 6px 16px rgba(14,124,107,.28); }}
[data-testid="stExpander"] {{ transition:border-color .18s ease, box-shadow .18s ease; }}
[data-testid="stExpander"]:hover {{ border-color:#C4D0D2; box-shadow:0 6px 18px rgba(16,36,43,.06); }}
@media (prefers-reduced-motion: reduce) {{
  *, .hero {{ animation:none !important; transition:none !important; }}
}}
</style>""", unsafe_allow_html=True)

st.markdown(f"""
<style>
.stApp {{
  background-color:{INK};
  background-image:
    radial-gradient(1200px 580px at 82% -12%, rgba(47,194,121,.13), transparent 60%),
    radial-gradient(960px 540px at -8% 2%, rgba(230,181,71,.08), transparent 58%),
    repeating-radial-gradient(circle at 50% -25%, transparent 0 27px, rgba(47,194,121,.024) 27px 28px);
  background-attachment:fixed;
}}
h1,h2,h3,h4 {{ color:{TEXT}; }}

/* masthead with glowing contour rings */
.masthead {{ position:relative; padding:10px 0 2px; }}
.masthead::before {{ content:''; position:absolute; right:-30px; top:-86px; width:480px; height:360px;
  background:repeating-radial-gradient(circle at center, transparent 0 21px, rgba(47,194,121,.17) 21px 22px);
  -webkit-mask:radial-gradient(circle at center,#000 0%,transparent 68%);
  mask:radial-gradient(circle at center,#000 0%,transparent 68%);
  pointer-events:none; animation:ringPulse 7s ease-in-out infinite alternate; }}
@keyframes ringPulse {{ from {{opacity:.45}} to {{opacity:.95}} }}
.mast-title {{ color:{TEXT}; font-size:3.05rem; text-shadow:0 0 34px rgba(47,194,121,.20); }}
.mast-sub {{ color:{GRAPHITE}; }}
.tickrule {{ border-top-color:{TEAL}; opacity:.6;
  background:repeating-linear-gradient(90deg,{TEAL} 0 1px, transparent 1px 38px) top/100% 6px no-repeat; }}

/* dark panels + glow hovers */
.card {{ color:{TEXT}; box-shadow:0 1px 2px rgba(0,0,0,.3); }}
.card:hover {{ box-shadow:0 14px 36px rgba(0,0,0,.5); border-color:{TEAL}66; }}
.kpi .val {{ color:{TEXT}; }}
.kpi:hover {{ box-shadow:0 12px 30px rgba(0,0,0,.5); border-color:{TEAL}66; }}
.kpi:hover::before {{ box-shadow:0 0 14px {TEAL}; }}

/* native chrome on dark */
.stTabs [aria-selected="true"] {{ color:{TEXT}; }}
.stTabs [data-baseweb="tab"]:hover {{ color:{TEXT}; }}
[data-testid="stExpander"] summary {{ color:{TEXT}; }}
[data-testid="stMetricValue"] {{ color:{TEXT}; }}
[data-baseweb="select"] > div {{ background:{PANEL}; }}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────
# DATA
# ──────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_lgas():
    g = gpd.read_file(C.FEATURES_GEOJSON)
    for col in ["population", "facility_count", "pop_per_facility",
                "facility_density", "prediction_confidence", "area_sqkm",
                "facility_delivery", "women_secondary_edu", "u5_mortality",
                "full_vaccination", "anc_4plus"]:
        if col in g:
            g[col] = pd.to_numeric(g[col], errors="coerce")
    g["population"] = g["population"].fillna(0).astype(int)
    g["facility_count"] = g["facility_count"].fillna(0).astype(int)
    g["needs_human_review"] = (
        g["needs_human_review"].astype(str).str.lower().isin(["true", "1"]))
    return g


@st.cache_data
def load_metrics():
    if C.MODEL_METRICS_PATH.exists():
        return json.loads(C.MODEL_METRICS_PATH.read_text())
    return {}


lgas = load_lgas()
M = load_metrics()
SLOPE = M.get("slope_util_per_ppf", 0.0028)
REL_UNC = M.get("model_rel_uncertainty", 0.27)


def money(x):
    a = abs(x)
    if a >= 1e9:  return f"${x/1e9:.1f}B"
    if a >= 1e6:  return f"${x/1e6:.1f}M"
    if a >= 1e3:  return f"${x/1e3:.0f}K"
    return f"${x:,.0f}"


def ppf_str(v):
    return "∞ (no facilities)" if pd.isna(v) or v == np.inf else f"{v:,.0f}"


def dark_chart(ch, height=300):
    """Apply the dark atlas theme to an Altair chart."""
    return (ch.properties(height=height)
            .configure(background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelColor=TEXT, titleColor=GRAPHITE, gridColor="#16281D",
                            domainColor=LINE, tickColor=LINE,
                            labelFont="Public Sans", titleFont="Space Grotesk")
            .configure_legend(labelColor=TEXT, titleColor=GRAPHITE, labelFont="Public Sans"))


# ──────────────────────────────────────────────────────────────────────────
# HEADER + KPI STRIP
# ──────────────────────────────────────────────────────────────────────────
crit = lgas[lgas.risk_level == "critical"]

st.markdown(f"""
<div class="masthead">
<div class="coord">Primary Healthcare Intelligence · Federal Republic of Nigeria · 9°08′N 8°40′E</div>
<div class="mast-title">Foresight</div>
<div class="mast-thesis">The Cost of Doing Nothing</div>
<div class="mast-sub">Across Nigeria's {len(lgas):,} local government areas, this system finds the
<b style="color:{TEXT}">healthcare deserts</b> and models what waiting costs —
<b style="color:{TEXT}">before the bill comes due in lives</b>.</div>
<div class="mast-sub" style="margin-top:8px;font-size:0.82rem;opacity:.75;max-width:none">This
system informs allocation decisions; it does not make them.</div>
<div class="tickrule"></div>
</div>
""", unsafe_allow_html=True)


def kpi(lab, val, sub, alarm=False):
    cls = "kpi alarm" if alarm else "kpi"
    return (f"<div class='{cls}'><div class='lab'>{lab}</div>"
            f"<div class='val'>{val}</div><div class='sub'>{sub}</div></div>")


st.markdown(
    "<div class='kpi-row'>" + "".join([
        kpi("LGAs analysed", f"{len(lgas):,}", "local government areas"),
        kpi("Critical deserts", f"{len(crit):,}", "&gt;10,000 people per clinic", alarm=True),
        kpi("Exposed population", f"{crit.population.sum()/1e6:.1f}M", "in critical zones", alarm=True),
        kpi("High risk", f"{(lgas.risk_level == 'high').sum():,}", "7,500–10,000 per clinic"),
        kpi("Model R²", f"{M.get('loo_r2', float('nan')):.2f}", "leave-one-state-out CV"),
        kpi("Flagged for review", f"{int(lgas.needs_human_review.sum()):,}", "low model confidence"),
    ]) + "</div>", unsafe_allow_html=True)

st.write("")
tab_sim, tab_map, tab_model, tab_field = st.tabs(
    ["  Cost-of-Inaction Simulator  ", "  Risk Map  ",
     "  Model & Governance  ", "  Field Reports  "])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — RISK MAP
# ══════════════════════════════════════════════════════════════════════════
with tab_map:
    left, right = st.columns([3, 1])

    with right:
        st.markdown('<div class="eyebrow">Filters</div>', unsafe_allow_html=True)
        sel_risk = st.multiselect(
            "Risk levels", C.RISK_ORDER, default=C.RISK_ORDER,
            format_func=lambda x: x.title())
        states = sorted(lgas.state_name.dropna().unique())
        sel_state = st.multiselect("States", states, default=[])

    flt = lgas[lgas.risk_level.isin(sel_risk)].copy()
    if sel_state:
        flt = flt[flt.state_name.isin(sel_state)]

    with left:
        st.markdown(f'<div class="eyebrow">Clinical access risk · {len(flt)} LGAs shown</div>',
                    unsafe_allow_html=True)
        m = folium.Map(location=[9.1, 8.1], zoom_start=6, tiles="CartoDB dark_matter",
                       min_zoom=5, max_zoom=12, max_bounds=True)
        if sel_state:                                # zoom to the selected state(s)
            _b = flt.total_bounds  # minx, miny, maxx, maxy
            if np.all(np.isfinite(_b)):
                m.fit_bounds([[_b[1], _b[0]], [_b[3], _b[2]]], padding=(8, 8))
        else:                                        # frame the whole country on first load
            m.fit_bounds([[4.0, 2.7], [13.9, 14.7]])
        m.options["maxBounds"] = [[2.5, 1.0], [20.0, 16.0]]   # big northern headroom for popups
        m.options["maxBoundsViscosity"] = 0.25                # loose, so popups auto-pan into view
        for _, r in flt.iterrows():
            color = C.RISK_COLORS.get(r.risk_level, "#888")
            review = ("<div style='background:#FFF4E5;border:1px solid #FFB74D;"
                      "border-radius:4px;padding:5px;margin-top:6px;font-size:11px'>"
                      "⚠ Low confidence — requires board review</div>"
                      if r.needs_human_review else "")
            popup = f"""
            <div style='font-family:Inter,sans-serif;min-width:230px'>
              <div style='font-weight:700;font-size:14px;color:{INK}'>{r.lga_name}</div>
              <small>{r.state_name}</small>
              <div style='margin:6px 0'><span class='pill' style='background:{color};color:#fff'>
                {r.risk_level.upper()}</span>
                <span style='font-size:11px;color:#555'> · confidence {r.prediction_confidence:.0%}</span></div>
              <hr style='margin:6px 0;border:none;border-top:1px solid #eee'>
              <table style='font-size:12px;width:100%'>
                <tr><td>Population</td><td style='text-align:right'><b>{int(r.population):,}</b></td></tr>
                <tr><td>Facilities</td><td style='text-align:right'><b>{int(r.facility_count)}</b></td></tr>
                <tr><td>People / facility</td><td style='text-align:right'><b>{ppf_str(r.pop_per_facility)}</b></td></tr>
                <tr><td>Facility delivery (state)</td><td style='text-align:right'>{r.get('facility_delivery', float('nan')):.0f}%</td></tr>
                <tr><td>Women w/ 2° education (state)</td><td style='text-align:right'>{r.get('women_secondary_edu', float('nan')):.0f}%</td></tr>
              </table>{review}
            </div>"""
            try:
                folium.GeoJson(
                    r.geometry.__geo_interface__,
                    style_function=lambda _f, c=color: {
                        "fillColor": c, "color": "#EAF2EE",
                        "weight": 0.5, "opacity": 0.7, "fillOpacity": 0.82},
                    tooltip=f"{r.lga_name} · {r.risk_level.upper()}",
                    popup=folium.Popup(popup, max_width=290)).add_to(m)
            except Exception:
                pass

        legend = ("<div style='position:fixed;bottom:28px;left:28px;z-index:1000;"
                  "background:rgba(11,22,16,.93);padding:11px 15px;border-radius:10px;"
                  "border:1px solid #203A2C;backdrop-filter:blur(4px);"
                  "box-shadow:0 6px 20px rgba(0,0,0,.55);font-family:Inter;"
                  "font-size:12px;color:#DBE7DE'>"
                  "<div style='font-weight:700;margin-bottom:5px;font-family:Space Grotesk'>People per facility</div>")
        for lvl, lbl in [("critical", "&gt;10,000"), ("high", "7,500–10,000"),
                         ("moderate", "5,000–7,500"), ("functional", "&lt;5,000")]:
            legend += (f"<div><span style='color:{C.RISK_COLORS[lvl]};font-size:15px'>■</span>"
                       f"&nbsp;{lvl.title()} ({lbl})</div>")
        legend += "</div>"
        m.get_root().html.add_child(folium.Element(legend))
        st_folium(m, width=None, height=560, returned_objects=[])

    st.markdown('<div class="eyebrow">Priority LGAs — worst access first</div>',
                unsafe_allow_html=True)
    pr = (flt[flt.risk_level.isin(["critical", "high"])]
          .sort_values("pop_per_facility", ascending=False)
          [["lga_name", "state_name", "risk_level", "population",
            "facility_count", "pop_per_facility", "prediction_confidence"]]
          .head(15).reset_index(drop=True))
    pr.index += 1
    pr.columns = ["LGA", "State", "Risk", "Population", "Facilities",
                  "People/Facility", "Confidence"]
    pr["Confidence"] = pr["Confidence"].map(lambda x: f"{x:.0%}")
    st.dataframe(pr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — COST-OF-INACTION SIMULATOR  (the centrepiece)
# ══════════════════════════════════════════════════════════════════════════
with tab_sim:
    c1, c2 = st.columns([1, 2])

    with c1:
        st.markdown('<div class="eyebrow">Choose a community</div>', unsafe_allow_html=True)
        s_states = sorted(lgas.state_name.dropna().unique())
        default_state = (crit.state_name.mode().iloc[0]
                         if len(crit) else s_states[0])
        sstate = st.selectbox("State", s_states,
                              index=s_states.index(default_state))
        sub = lgas[lgas.state_name == sstate].sort_values(
            "pop_per_facility", ascending=False)
        slga_name = st.selectbox("LGA", sub.lga_name.tolist())
        lga = sub[sub.lga_name == slga_name].iloc[0]

        st.markdown('<div class="eyebrow">Proposed intervention</div>',
                    unsafe_allow_html=True)
        n_clinics = st.slider("New primary-health centres", 0, 10, 2)
        n_staff = st.slider("Health workers deployed", 0, 40, 15)
        supply = st.checkbox("Supply-chain capacity fix (+15%)", value=True)

    # bypass-condition gate
    conf = float(lga.prediction_confidence)
    bypass = (conf < C.BYPASS_CONDITIONS["min_prediction_confidence"]
              or bool(lga.needs_human_review))

    # build scenarios
    scen = {"Do nothing": {"kind": "do_nothing"}}
    if n_clinics > 0:
        scen[f"Build {n_clinics} clinics"] = {"kind": "build_facilities",
                                              "n_facilities": n_clinics}
    if n_staff > 0:
        scen[f"Deploy {n_staff} staff"] = {"kind": "deploy_staff", "n_staff": n_staff}
    if supply:
        scen["Supply-chain fix"] = {"kind": "supply_chain", "capacity_boost": 0.15}

    lga_d = {"population": int(lga.population),
             "facility_count": int(lga.facility_count),
             "pop_per_facility": float(lga.pop_per_facility)
                if not pd.isna(lga.pop_per_facility) else float(lga.population)}
    res = compare_scenarios(lga_d, scen, SLOPE, REL_UNC)

    actions = [n for n in scen if n != "Do nothing"]
    best = max(actions, key=lambda n: res[n][5]["net_benefit_usd"]) if actions else None

    with c2:
        st.markdown(
            f"<div class='card'><div class='eyebrow'>Selected community</div>"
            f"<div style='font-family:Archivo;font-weight:700;font-size:1.3rem'>{lga.lga_name}"
            f"<span style='font-weight:400;color:{SLATE}'> · {lga.state_name}</span></div>"
            f"<div style='display:flex;gap:26px;margin-top:8px;flex-wrap:wrap'>"
            f"<div><small>Population</small><br><span class='num' style='font-size:1.1rem'>{int(lga.population):,}</span></div>"
            f"<div><small>Facilities</small><br><span class='num' style='font-size:1.1rem'>{int(lga.facility_count)}</span></div>"
            f"<div><small>People / facility</small><br><span class='num' style='font-size:1.1rem'>{ppf_str(lga.pop_per_facility)}</span></div>"
            f"<div><small>Risk</small><br><span class='pill' style='background:{C.RISK_COLORS[lga.risk_level]};color:#fff'>{lga.risk_level.upper()}</span></div>"
            f"<div><small>Facility delivery</small><br><span class='num' style='font-size:1.1rem'>{lga.get('facility_delivery', float('nan')):.0f}%</span></div>"
            f"<div><small>Women, 2° educ.</small><br><span class='num' style='font-size:1.1rem'>{lga.get('women_secondary_edu', float('nan')):.0f}%</span></div>"
            f"</div></div>", unsafe_allow_html=True)

        if bypass:
            st.warning(
                f"**Human-led assessment required.** Model confidence here is "
                f"{conf:.0%}, below the {C.BYPASS_CONDITIONS['min_prediction_confidence']:.0%} "
                f"bypass threshold. Figures below are shown for context only and must "
                f"not drive allocation without on-site verification.")

        if best:
            h5 = res[best][5]
            d = h5["deaths_averted"]
            st.markdown(
                f"<div class='hero'><div class='eyebrow' style='color:#7FD1CE'>The cost of doing nothing · 5-year horizon</div>"
                f"<div class='big'>{d['p50']:.0f} preventable child deaths</div>"
                f"<div class='sub'>and {money(h5['cost_of_inaction_usd'])} in health value forgone if the "
                f"recommended plan (<b>{best}</b>) is delayed five years.</div>"
                f"<div style='margin-top:10px;font-size:0.82rem;color:#AFC4D6'>"
                f"Uncertainty range: {d['p10']:.0f}–{d['p90']:.0f} deaths "
                f"(10th–90th percentile, {C.MONTE_CARLO_RUNS:,} Monte-Carlo runs)</div></div>",
                unsafe_allow_html=True)
        else:
            st.info("Set at least one intervention on the left to run the simulation.")

    if best:
        st.write("")
        st.markdown('<div class="eyebrow">Scenario comparison · cumulative over 5 years</div>',
                    unsafe_allow_html=True)
        cols = st.columns(len(scen))
        for col, name in zip(cols, scen):
            h5 = res[name][5]
            d = h5["deaths_averted"]
            is_best = (name == best)
            border = TEAL if is_best else "#E4E9F0"
            tag = (f"<span class='pill' style='background:{TEAL};color:#fff'>RECOMMENDED</span>"
                   if is_best else "")
            col.markdown(
                f"<div class='card' style='border:2px solid {border}'>"
                f"<div style='font-family:Archivo;font-weight:700'>{name}</div>{tag}"
                f"<div style='margin-top:10px'><small>Deaths averted (p50)</small><br>"
                f"<span class='num' style='font-size:1.5rem;color:{TEXT}'>{d['p50']:.0f}</span>"
                f"<small> &nbsp;{d['p10']:.0f}–{d['p90']:.0f}</small></div>"
                f"<div style='margin-top:8px'><small>Intervention cost</small><br>"
                f"<span class='num'>{money(h5['intervention_cost_usd'])}</span></div>"
                f"<div style='margin-top:6px'><small>Net benefit (p50)</small><br>"
                f"<span class='num' style='color:{TEAL};font-weight:600'>{money(h5['net_benefit_usd'])}</span></div>"
                f"</div>", unsafe_allow_html=True)

        # ── plain-language "how this number is built" ─────────────────────
        chain = explain_chain(lga_d, scen[best], SLOPE)
        d5 = res[best][5]["deaths_averted"]["p50"]

        def step(n, t):
            return (f"<div style='flex:1;min-width:175px'>"
                    f"<div class='num' style='color:{TEAL};font-weight:600;font-size:0.8rem'>{n}</div>"
                    f"<div style='font-size:0.88rem;margin-top:3px'>{t}</div></div>")
        arrow = "<div style='align-self:center;color:#9AA8B8;font-size:1.3rem'>→</div>"

        st.write("")
        st.markdown(
            f"<div class='card'><div class='eyebrow'>How we get this number · {best}</div>"
            f"<div style='display:flex;gap:14px;flex-wrap:wrap;margin-top:8px'>"
            + step("Step 1", f"People per clinic falls from <b>{chain['ppf_before']:,.0f}</b> "
                             f"to <b>{chain['ppf_after']:,.0f}</b>.")
            + arrow
            + step("Step 2", f"Using 37 states of real data, the model estimates about "
                             f"<b>{chain['util_gain']:.1f} more in every 100 births</b> happen in a clinic.")
            + arrow
            + step("Step 3", f"That prevents roughly <b>{chain['deaths_per_year']:.0f} child deaths a year</b> "
                             f"here — about <b>{d5:.0f}</b> over five years.")
            + "</div>"
            f"<div style='margin-top:12px'><small><b>Why women's education shows up in the model tab, not here:</b> "
            f"education explains why a community <i>already</i> uses clinics a lot or a little — it's the starting "
            f"point, not something this plan changes. The levers here are clinics and staff; we control for "
            f"education only so the tool doesn't over-credit building.</small></div></div>",
            unsafe_allow_html=True)

        with st.expander("What's behind these estimates?"):
            st.markdown(
                "- **Births** are estimated from the community's population.\n"
                "- Nigeria's under-five death rate is **110 per 1,000** live births (DHS 2023–24), and about "
                "**41%** of those deaths are preventable with basic care and vaccines (Global Burden of Disease).\n"
                "- Deaths are converted to a money figure using a standard value per healthy life-year.\n"
                "- Every figure is recomputed **2,000 times** with these inputs varied across plausible ranges — "
                "that is why you see a low–high range, not a single number.\n"
                "- Full sources and limitations are in the **Model & Governance** tab.")

        st.write("")
        cc1, cc2 = st.columns([3, 2])
        rows = []
        for name in actions:
            for y in C.HORIZONS_YEARS:
                dd = res[name][y]["deaths_averted"]
                rows.append({"Year": y, "Scenario": name,
                             "p10": dd["p10"], "p50": dd["p50"], "p90": dd["p90"]})
        pdf = pd.DataFrame(rows)

        with cc1:
            st.markdown('<div class="eyebrow">Cumulative deaths averted over time</div>',
                        unsafe_allow_html=True)
            band = (alt.Chart(pdf[pdf.Scenario == best])
                    .mark_area(opacity=0.22, color=TEAL)
                    .encode(x=alt.X("Year:O", title="Years from now",
                                    axis=alt.Axis(labelAngle=-90)),
                            y=alt.Y("p10:Q", title="Deaths averted (cumulative)"),
                            y2="p90:Q"))
            lines = (alt.Chart(pdf).mark_line(point=True, strokeWidth=2.5)
                    .encode(x=alt.X("Year:O", axis=alt.Axis(labelAngle=-90)),
                            y="p50:Q",
                            color=alt.Color("Scenario:N",
                                            scale=alt.Scale(range=[TEAL, AMBER, "#7FB3FF", "#C792EA"]),
                                            legend=alt.Legend(orient="bottom"))))
            st.altair_chart(dark_chart(band + lines, 300), use_container_width=True)
            st.caption(f"Shaded band = 10th–90th percentile for the recommended plan ({best}).")

        with cc2:
            st.markdown('<div class="eyebrow">Act now vs. wait 5 years</div>',
                        unsafe_allow_html=True)
            h5 = res[best][5]
            st.markdown(
                f"<div class='card'>"
                f"<div style='display:flex;justify-content:space-between;padding:6px 0'>"
                f"<span>Act now — deaths averted</span>"
                f"<b class='num'>{h5['deaths_averted']['p50']:.0f}</b></div>"
                f"<div style='display:flex;justify-content:space-between;padding:6px 0;border-top:1px solid #eee'>"
                f"<span>Wait 5 years — deaths averted</span><b class='num'>~0</b></div>"
                f"<div style='display:flex;justify-content:space-between;padding:8px 0;border-top:2px solid {LINE};margin-top:4px'>"
                f"<span><b>Cost of the delay</b></span>"
                f"<b class='num' style='color:#d62728'>{h5['deaths_averted']['p50']:.0f} lives · {money(h5['cost_of_inaction_usd'])}</b></div>"
                f"<div style='margin-top:8px'><small>Benefits compound annually; a plan started in year 5 "
                f"delivers almost none of its value within this 5-year window.</small></div>"
                f"</div>", unsafe_allow_html=True)

        st.caption("All figures are modelled projections with documented assumptions "
                   "(see Model & Governance). They are not guaranteed outcomes.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — MODEL & GOVERNANCE
# ══════════════════════════════════════════════════════════════════════════
with tab_model:
    g1, g2 = st.columns(2)

    with g1:
        st.markdown('<div class="eyebrow">How the system works</div>', unsafe_allow_html=True)
        st.markdown(
            "Risk classification is **rule-based** (people per facility vs WHO-aligned "
            "thresholds) — transparent and auditable. A separate **learned model** predicts "
            "care utilisation from access structure, controlling for the dominant driver "
            "(women's education), and supplies the slope the simulator uses. The simulator "
            "then projects deaths, DALYs, and cost of inaction over 1/3/5 years via "
            "Monte-Carlo, so every figure is a range.")
        st.markdown("**Pipeline:** GRID3 + WorldPop + HDX + DHS → features → "
                    "utilisation model → cost-of-inaction simulator → decision support.")

        st.markdown('<div class="eyebrow" style="margin-top:14px">Model evaluation</div>',
                    unsafe_allow_html=True)
        e = st.columns(3)
        e[0].metric("LOO R²", f"{M.get('loo_r2', float('nan')):.2f}")
        e[1].metric("LOO MAE", f"{M.get('loo_mae', float('nan')):.1f} pp",
                    delta=f"vs {M.get('naive_mae', float('nan')):.1f} naive",
                    delta_color="off")
        e[2].metric("States", M.get("n_states", "—"))
        st.caption(f"Nested leave-one-state-out CV · target: {M.get('target','—')} · "
                   f"Ridge α={M.get('chosen_alpha',0):.1f} chosen by cross-validation.")

    with g2:
        st.markdown('<div class="eyebrow">What explains where women already give birth in a clinic</div>',
                    unsafe_allow_html=True)
        coefs = M.get("coefficients_per_feature_sd", {})
        label = {"women_secondary_edu": "Women's 2° education",
                 "mean_log_ppf": "Access (people/facility)",
                 "mean_facility_density": "Facility density"}
        cdf = pd.DataFrame([{"Driver": label.get(kk, kk), "Effect": vv}
                            for kk, vv in coefs.items()])
        if len(cdf):
            bar = (alt.Chart(cdf).mark_bar(cornerRadius=2)
                   .encode(x=alt.X("Effect:Q", title="Std. effect on facility delivery (pp)"),
                           y=alt.Y("Driver:N", sort="-x", title=None),
                           color=alt.condition(alt.datum.Effect > 0,
                                               alt.value(TEAL), alt.value(ALARM))))
            st.altair_chart(dark_chart(bar, 160), use_container_width=True)
        st.markdown(
            "<small><b>How to read this:</b> women's education is the biggest reason some "
            "states already have high facility use — it's <b>context the tool can't change</b>, "
            "not a lever. Building clinics is a real but <b>bounded</b> lever. We include education "
            "here only so the simulator uses the access effect <i>after</i> accounting for it — it is "
            "never counted twice.</small>",
            unsafe_allow_html=True)

    st.divider()
    gg1, gg2 = st.columns(2)

    with gg1:
        st.markdown('<div class="eyebrow">Human-in-the-loop</div>', unsafe_allow_html=True)
        st.markdown(
            "- The AI **informs**; it never authorises resource moves.\n"
            "- Every deployment passes **two review cycles**: state PHC technical "
            "review, then medical-board authorisation.\n"
            "- **Non-goals:** no individual patient data; not real-time surveillance; "
            "does not rank human worth — it ranks structural access gaps.")

        st.markdown('<div class="eyebrow" style="margin-top:12px">Bypass conditions — when NOT to use</div>',
                    unsafe_allow_html=True)
        bc = C.BYPASS_CONDITIONS
        st.markdown(
            f"- Data completeness &lt; {bc['min_data_completeness']:.0%}\n"
            f"- Model confidence &lt; {bc['min_prediction_confidence']:.0%}\n"
            f"- A shock (flood, conflict, displacement) that invalidates the baseline\n"
            f"- Population basis older than {bc['max_population_data_age_years']} years")

    with gg2:
        st.markdown('<div class="eyebrow">Lifecycle & drift monitoring</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Re-fit on each DHS release. Monthly check: if the distribution of predicted "
            "utilisation shifts beyond a set tolerance from the training distribution, "
            "raise an alert and require re-evaluation before new recommendations are issued.")
        st.markdown('<div class="eyebrow" style="margin-top:12px">LGAs flagged for board review</div>',
                    unsafe_allow_html=True)
        rev = (lgas[lgas.needs_human_review]
               .sort_values("prediction_confidence")
               [["lga_name", "state_name", "risk_level", "prediction_confidence"]]
               .head(20).reset_index(drop=True))
        rev.index += 1
        rev.columns = ["LGA", "State", "Risk", "Confidence"]
        rev["Confidence"] = rev["Confidence"].map(lambda x: f"{x:.0%}")
        st.dataframe(rev, use_container_width=True, height=240)


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — FIELD REPORTS  (offline / zero-signal capture)
# ══════════════════════════════════════════════════════════════════════════
with tab_field:
    st.markdown('<div class="eyebrow">Offline community health report</div>',
                unsafe_allow_html=True)
    st.markdown(
        "<small>Closes the exclusion-bias gap: communities with no network signal "
        "generate no digital footprint, so field workers report them by hand. "
        "Zero-signal zones are escalated to critical priority.</small>",
        unsafe_allow_html=True)

    with st.form("field_report"):
        a, b, cc = st.columns(3)
        with a:
            lga_in = st.text_input("LGA name *")
            state_in = st.text_input("State *")
            community = st.text_input("Community / settlement")
        with b:
            headcount = st.number_input("Estimated population", min_value=0, step=100)
            clinic = st.selectbox("Functional clinic?", ["Yes", "No", "Partially"])
            last_supply = st.date_input("Last vaccine / drug supply")
        with cc:
            zero_signal = st.checkbox("⚠ Zero network signal in this area")
            staff = st.number_input("Health staff present", min_value=0, step=1)
            notes = st.text_area("Field notes", height=90)

        submitted = st.form_submit_button("Submit field report", type="primary")
        if submitted:
            if not lga_in or not state_in:
                st.error("LGA name and State are required.")
            else:
                rec = {"lga": lga_in, "state": state_in, "community": community,
                       "population": headcount, "clinic": clinic, "staff": staff,
                       "last_supply": str(last_supply), "zero_signal": zero_signal,
                       "notes": notes, "submitted": pd.Timestamp.now().isoformat()}
                fp = C.PROCESSED / "field_reports.csv"
                pd.DataFrame([rec]).to_csv(fp, mode="a", header=not fp.exists(),
                                           index=False)
                flag = " · escalated to CRITICAL (zero signal)" if zero_signal else ""
                st.success(f"Report saved for {community or lga_in}, {state_in}.{flag}")
                if zero_signal:
                    st.info("This zero-signal community is now elevated to critical "
                            "priority for board review, independent of any digital data.")