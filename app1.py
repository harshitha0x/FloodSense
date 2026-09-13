import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import shap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from streamlit_autorefresh import st_autorefresh
from datetime import datetime
import os, json, requests
import folium
from streamlit_folium import st_folium
from data_ingestion import (
    CHEMB_CAPACITY_MCM,
    fetch_chembarambakkam_level,
    fetch_rainfall_for,
)

#  LOCATION DATA & FETCH HELPERS
CHENNAI_LOCATIONS = {
    "Chennai":    {"lat": 13.0827, "lon": 80.2707, "vulner": 1.00},
    "Velachery":                {"lat": 13.0050, "lon": 80.2207, "vulner": 1.25},
    "Adyar":                    {"lat": 13.0450, "lon": 80.2450, "vulner": 1.20},
    "Ambattur":                 {"lat": 13.1135, "lon": 80.1548, "vulner": 1.05},
    "Royapuram (Coastal)":      {"lat": 13.1300, "lon": 80.2900, "vulner": 1.15},
    "Pallikaranai":             {"lat": 13.0300, "lon": 80.2600, "vulner": 1.22},
    "T.Nagar":                  {"lat": 13.0600, "lon": 80.2700, "vulner": 1.00},
    "Perambur":                 {"lat": 13.0900, "lon": 80.2400, "vulner": 1.08},
    "Tambaram":                 {"lat": 13.0100, "lon": 80.2500, "vulner": 1.10},
    "Tiruvottiyur":             {"lat": 13.1800, "lon": 80.2700, "vulner": 1.15},
    "Sholinganallur (IT)":      {"lat": 13.0200, "lon": 80.2700, "vulner": 1.08},
    "Poonamallee":              {"lat": 13.1200, "lon": 80.2100, "vulner": 1.00},
    "Mudichur":                 {"lat": 13.0200, "lon": 80.2100, "vulner": 1.30},
}

CHENNAI_REGIONS = {
    "North":   {"lat": 13.1135, "lon": 80.1548, "vulner": 1.05},  # Ambattur
    "Central": {"lat": 13.0827, "lon": 80.2707, "vulner": 1.00},  # City Centre
    "South":   {"lat": 13.0050, "lon": 80.2207, "vulner": 1.25},  # Velachery / Mudichur
    "Coastal": {"lat": 13.1300, "lon": 80.2900, "vulner": 1.15},  # Royapuram
}

def nearest_location(lat, lon):
    best_name, best, best_d = None, None, 1e9
    for name, c in CHENNAI_LOCATIONS.items():
        d = (c["lat"] - lat) ** 2 + (c["lon"] - lon) ** 2
        if d < best_d:
            best_name, best, best_d = name, c, d
    return best_name, best

@st.cache_data(ttl=3600)
def detect_user_location():
    try:
        r = requests.get("https://ipapi.co/json/", timeout=6)
        r.raise_for_status()
        d = r.json()
        return {
            "lat": float(d["latitude"]),
            "lon": float(d["longitude"]),
            "city": d.get("city") or "Unknown",
        }
    except Exception:
        return None

def probe_live_apis():
    rain_ok = fetch_rainfall_for(13.0827, 80.2707) is not None
    res_ok = fetch_chembarambakkam_level() is not None
    return rain_ok, res_ok

#  PAGE CONFIG 
st.set_page_config(
    page_title="FloodSense ",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#  GLOBAL CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Mono', monospace;
    background-color: #060d1a;
    color: #e8f4ff;
}

/* hide default streamlit header/footer */
#MainMenu, footer, header { visibility: hidden; }

.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── CUSTOM HEADER ── */
.app-header {
    background: linear-gradient(135deg, #0b1628 0%, #0f2040 100%);
    border: 1px solid rgba(0,180,255,0.15);
    border-radius: 16px;
    padding: 28px 36px;
    margin-bottom: 28px;
    display: flex;
    align-items: center;
    gap: 24px;
    position: relative;
    overflow: hidden;
}
.app-header::before {
    content:'';
    position:absolute; top:0; left:0; right:0; height:2px;
    background: linear-gradient(90deg, transparent, #00b4ff, #00ffc8, transparent);
}
.app-header img { border-radius: 10px; object-fit: cover; }
.header-text h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
    color: #fff; margin: 0;
    line-height: 1.1;
}
.header-text h1 span { color: #00b4ff; }
.header-text p { color: #5a7a9a; font-size: 0.75rem; letter-spacing: 2px; margin-top: 6px; }
.status-pill {
    margin-left: auto;
    background: rgba(0,180,255,0.08);
    border: 1px solid rgba(0,180,255,0.2);
    border-radius: 40px;
    padding: 8px 18px;
    font-size: 0.65rem; letter-spacing: 2px; color: #00b4ff;
}

/* ── SECTION LABELS ── */
.section-label {
    font-size: 0.6rem; letter-spacing: 4px; color: #5a7a9a;
    text-transform: uppercase; margin-bottom: 14px;
    display: flex; align-items: center; gap: 10px;
}
.section-label::after {
    content:''; flex:1; height:1px; background: rgba(0,180,255,0.12);
}

/* ── INPUT CARDS ── */
.input-card {
    background: #0b1628;
    border: 1px solid rgba(0,180,255,0.1);
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
}

/* ── RESULT CARDS ── */
.risk-box {
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    border: 2px solid;
    margin-bottom: 20px;
}
.risk-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem; font-weight: 800;
}
.risk-score {
    font-size: 4rem; font-weight: 700;
    font-family: 'Syne', sans-serif; line-height: 1;
}
.risk-subtitle { font-size: 0.7rem; letter-spacing: 3px; opacity: 0.6; margin-top: 4px; }

.metric-card {
    background: #0f1e35;
    border: 1px solid rgba(0,180,255,0.1);
    border-radius: 12px;
    padding: 18px;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem; font-weight: 800;
}
.metric-label { font-size: 0.6rem; letter-spacing: 2px; color: #5a7a9a; margin-top: 4px; }

.advice-box {
    background: #0b1628;
    border-left: 4px solid;
    border-radius: 0 12px 12px 0;
    padding: 20px 24px;
    margin-top: 16px;
    font-size: 0.85rem;
    line-height: 1.8;
    color: #b0c8e0;
}

/* ── PREDICT BUTTON ── */
.stButton > button {
    width: 100%;
    background: transparent !important;
    border: 1px solid #00b4ff !important;
    color: #00b4ff !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.9rem !important;
    font-weight: 700 !important;
    letter-spacing: 4px !important;
    padding: 16px !important;
    border-radius: 12px !important;
    transition: all 0.3s !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00b4ff, #00ffc8) !important;
    color: #000 !important;
    border-color: transparent !important;
}

/* ── SLIDERS ── */
.stSlider > div > div > div > div { background: #00b4ff !important; }

/* ── LOG TABLE ── */
.log-table { width:100%; border-collapse: collapse; font-size: 0.72rem; }
.log-table th {
    background: #0f1e35; color: #5a7a9a;
    letter-spacing: 2px; padding: 10px 14px; text-align: left;
    border-bottom: 1px solid rgba(0,180,255,0.1);
}
.log-table td {
    padding: 10px 14px;
    border-bottom: 1px solid rgba(0,180,255,0.06);
    color: #b0c8e0;
}
.log-table tr:hover td { background: rgba(0,180,255,0.04); }

.stTabs [data-baseweb="tab-list"] {
    background-color: transparent !important;
    gap: 15px !important; \
    padding: 10px 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background-color: #0b1628 !important;
    border: 1px solid rgba(0,180,255,0.15) !important;
    border-radius: 12px !important;
    padding: 12px 25px !important;
    color: #5a7a9a !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}

.stTabs [aria-selected="true"] {
    background-color: rgba(0,180,255,0.12) !important;
    border-color: #00b4ff !important;
    color: #00b4ff !important;
    box-shadow: 0 0 15px rgba(0,180,255,0.1) !important;
}

.stTabs [data-baseweb="tab"]:hover {
    border-color: #00b4ff !important;
    color: #e8f4ff !important;
}
""", unsafe_allow_html=True)

#  LOAD MODEL
@st.cache_resource
def load_model():
    return joblib.load("flood_model_newbalanced.pkl")

model = load_model()
if "api_ok" not in st.session_state:
    rain_ok, res_ok = probe_live_apis()
    st.session_state["api_ok"] = rain_ok and res_ok
    st.session_state["api_rain_ok"] = rain_ok
    st.session_state["api_res_ok"] = res_ok

MAX_CAP = 10568
LOG_FILE = "flood_log.csv"

#  HELPERS
def compute_risk(ml_score, res_level, vulnerability=1.0): # added vulnerability
    res_pct = (res_level / MAX_CAP) * 100
    final = (0.6 * ml_score + 0.4 * res_pct) * vulnerability
    final = min(final, 100.0) # ensure it doesn't exceed 100%
    return round(final, 1), round(res_pct, 1)

def risk_level(final_risk, res_pct):
    if res_pct > 96:
        return "HIGH", "#ff3b3b", ""
    elif res_pct > 92 or final_risk >= 70:
        return "MEDIUM-HIGH", "#ff8800", ""
    elif final_risk >= 40:
        return "MEDIUM", "#ffaa00", ""
    else:
        return "LOW", "#00e87a", ""

def get_advice(level):
    return {
        "HIGH": "Reservoir is at critical capacity. Immediately evacuate low-lying areas. Alert emergency services and activate disaster management protocols. Do NOT wait.",
        "MEDIUM-HIGH": " Reservoir is very high and rainfall is significant. Authorities should activate flood alert systems. Residents near rivers/drains should be ready to move.",
        "MEDIUM": " Moderate risk. Monitor weather and reservoir updates every 6 hours. Avoid low-lying areas. Keep emergency kits ready.",
        "LOW": " Conditions are currently safe. Rainfall and reservoir levels are within normal range. Continue routine monitoring."
    }[level]

def save_log(entry):
    df_new = pd.DataFrame([entry])
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df_new, df], ignore_index=True)
    else:
        df = df_new
    df.head(50).to_csv(LOG_FILE, index=False)

def load_logs():
    if os.path.exists(LOG_FILE):
        try:
            return pd.read_csv(LOG_FILE).to_dict(orient="records")
        except Exception:
            return []
    return []

def get_confidence(input_df):
    tree_preds = [t.predict_proba(input_df)[0][1] * 100 for t in model.estimators_]
    std_val = np.std(tree_preds)
    return {
        "mean": round(np.mean(tree_preds), 1),
        "lower": round(np.percentile(tree_preds, 10), 1),
        "upper": round(np.percentile(tree_preds, 90), 1),
        "std": round(std_val, 1),
        "uncertain": std_val > 15 
    }

#HEADER
col_txt, col_mapbtn = st.columns([4, 1])

with col_txt:
    st.markdown("""
    <div style="padding: 10px 0">
        <div style="font-family:'Syne',sans-serif; font-size:2.4rem; font-weight:800; color:#fff; line-height:1">
            Flood<span style="color:#00b4ff">Sense</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"<div style='font-size:0.65rem;color:#5a7a9a;margin-top:4px'> {datetime.now().strftime('%d %b %Y, %I:%M %p')}</div>", unsafe_allow_html=True)

with col_mapbtn:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    current_view = st.session_state.get("current_view", "predictor")
    btn_label = "PREDICTOR" if current_view == "map" else "FLOOD MAP"
    if st.button(btn_label, key="toggle_main_view", use_container_width=True):
        st.session_state["current_view"] = "map" if current_view == "predictor" else "predictor"
        st.rerun()
 
#TOGGLEABLE TOP MAP
if st.session_state.get('show_top_map', False):
    st.markdown('<div class="section-label">REGIONAL REAL-TIME FLOOD OVERLAY</div>', unsafe_allow_html=True)
    top_map = folium.Map(
    location=[13.0827, 80.2707],
    zoom_start=7,
    min_zoom=7,
    max_zoom=7,
    tiles="CartoDB dark_matter",
    )
    
    latest_ts = int(time.time() // 600 * 600) 
    folium.TileLayer(
        tiles=f'https://tilecache.rainviewer.com/v2/radar/{latest_ts}/512/{{z}}/{{x}}/{{y}}/2/1_1.png',
        attr='RainViewer', name='Live Precipitation Radar', overlay=True, control=True, opacity=0.6, max_zoom=7
    ).add_to(top_map)

    for name, coords in CHENNAI_LOCATIONS.items():
        color = st.session_state['result']['color'] if 'result' in st.session_state else "#00b4ff"
        folium.CircleMarker(location=[coords["lat"], coords["lon"]], radius=10, color=color, fill=True, fill_opacity=0.7, tooltip=f"<b>{name}</b>").add_to(top_map)

    st_folium(top_map, width="100%", height=450, key="top_toggle_map")
    
    st.sidebar.success("Done viewing? Click 'app1' above to return to the Predictor.")

st.markdown("---")

#  TABS
if st.session_state.get("current_view", "predictor") == "map":
    import map_features
    map_features.render_map_view()
    st.stop()

tab1, tab2, tab3 = st.tabs(["  PREDICT", "  ANALYSIS & SHAP", "  HISTORY LOG"])
#  TAB 1 — PREDICT
with tab1:
    st_autorefresh(interval=600000, key="floodcheck")
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown('<div class="section-label">RAINFALL PARAMETERS</div>', unsafe_allow_html=True)
        rain_today, rain_3, rain_5, rain_7 = 10.0, 20.0, 50.0, 100.0
        res = 3000.0
        #INPUT MODE SELECTOR
        input_mode = st.radio(
            "Input Mode",
            [" Manual", "Live — Chennai", " Choose Location", "Auto - My location"],
            horizontal=True,
            label_visibility="collapsed"
        )

        fetched = None

        #  MODE 1 — MANUAL
        if input_mode == "Manual":
            st.markdown(
                '<div style="font-size:0.6rem;color:#5a7a9a;letter-spacing:1px;margin-bottom:10px">'
                'Drag to enter values manually</div>',
                unsafe_allow_html=True
            )
            rain_today = st.slider("Rainfall Today (mm)",        0.0, 500.0,  10.0,  0.5)
            if rain_today > 450:
                st.markdown("""
                    <div style="background:rgba(255,59,59,0.1); border-left:4px solid #ff3b3b; padding:10px 15px; margin-top:5px; margin-bottom:15px;">
                        <span style="color:#ff3b3b; font-weight:700;">⚠️ EXTREME VALUE</span><br>
                        <span style="color:#5a7a9a; font-size:0.7rem;">Exceeds Chennai's 2015 record. Please verify inputs.</span>
                    </div>
                """, unsafe_allow_html=True)
            rain_3     = st.slider("3-Day Cumulative (mm)",       0.0, 1000.0, 20.0,  1.0)
            rain_5     = st.slider("5-Day Cumulative (mm)",       0.0, 2000.0, 50.0,  1.0)
            rain_7     = st.slider("7-Day Cumulative (mm)",       0.0, 3000.0, 100.0, 1.0)

        #  MODE 2 — LIVE CHENNAI
        elif input_mode == "Live — Chennai":
            with st.spinner("Fetching live Chennai rainfall…"):
                fetched = fetch_rainfall_for(region_coords["lat"], region_coords["lon"])

            if fetched:
                st.markdown(
                    f'<div style="font-size:0.6rem;color:#1d9e75;letter-spacing:1px;margin-bottom:10px">'
                    f'✓ OPEN-METEO · CHENNAI · Updated {fetched["fetched_at"]}</div>',
                    unsafe_allow_html=True
                )
                rain_today = fetched["rain_today"]
                rain_3     = fetched["rain_3"]
                rain_5     = fetched["rain_5"]
                rain_7     = fetched["rain_7"]
            else:
                st.warning("Live fetch failed — using defaults.")
                rain_today, rain_3, rain_5, rain_7 = 10.0, 20.0, 50.0, 100.0

            st.slider("Rainfall Today (mm)",  0.0, 500.0,  float(min(rain_today, 500.0)),  0.5,  disabled=True)
            st.slider("3-Day Cumulative (mm)", 0.0, 1000.0, float(min(rain_3,     1000.0)), 1.0,  disabled=True)
            st.slider("5-Day Cumulative (mm)", 0.0, 2000.0, float(min(rain_5,     2000.0)), 1.0,  disabled=True)
            st.slider("7-Day Cumulative (mm)", 0.0, 3000.0, float(min(rain_7,     3000.0)), 1.0,  disabled=True)

            # 7-day sparkline
            if fetched:
                fig_s, ax_s = plt.subplots(figsize=(5, 1.2))
                fig_s.patch.set_facecolor('#0b1628')
                ax_s.set_facecolor('#0b1628')
                day_labels = [d[5:] for d in fetched["dates"]]
                dm = fetched["daily_mm"]
                bar_colors = ['#ff3b3b' if v > 100 else '#ffaa00' if v > 40 else '#00b4ff' for v in dm]
                ax_s.bar(day_labels, dm, color=bar_colors, width=0.6)
                ax_s.set_ylabel("mm", color='#5a7a9a', fontsize=7)
                ax_s.tick_params(colors='#b0c8e0', labelsize=6.5)
                for spine in ax_s.spines.values(): spine.set_visible(False)
                plt.xticks(rotation=30)
                plt.tight_layout()
                st.pyplot(fig_s)
                plt.close()

        #  MODE 3 — PICK ON MAP
        elif input_mode == "Choose Location":
            st.markdown(
                '<div style="font-size:0.6rem;color:#5a7a9a;letter-spacing:1px;margin-bottom:8px">'
                '🖱 Click anywhere on the map to select your location</div>',
                unsafe_allow_html=True
            )

            # Retrieve previously clicked point from session state
            clicked_lat = st.session_state.get("map_lat", None)
            clicked_lon = st.session_state.get("map_lon", None)

            # Build folium map centred on Chennai
            map_center = [clicked_lat, clicked_lon] if clicked_lat else [13.0827, 80.2707]
            m = folium.Map(
            Location=zone_data["center"],
            zoom_start=7,
            min_zoom=7,
            max_zoom=7,
            tiles="CartoDB dark_matter",
            control_scale=True
            )

            # Known zone markers as reference points
            for name, coords in CHENNAI_LOCATIONS.items():
                label = name.split(" ", 1)[-1]  # strip emoji
                folium.CircleMarker(
                    location=[coords["lat"], coords["lon"]],
                    radius=5,
                    color="#00b4ff",
                    fill=True,
                    fill_color="#00b4ff",
                    fill_opacity=0.4,
                    tooltip=label,
                ).add_to(m)

            # Show previously clicked marker in teal
            if clicked_lat:
                folium.Marker(
                    location=[clicked_lat, clicked_lon],
                    icon=folium.Icon(color="green", icon="map-marker", prefix="fa"),
                    tooltip=f"📍 Selected: {clicked_lat:.4f}°N, {clicked_lon:.4f}°E"
                ).add_to(m)

            # Render map and capture click
            map_result = st_folium(
                m,
                width=None,
                height=260,
                returned_objects=["last_clicked"],
                key="location_picker_map"
            )

            # Update session state when user clicks
            if map_result and map_result.get("last_clicked"):
                st.session_state["map_lat"] = round(map_result["last_clicked"]["lat"], 4)
                st.session_state["map_lon"] = round(map_result["last_clicked"]["lng"], 4)
                clicked_lat = st.session_state["map_lat"]
                clicked_lon = st.session_state["map_lon"]
                st.rerun()

            # Fetch & display only once a point is selected
            if clicked_lat and clicked_lon:
                st.markdown(
                    f'<div style="font-size:0.6rem;color:#5a7a9a;letter-spacing:1px;margin:6px 0">'
                    f'📡 Fetching data for <b style="color:#00b4ff">{clicked_lat}°N, {clicked_lon}°E</b></div>',
                    unsafe_allow_html=True
                )
                with st.spinner("Fetching rainfall for selected location…"):
                    fetched = fetch_rainfall_for(clicked_lat, clicked_lon)

                if fetched:
                    st.markdown(
                        f'<div style="font-size:0.6rem;color:#1d9e75;letter-spacing:1px;margin-bottom:8px">'
                        f'✓ OPEN-METEO · Updated {fetched["fetched_at"]}</div>',
                        unsafe_allow_html=True
                    )
                    rain_today = fetched["rain_today"]
                    rain_3     = fetched["rain_3"]
                    rain_5     = fetched["rain_5"]
                    rain_7     = fetched["rain_7"]
                else:
                    st.warning("⚠ Live fetch failed — using defaults.")
                    rain_today, rain_3, rain_5, rain_7 = 10.0, 20.0, 50.0, 100.0

                st.slider("Rainfall Today (mm)",  0.0, 500.0,  float(min(rain_today, 500.0)),  0.5,  disabled=True)
                st.slider("3-Day Cumulative (mm)", 0.0, 1000.0, float(min(rain_3,     1000.0)), 1.0,  disabled=True)
                st.slider("5-Day Cumulative (mm)", 0.0, 2000.0, float(min(rain_5,     2000.0)), 1.0,  disabled=True)
                st.slider("7-Day Cumulative (mm)", 0.0, 3000.0, float(min(rain_7,     3000.0)), 1.0,  disabled=True)

                # 7-day sparkline
                if fetched:
                    fig_s, ax_s = plt.subplots(figsize=(5, 1.2))
                    fig_s.patch.set_facecolor('#0b1628')
                    ax_s.set_facecolor('#0b1628')
                    day_labels = [d[5:] for d in fetched["dates"]]
                    dm = fetched["daily_mm"]
                    bar_colors = ['#ff3b3b' if v > 100 else '#ffaa00' if v > 40 else '#00b4ff' for v in dm]
                    ax_s.bar(day_labels, dm, color=bar_colors, width=0.6)
                    ax_s.set_ylabel("mm", color='#5a7a9a', fontsize=7)
                    ax_s.tick_params(colors='#b0c8e0', labelsize=6.5)
                    for spine in ax_s.spines.values(): spine.set_visible(False)
                    plt.xticks(rotation=30)
                    plt.tight_layout()
                    st.pyplot(fig_s)
                    plt.close()

                # info card
                st.markdown(f"""
                <div style="background:#0f1e35;border:1px solid rgba(0,180,255,0.1);border-radius:10px;
                padding:10px 14px;margin-top:6px;font-size:0.68rem;color:#5a7a9a;line-height:1.8">
                    📌 <b style="color:#00b4ff">{clicked_lat}°N, {clicked_lon}°E</b><br>
                    🌧 Today: <b style="color:#e8f4ff">{rain_today} mm</b> &nbsp;|&nbsp;
                    📅 7-Day Total: <b style="color:#e8f4ff">{rain_7} mm</b>
                </div>
                """, unsafe_allow_html=True)

                # reset button
                if st.button("🗑 Clear selected location", key="clear_map_pin"):
                    st.session_state.pop("map_lat", None)
                    st.session_state.pop("map_lon", None)
                    st.rerun()
            else:
                # No point selected yet — show placeholder sliders
                st.markdown(
                    '<div style="font-size:0.65rem;color:#5a7a9a;padding:10px 0">'
                    '⬆ Click a point on the map above to load live rainfall data.</div>',
                    unsafe_allow_html=True
                )
                rain_today, rain_3, rain_5, rain_7 = 10.0, 20.0, 50.0, 100.0
                st.slider("Rainfall Today (mm)",  0.0, 500.0,  rain_today, 0.5,  disabled=True)
                st.slider("3-Day Cumulative (mm)", 0.0, 1000.0, rain_3,    1.0,  disabled=True)
                st.slider("5-Day Cumulative (mm)", 0.0, 2000.0, rain_5,    1.0,  disabled=True)
                st.slider("7-Day Cumulative (mm)", 0.0, 3000.0, rain_7,    1.0,  disabled=True)
        elif input_mode == "Auto — My location":
            loc = detect_user_location()
            if loc:
                st.session_state["map_lat"] = loc["lat"]
                st.session_state["map_lon"] = loc["lon"]
                st.caption(f"Detected: {loc['city']} ({loc['lat']:.4f}, {loc['lon']:.4f})")
                fetched = fetch_rainfall_for(loc["lat"], loc["lon"])
                if fetched:
                    rain_today = fetched["rain_today"]
                    rain_3 = fetched["rain_3"]
                    rain_5 = fetched["rain_5"]
                    rain_7 = fetched["rain_7"]
                else:
                    st.warning("Live fetch failed — switch to Manual.")
                    rain_today, rain_3, rain_5, rain_7 = 10.0, 20.0, 50.0, 100.0
            else:
                st.warning("Could not detect location — using Manual defaults.")
                rain_today, rain_3, rain_5, rain_7 = 10.0, 20.0, 50.0, 100.0

        st.markdown('<div class="section-label" style="margin-top:20px">RESERVOIR LEVEL</div>', unsafe_allow_html=True)

        if input_mode == "Choose Location":
            with st.spinner("Fetching Chembarambakkam live level…"):
                res_data = fetch_chembarambakkam_level()

            if res_data:
                res = res_data["mcm"]
                res_pct_live = res_data["pct"]
                bar_color = "#ff3b3b" if res_pct_live > 96 else "#ffaa00" if res_pct_live > 85 else "#00b4ff"
                st.markdown(f"""
                <div style="background:#0f1e35;border:1px solid rgba(0,180,255,0.1);border-radius:10px;
                padding:12px 16px;margin-top:4px">
                    <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#5a7a9a;margin-bottom:8px">
                        <span>CHEMBARAMBAKKAM · LIVE</span>
                        <span style="color:{bar_color};font-weight:700">{res_pct_live:.1f}%</span>
                    </div>
                    <div style="background:#060d1a;border-radius:6px;height:10px;overflow:hidden">
                        <div style="width:{min(res_pct_live,100)}%;height:100%;
                        background:linear-gradient(90deg,{bar_color}88,{bar_color});border-radius:6px"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:0.6rem;color:#5a7a9a;margin-top:6px">
                        <span>💧 {res} MCM</span>
                        <span style="color:#ff3b3b">⚠ 96% CRITICAL</span>
                        <span>MAX {CHEMB_CAPACITY_MCM:.0f} MCM</span>
                    </div>
                    <div style="font-size:0.55rem;color:#2a6a5a;margin-top:6px">
                        ✓ SOURCE: {res_data['source']} · {res_data['fetched_at']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Fetch failed — fall back to slider
                st.markdown(
                    '<div style="font-size:0.6rem;color:#ff8800;margin-bottom:6px">'
                    'Live reservoir fetch failed — enter manually</div>',
                    unsafe_allow_html=True
                )
                res = st.slider("Reservoir Level (MCM)", 0.0, float(MAX_CAP), 3000.0, 10.0)
                res_pct_live = (res / MAX_CAP) * 100
                bar_color = "#ff3b3b" if res_pct_live > 96 else "#ffaa00" if res_pct_live > 85 else "#00b4ff"
                st.markdown(f"""
                <div style="background:#0f1e35;border-radius:10px;padding:12px 16px;margin-top:4px;border:1px solid rgba(0,180,255,0.1)">
                    <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#5a7a9a;margin-bottom:8px">
                        <span>RESERVOIR CAPACITY</span>
                        <span style="color:{bar_color};font-weight:700">{res_pct_live:.1f}%</span>
                    </div>
                    <div style="background:#060d1a;border-radius:6px;height:10px;overflow:hidden">
                        <div style="width:{min(res_pct_live,100)}%;height:100%;background:linear-gradient(90deg,{bar_color}88,{bar_color});border-radius:6px"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:0.6rem;color:#5a7a9a;margin-top:6px">
                        <span>0</span><span style="color:#ff3b3b">⚠ 96% CRITICAL</span><span>MAX 10568 MCM</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        elif input_mode == "Live — Chennai":
            # Auto-fetch reservoir for Live Chennai mode too
            with st.spinner("Fetching Chembarambakkam live level…"):
                res_data = fetch_chembarambakkam_level()

            if res_data:
                res = res_data["mcm"]
                res_pct_live = res_data["pct"]
                bar_color = "#ff3b3b" if res_pct_live > 96 else "#ffaa00" if res_pct_live > 85 else "#00b4ff"
                st.markdown(f"""
                <div style="background:#0f1e35;border:1px solid rgba(0,180,255,0.1);border-radius:10px;
                padding:12px 16px;margin-top:4px">
                    <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#5a7a9a;margin-bottom:8px">
                        <span>CHEMBARAMBAKKAM · LIVE</span>
                        <span style="color:{bar_color};font-weight:700">{res_pct_live:.1f}%</span>
                    </div>
                    <div style="background:#060d1a;border-radius:6px;height:10px;overflow:hidden">
                        <div style="width:{min(res_pct_live,100)}%;height:100%;
                        background:linear-gradient(90deg,{bar_color}88,{bar_color});border-radius:6px"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:0.6rem;color:#5a7a9a;margin-top:6px">
                        <span>💧 {res} MCM</span>
                        <span style="color:#ff3b3b">⚠ 96% CRITICAL</span>
                        <span>MAX {CHEMB_CAPACITY_MCM:.0f} MCM</span>
                    </div>
                    <div style="font-size:0.55rem;color:#2a6a5a;margin-top:6px">
                        ✓ SOURCE: {res_data['source']} · {res_data['fetched_at']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Fetch failed — fall back to slider
                st.markdown(
                    '<div style="font-size:0.6rem;color:#ff8800;margin-bottom:6px">'
                    'Live reservoir fetch failed — enter manually</div>',
                    unsafe_allow_html=True
                )
                res = st.slider("Reservoir Level (MCM)", 0.0, float(MAX_CAP), 3000.0, 10.0)
                res_pct_live = (res / MAX_CAP) * 100
                bar_color = "#ff3b3b" if res_pct_live > 96 else "#ffaa00" if res_pct_live > 85 else "#00b4ff"
                st.markdown(f"""
                <div style="background:#0f1e35;border-radius:10px;padding:12px 16px;margin-top:4px;border:1px solid rgba(0,180,255,0.1)">
                    <div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#5a7a9a;margin-bottom:8px">
                        <span>RESERVOIR CAPACITY</span>
                        <span style="color:{bar_color};font-weight:700">{res_pct_live:.1f}%</span>
                    </div>
                    <div style="background:#060d1a;border-radius:6px;height:10px;overflow:hidden">
                        <div style="width:{min(res_pct_live,100)}%;height:100%;background:linear-gradient(90deg,{bar_color}88,{bar_color});border-radius:6px"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:0.6rem;color:#5a7a9a;margin-top:6px">
                        <span>0</span><span style="color:#ff3b3b">⚠ 96% CRITICAL</span><span>MAX 10568 MCM</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            # Manual mode — editable slider
            res = st.slider("Reservoir Level (MCM)", 0.0, float(MAX_CAP), 3000.0, 10.0)
            res_pct_live = (res / MAX_CAP) * 100
            bar_color = "#ff3b3b" if res_pct_live > 96 else "#ffaa00" if res_pct_live > 85 else "#00b4ff"
            st.markdown(f"""
            <div style="background:#0f1e35; border-radius:10px; padding:12px 16px; margin-top:4px; border:1px solid rgba(0,180,255,0.1)">
                <div style="display:flex; justify-content:space-between; font-size:0.7rem; color:#5a7a9a; margin-bottom:8px">
                    <span>RESERVOIR CAPACITY</span>
                    <span style="color:{bar_color}; font-weight:700">{res_pct_live:.1f}%</span>
                </div>
                <div style="background:#060d1a; border-radius:6px; height:10px; overflow:hidden">
                    <div style="width:{min(res_pct_live,100)}%; height:100%; background:linear-gradient(90deg,{bar_color}88,{bar_color}); border-radius:6px; transition:width 0.3s"></div>
                </div>
                <div style="display:flex; justify-content:space-between; font-size:0.6rem; color:#5a7a9a; margin-top:6px">
                    <span>0</span>
                    <span style="color:#ff3b3b">⚠ 96% CRITICAL</span>
                    <span>MAX 10568 MCM</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button(" ANALYZE FLOOD RISK")

    with right:
        st.markdown('<div class="section-label">LIVE INPUT SUMMARY</div>', unsafe_allow_html=True)

        input_df = pd.DataFrame(
            [[rain_today, rain_3, rain_5, rain_7, res]],
            columns=["RAINFALL","RAIN_3DAY","RAIN_5DAY","RAIN_7DAY","RES_TOTAL"]
        )

        # Live bar chart of inputs (normalized)
        labels = ["Today", "3-Day", "5-Day", "7-Day", "Reservoir"]
        maxvals = [500, 1000, 2000, 3000, MAX_CAP]
        values  = [rain_today, rain_3, rain_5, rain_7, res]
        pcts    = [v/m*100 for v,m in zip(values,maxvals)]

        fig, ax = plt.subplots(figsize=(5, 3.2))
        fig.patch.set_facecolor('#0b1628')
        ax.set_facecolor('#0b1628')

        colors = ['#ff3b3b' if p > 80 else '#ffaa00' if p > 60 else '#00b4ff' for p in pcts]
        bars = ax.barh(labels, pcts, color=colors, height=0.5)
        ax.set_xlim(0, 110)
        ax.set_xlabel('% of Max Capacity', color='#5a7a9a', fontsize=8)
        ax.tick_params(colors='#b0c8e0', labelsize=8)
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.xaxis.label.set_color('#5a7a9a')
        ax.tick_params(axis='x', colors='#5a7a9a')
        for bar, pct, val, mx in zip(bars, pcts, values, maxvals):
            ax.text(pct + 1, bar.get_y() + bar.get_height()/2,
                    f'{val:.0f}', va='center', color='#e8f4ff', fontsize=8)
        ax.axvline(x=80, color='#ffaa00', linewidth=0.8, linestyle='--', alpha=0.5)
        ax.axvline(x=96, color='#ff3b3b', linewidth=0.8, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Show result when button clicked
        if predict_btn:
            status_container = st.empty()
            with status_container.container():
                st.markdown("""
                    <div style="background:#0b1628; border:1px solid #00b4ff; border-radius:12px; padding:20px; text-align:center;">
                        <div class="spinner"></div>
                    </div>
                    <style>
                        .spinner { border: 2px solid rgba(0,180,255,0.1); border-top: 2px solid #00b4ff; border-radius: 50%; width: 30px; height: 30px; animation: spin 1s linear infinite; margin: auto; }
                        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
                    </style>
                """, unsafe_allow_html=True)
            time.sleep(1.2) 
            status_container.empty()
            vulner = 1.0
            if input_mode == "Live — Chennai":
                vulner = region_coords["vulner"]
            elif input_mode in ("Choose Location", "Auto — My location"):
                lat = st.session_state.get("map_lat")
                lon = st.session_state.get("map_lon")
                if lat is not None and lon is not None:
                    _, nearest = nearest_location(lat, lon)
                    vulner = nearest.get("vulner", 1.0)
            
            ml_score_raw = model.predict_proba(input_df)[0][1] * 100
            final_risk, res_pct = compute_risk(ml_score_raw, res, vulnerability=vulner)
            level, color, emoji = risk_level(final_risk, res_pct)
            advice = get_advice(level)
            conf = get_confidence(input_df)

            # Store in session state
            st.session_state['result'] = {
                'ml_score': ml_score_raw,
                'final_risk': final_risk,
                'res_pct': res_pct,
                'level': level,
                'color': color,
                'emoji': emoji,
                'advice': advice,
                'conf': conf,
                'inputs': input_df
            }

            # Log it
            if input_mode == "Choose Location":
                loc_label = f"{st.session_state.get('map_lat','?')}°N,{st.session_state.get('map_lon','?')}°E"
            elif input_mode == "Auto — My location":
                loc_label = f"Auto {st.session_state.get('map_lat','?')},{st.session_state.get('map_lon','?')}"
            elif input_mode == "Live — Chennai":
                loc_label = f"Chennai ({region})"
            else:
                loc_label = "Manual"

            save_log({
                "time": datetime.now().strftime("%d %b %Y %H:%M"),
                "location": loc_label,
                "rain_today": rain_today,
                "rain_3": rain_3,
                "rain_5": rain_5,
                "rain_7": rain_7,
                "res": res,
                "ml_score": round(ml_score_raw, 1),
                "final_risk": float(final_risk), # added float() for safety
                "level": level
            })

        # ── Show result if available ──
        if 'result' in st.session_state:
            r = st.session_state['result']
            c = r['color']

            st.markdown(f"""
            <div style="background:{c}11; border:2px solid {c}44; border-radius:16px; padding:28px; text-align:center; margin-top:16px; position:relative; overflow:hidden">
                <div style="font-family:'Syne',sans-serif; font-size:1.8rem; font-weight:800; color:{c}">{r['level']} RISK</div>
                <div style="font-family:'Syne',sans-serif; font-size:3.5rem; font-weight:800; color:{c}; line-height:1; margin:8px 0">{r['final_risk']}</div>
                <div style="font-size:0.6rem; letter-spacing:3px; color:#5a7a9a">COMPOSITE RISK SCORE / 100</div>
            </div>
            """, unsafe_allow_html=True)

            # 3 metric cols
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val" style="color:#00b4ff">{r['ml_score']:.1f}%</div>
                    <div class="metric-label">ML MODEL SCORE</div>
                </div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val" style="color:#ffaa00">{r['res_pct']:.1f}%</div>
                    <div class="metric-label">RESERVOIR FILL %</div>
                </div>""", unsafe_allow_html=True)
            with m3:
                conf = r['conf']
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-val" style="color:#00ffc8">{conf['lower']}–{conf['upper']}%</div>
                    <div class="metric-label">CONFIDENCE RANGE</div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""<div class="advice-box" style="border-color:{c}">
                <strong style="color:{c}">ACTION ADVISORY</strong><br>{r['advice']}
            </div>""", unsafe_allow_html=True)
            if r['conf']['uncertain']:
                st.warning(" **Note:** The AI model shows high variance. Conditions are evolving rapidly; verify with local sensors.")
           

# ══════════════════════════════════════════
#  TAB 2 — ANALYSIS & SHAP
# ══════════════════════════════════════════
with tab2:
    if 'result' not in st.session_state:
        st.markdown("""
        <div style="text-align:center; padding:60px; color:#5a7a9a">
            <div style="font-size:2rem"></div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; margin-top:12px">Run a prediction first</div>
            <div style="font-size:0.75rem; margin-top:8px">Go to the PREDICT tab, enter values and click Analyze.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        r = st.session_state['result']
        input_df = r['inputs']

        col_a, col_b = st.columns(2, gap="large")

        # ── SHAP Waterfall ──
        with col_a:
            st.markdown('<div class="section-label"> SHAP — FEATURE CONTRIBUTIONS</div>', unsafe_allow_html=True)
            st.caption(" Each bar shows how much each feature pushed the risk up (+) or down (−).")

            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                # For RF binary: shap_values[1] = class 1 (flood)
                if isinstance(shap_values, list):
                    sv = shap_values[1][0]
                else:
                    sv = shap_values[0, :, 1] if shap_values.ndim == 3 else shap_values[0]
                feat_names = ["Rainfall Today", "3-Day Rain", "5-Day Rain", "7-Day Rain", "Reservoir"]

                fig2, ax2 = plt.subplots(figsize=(5.5, 3.5))
                fig2.patch.set_facecolor('#0b1628')
                ax2.set_facecolor('#0b1628')

                colors_shap = ['#ff3b3b' if v > 0 else '#00e87a' for v in sv]
                ax2.barh(feat_names, sv, color=colors_shap, height=0.5)
                ax2.axvline(0, color='#5a7a9a', linewidth=1)
                ax2.set_xlabel('SHAP Value (impact on flood probability)', color='#5a7a9a', fontsize=8)
                ax2.tick_params(colors='#b0c8e0', labelsize=8)
                for spine in ax2.spines.values(): spine.set_visible(False)
                ax2.tick_params(axis='x', colors='#5a7a9a')
                red_patch = mpatches.Patch(color='#ff3b3b', label='Increases risk')
                green_patch = mpatches.Patch(color='#00e87a', label='Decreases risk')
                ax2.legend(handles=[red_patch, green_patch], facecolor='#0b1628',
                           labelcolor='#b0c8e0', fontsize=7, framealpha=0.5)
                plt.tight_layout()
                st.pyplot(fig2)
                plt.close()
            except Exception as e:
                st.warning(f"SHAP unavailable: {e}")

        # ── Confidence Distribution ──
        with col_b:
            st.markdown('<div class="section-label"> MODEL CONFIDENCE DISTRIBUTION</div>', unsafe_allow_html=True)
            st.caption("Distribution of predictions from all trees in the Random Forest. Wider spread = less certainty.")

            try:
                tree_preds = [t.predict_proba(input_df)[0][1] * 100 for t in model.estimators_]

                fig3, ax3 = plt.subplots(figsize=(5.5, 3.5))
                fig3.patch.set_facecolor('#0b1628')
                ax3.set_facecolor('#0b1628')

                ax3.hist(tree_preds, bins=20, color='#00b4ff', alpha=0.7, edgecolor='#060d1a')
                ax3.axvline(np.mean(tree_preds), color='#00ffc8', linewidth=2, label=f'Mean: {np.mean(tree_preds):.1f}%')
                ax3.axvline(np.percentile(tree_preds, 10), color='#ffaa00', linewidth=1.5, linestyle='--', label=f'10th pct: {np.percentile(tree_preds,10):.1f}%')
                ax3.axvline(np.percentile(tree_preds, 90), color='#ff3b3b', linewidth=1.5, linestyle='--', label=f'90th pct: {np.percentile(tree_preds,90):.1f}%')
                ax3.set_xlabel('Predicted Flood Probability (%)', color='#5a7a9a', fontsize=8)
                ax3.set_ylabel('No. of Trees', color='#5a7a9a', fontsize=8)
                ax3.tick_params(colors='#b0c8e0', labelsize=8)
                for spine in ax3.spines.values(): spine.set_visible(False)
                ax3.tick_params(axis='both', colors='#5a7a9a')
                ax3.legend(facecolor='#0b1628', labelcolor='#b0c8e0', fontsize=7, framealpha=0.5)
                plt.tight_layout()
                st.pyplot(fig3)
                plt.close()
            except Exception as e:
                st.warning(f"Confidence chart error: {e}")

        
       # ── Risk Breakdown Gauge ──
        st.markdown('<div class="section-label" style="margin-top:24px"> RISK SCORE BREAKDOWN</div>', unsafe_allow_html=True)
        g1, g2, g3 = st.columns(3)

        breakdown = [
            ("ML Model Score (60% weight)", r['ml_score'], "#00b4ff"),
            ("Reservoir Factor (40% weight)", r['res_pct'], "#ffaa00"),
            ("Composite Final Score", r['final_risk'], r['color'])
        ]

        for col, (label, val, clr) in zip([g1, g2, g3], breakdown):
            with col:
                fig_g, ax_g = plt.subplots(figsize=(3,3), subplot_kw=dict(polar=True))
                fig_g.patch.set_facecolor('#0b1628')
                ax_g.set_facecolor('#0b1628')
                
                # Semi-circle math
                val_norm = max(0, min(val, 100))
                end_angle = (val_norm / 100) * np.pi
                
                # Background track
                theta_full = np.linspace(0, np.pi, 100)
                ax_g.plot(theta_full, [1]*100, color='#0f1e35', linewidth=15)
                
                # Risk fill
                theta_fill = np.linspace(0, end_angle, 100)
                ax_g.plot(theta_fill, [1]*100, color=clr, linewidth=15)
                
                ax_g.set_ylim(0, 1.2)
                ax_g.set_theta_zero_location('W')
                ax_g.set_theta_direction(-1)
                ax_g.set_axis_off()
                ax_g.text(0, -0.2, f'{val:.1f}%', ha='center', va='center', fontsize=18, fontweight='bold', color=clr, transform=ax_g.transData)
                plt.tight_layout()
                st.pyplot(fig_g)
                plt.close()
                st.markdown(f"<div style='text-align:center;font-size:0.65rem;color:#5a7a9a;letter-spacing:1px;margin-top:-8px'>{label}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════
#  TAB 3 — HISTORY LOG
# ══════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-label">PREDICTION HISTORY</div>', unsafe_allow_html=True)

    logs = load_logs()
    if not logs:
        st.markdown("""
        <div style="text-align:center; padding:60px; color:#5a7a9a">
            <div style="font-size:2rem">🗂</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.2rem; margin-top:12px">No history yet</div>
            <div style="font-size:0.75rem; margin-top:8px">Make a prediction to start logging.</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Risk trend chart
        if len(logs) >= 2:
            st.markdown('<div class="section-label">RISK SCORE TREND</div>', unsafe_allow_html=True)
            fig_t, ax_t = plt.subplots(figsize=(10, 2.5))
            fig_t.patch.set_facecolor('#0b1628')
            ax_t.set_facecolor('#0b1628')
            times = [l.get('time', '00:00').split(' ')[1] if ' ' in l.get('time', '') else l.get('time', '00:00') for l in reversed(logs[-15:])]
            risks = [l.get('final_risk', 0) for l in reversed(logs[-15:])]
            clrs  = ['#ff3b3b' if r>=70 else '#ffaa00' if r>=40 else '#00e87a' for r in risks]
            ax_t.plot(times, risks, color='#00b4ff', linewidth=1.5, alpha=0.6)
            ax_t.scatter(times, risks, c=clrs, s=50, zorder=5)
            ax_t.axhline(70, color='#ff3b3b', linewidth=0.8, linestyle='--', alpha=0.4)
            ax_t.axhline(40, color='#ffaa00', linewidth=0.8, linestyle='--', alpha=0.4)
            ax_t.set_ylabel('Risk Score', color='#5a7a9a', fontsize=8)
            ax_t.tick_params(colors='#b0c8e0', labelsize=7)
            for spine in ax_t.spines.values(): spine.set_visible(False)
            ax_t.tick_params(axis='both', colors='#5a7a9a')
            plt.xticks(rotation=30)
            plt.tight_layout()
            st.pyplot(fig_t)
            plt.close()

        # Table
        level_badge = {"HIGH": " ", "MEDIUM-HIGH": " ", "MEDIUM": " ", "LOW": " "}
        rows = ""
        for l in logs:
            badge = level_badge.get(l.get('level',''), "—")
            rows += f"""<tr>
                <td>{l.get('time','—')}</td>
                <td>{l.get('location','—')}</td>
                <td>{l.get('rain_today','—')}</td>
                <td>{l.get('rain_7','—')}</td>
                <td>{l.get('res','—')}</td>
                <td>{l.get('ml_score','—')}%</td>
                <td><strong>{badge} {l.get('level','—')}</strong></td>
                <td>{l.get('final_risk','—')}</td>
            </tr>"""

        st.markdown(f"""
        <table class="log-table">
            <thead><tr>
                <th>TIME</th><th>LOCATION</th><th>TODAY RAIN</th><th>7-DAY RAIN</th>
                <th>RESERVOIR</th><th>ML SCORE</th><th>LEVEL</th><th>FINAL SCORE</th>
            </tr></thead>
            <tbody>{rows}</tbody>
        </table>
        """, unsafe_allow_html=True)
        if logs:
            st.markdown("<br>", unsafe_allow_html=True)
            df_logs = pd.DataFrame(logs)
            csv = df_logs.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="DOWNLOAD INCIDENT REPORT IN CSV",
                data=csv,
                file_name=f"FloodSense_Report_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        if st.button(" Clear History"):
            if os.path.exists(LOG_FILE):
                os.remove(LOG_FILE)
            st.rerun()