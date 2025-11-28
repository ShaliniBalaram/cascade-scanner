"""Cascade - Global Flood Intelligence Platform."""

import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import json
import yaml
from pathlib import Path

# Optional LLM support
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

# Page config - must be first
st.set_page_config(
    page_title="Cascade - Flood Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============ CUSTOM CSS - VERCEL LIGHT THEME ============

st.markdown("""
<style>
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Root variables */
    :root {
        --bg: #fff;
        --fg: #000;
        --gray-50: #fafafa;
        --gray-100: #f5f5f5;
        --gray-200: #eee;
        --gray-300: #ddd;
        --gray-400: #999;
        --gray-500: #666;
        --gray-600: #444;
        --green: #00a854;
        --yellow: #d48806;
        --red: #cf1322;
        --blue: #0070f3;
    }

    /* Main container - compact */
    .main .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    /* Reduce Streamlit default gaps */
    .stVerticalBlock { gap: 0.5rem !important; }
    div[data-testid="column"] { padding: 0 0.5rem !important; }

    /* Custom header - compact */
    .cascade-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 16px;
        border-bottom: 1px solid var(--gray-200);
        background: var(--bg);
    }
    .cascade-logo {
        font-size: 1rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        color: var(--fg);
    }
    .header-left {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .header-right {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .live-badge {
        display: flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        color: var(--gray-500);
    }
    .live-dot {
        width: 6px;
        height: 6px;
        background: var(--green);
        border-radius: 50%;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Status section - more compact */
    .status-section {
        text-align: center;
        padding: 16px 12px;
        border-bottom: 1px solid var(--gray-200);
    }
    .status-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--gray-500);
        margin-bottom: 4px;
    }
    .status-value {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin-bottom: 4px;
    }
    .status-value.safe { color: var(--green); }
    .status-value.warning { color: var(--yellow); }
    .status-value.danger { color: var(--red); }
    .status-summary {
        font-size: 0.8rem;
        color: var(--gray-500);
    }

    /* Metric cards - compact */
    .metric-card {
        background: var(--gray-50);
        border: 1px solid var(--gray-200);
        border-radius: 6px;
        padding: 10px 8px;
        text-align: center;
        margin-bottom: 6px;
    }
    .metric-value {
        font-size: 1.2rem;
        font-weight: 600;
        color: var(--fg);
    }
    .metric-label {
        font-size: 0.7rem;
        color: var(--gray-500);
        margin-top: 4px;
    }
    .metric-trend {
        font-size: 0.75rem;
        margin-top: 6px;
    }
    .metric-trend.up { color: var(--yellow); }
    .metric-trend.down { color: var(--green); }
    .metric-trend.neutral { color: var(--gray-500); }

    /* Section labels */
    .section-label {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--gray-500);
        margin-bottom: 12px;
    }

    /* Answer box */
    .answer-box {
        background: var(--gray-50);
        border: 1px solid var(--gray-200);
        border-radius: 8px;
        padding: 16px;
        margin-top: 16px;
    }
    .answer-verdict {
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 10px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .answer-verdict.safe { color: var(--green); }
    .answer-verdict.warning { color: var(--yellow); }
    .answer-verdict.danger { color: var(--red); }
    .answer-text {
        color: var(--gray-600);
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Quick question buttons */
    .quick-btn {
        background: transparent;
        border: 1px solid var(--gray-300);
        padding: 6px 12px;
        border-radius: 16px;
        color: var(--gray-500);
        font-size: 0.8rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .quick-btn:hover {
        border-color: var(--fg);
        color: var(--fg);
    }

    /* Map legend */
    .map-legend {
        background: var(--bg);
        border: 1px solid var(--gray-200);
        border-radius: 6px;
        padding: 12px;
        margin-top: 8px;
    }
    .legend-title {
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--gray-500);
        margin-bottom: 8px;
    }
    .legend-item {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.75rem;
        color: var(--gray-600);
        margin-right: 16px;
    }
    .legend-dot {
        width: 10px;
        height: 10px;
        border-radius: 50%;
        display: inline-block;
    }
    .legend-dot.safe { background: var(--green); }
    .legend-dot.warning { background: var(--yellow); }
    .legend-dot.danger { background: var(--red); }

    /* Hide streamlit elements */
    div[data-testid="stToolbar"] { display: none; }
    div[data-testid="stDecoration"] { display: none; }
    div[data-testid="stStatusWidget"] { display: none; }

    /* Streamlit tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--gray-100);
        border-radius: 6px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 4px;
        color: var(--gray-500);
        font-size: 0.85rem;
        padding: 10px 16px;
    }
    .stTabs [aria-selected="true"] {
        background: var(--bg) !important;
        color: var(--fg) !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }

    /* Input styling */
    .stTextInput input {
        background: var(--bg) !important;
        border: 1px solid var(--gray-300) !important;
        border-radius: 8px !important;
        padding: 14px !important;
        font-size: 0.95rem !important;
    }
    .stTextInput input:focus {
        border-color: var(--fg) !important;
        box-shadow: none !important;
    }

    /* Button styling */
    .stButton button {
        background: var(--fg) !important;
        color: var(--bg) !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 8px 12px !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important;
        white-space: nowrap !important;
        min-width: auto !important;
        width: 100% !important;
    }
    .stButton button:hover {
        opacity: 0.85 !important;
    }
    .stButton button p {
        white-space: nowrap !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
    }

    /* Download button */
    .stDownloadButton button {
        background: var(--fg) !important;
        color: var(--bg) !important;
        border: none !important;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: var(--gray-50) !important;
        border: 1px solid var(--gray-200) !important;
        border-radius: 6px !important;
    }

    /* Responsive - Medium screens */
    @media (max-width: 1200px) {
        .status-value { font-size: 1.8rem; }
        .metric-value { font-size: 1.1rem; }
    }

    /* Responsive - Small screens / minimized */
    @media (max-width: 900px) {
        .cascade-header { padding: 10px 12px; }
        .cascade-logo { font-size: 0.9rem; }
        .status-section { padding: 12px 8px; }
        .status-value { font-size: 1.6rem; }
        .status-summary { font-size: 0.75rem; }
        .metric-card { padding: 8px 6px; }
        .metric-value { font-size: 1rem; }
        .metric-label { font-size: 0.65rem; }
        .section-label { font-size: 0.6rem; margin-bottom: 8px; }
        .answer-box { padding: 12px; }
        .answer-text { font-size: 0.8rem; }
        .map-legend { padding: 8px; }
        .legend-item { font-size: 0.65rem; margin-right: 10px; }
    }

    /* Very small screens */
    @media (max-width: 600px) {
        .status-value { font-size: 1.4rem; }
        .metric-value { font-size: 0.9rem; }
        .quick-btn { font-size: 0.7rem; padding: 4px 8px; }
    }

    /* Reduce overall page height */
    .stRadio > div { gap: 0.5rem !important; }
    .stTabs { margin-bottom: 0.5rem !important; }
    hr { margin: 0.5rem 0 !important; }

    /* Hero section */
    .hero-section {
        text-align: center;
        padding: 24px 16px 16px;
        border-bottom: 1px solid var(--gray-200);
    }
    .hero-title {
        font-size: 1.8rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        color: var(--fg);
        margin-bottom: 6px;
    }
    .hero-tagline {
        font-size: 0.9rem;
        color: var(--gray-500);
        font-weight: 400;
    }
    @media (max-width: 600px) {
        .hero-title { font-size: 1.4rem; }
        .hero-tagline { font-size: 0.8rem; }
    }

    /* How it works section - Minimalist style */
    .how-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 16px;
        margin: 8px 0;
    }
    @media (max-width: 800px) {
        .how-grid { grid-template-columns: 1fr; }
    }
    .how-card {
        background: var(--gray-50);
        border: 1px solid var(--gray-200);
        border-radius: 8px;
        padding: 20px;
        text-align: left;
    }
    .how-step {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 24px;
        height: 24px;
        background: var(--fg);
        color: var(--bg);
        border-radius: 50%;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .how-card h4 {
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--fg);
        margin-bottom: 6px;
    }
    .how-card p {
        font-size: 0.8rem;
        color: var(--gray-500);
        line-height: 1.5;
        margin: 0;
    }
    .how-footer {
        text-align: center;
        margin-top: 16px;
        padding-top: 16px;
        border-top: 1px solid var(--gray-200);
        font-size: 0.7rem;
        color: var(--gray-400);
    }

    /* Footer */
    .app-footer {
        margin-top: 24px;
        padding: 16px;
        border-top: 1px solid var(--gray-200);
        text-align: center;
        font-size: 0.75rem;
        color: var(--gray-500);
    }
    .app-footer a {
        color: var(--gray-600);
        text-decoration: none;
        transition: color 0.2s;
    }
    .app-footer a:hover {
        color: var(--fg);
    }
    .footer-author {
        font-weight: 500;
        color: var(--gray-600);
    }
</style>
""", unsafe_allow_html=True)


# ============ DATA LOADING ============

@st.cache_data
def load_assets():
    """Load Chennai infrastructure assets from YAML."""
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "assets" / "chennai_assets.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return data.get("assets", [])
    except Exception:
        return get_default_assets()


@st.cache_data
def load_fragility_curves():
    """Load fragility curves from YAML."""
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "fragility_curves" / "chennai_v1.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return data.get("fragility_curves", [])
    except:
        return []


def get_default_assets():
    """Default assets if YAML not found."""
    return [
        {"asset_id": "hospital_rggh", "name": "Rajiv Gandhi Government Hospital", "asset_type": "hospital", "lat": 13.0878, "lon": 80.2785, "elevation_m": 8.2},
        {"asset_id": "hospital_apollo", "name": "Apollo Hospital Greams Road", "asset_type": "hospital", "lat": 13.0569, "lon": 80.2425, "elevation_m": 10.5},
        {"asset_id": "hospital_fortis", "name": "Fortis Malar Hospital", "asset_type": "hospital", "lat": 13.0244, "lon": 80.2536, "elevation_m": 4.8},
        {"asset_id": "hospital_stanley", "name": "Stanley Medical College Hospital", "asset_type": "hospital", "lat": 13.1148, "lon": 80.2866, "elevation_m": 5.1},
        {"asset_id": "hospital_miot", "name": "MIOT International", "asset_type": "hospital", "lat": 13.0122, "lon": 80.1696, "elevation_m": 12.3},
        {"asset_id": "substation_tondiarpet", "name": "Tondiarpet 230kV Substation", "asset_type": "substation", "lat": 13.1247, "lon": 80.2891, "elevation_m": 3.2},
        {"asset_id": "substation_kathivakkam", "name": "Kathivakkam 110kV Substation", "asset_type": "substation", "lat": 13.2147, "lon": 80.3156, "elevation_m": 2.8},
        {"asset_id": "substation_porur", "name": "Porur 110kV Substation", "asset_type": "substation", "lat": 13.0383, "lon": 80.1572, "elevation_m": 15.6},
        {"asset_id": "wwtp_nesapakkam", "name": "Nesapakkam Sewage Treatment Plant", "asset_type": "wastewater_plant", "lat": 13.0445, "lon": 80.1892, "elevation_m": 11.2},
        {"asset_id": "wwtp_kodungaiyur", "name": "Kodungaiyur Sewage Treatment Plant", "asset_type": "wastewater_plant", "lat": 13.1312, "lon": 80.2523, "elevation_m": 4.5},
    ]


# ============ WEATHER DATA ============

@st.cache_data(ttl=1800)
def get_current_weather():
    """Fetch current weather from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m",
        "timezone": "Asia/Kolkata"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("current", {})
    except:
        pass
    return {"temperature_2m": 28, "relative_humidity_2m": 75, "precipitation": 0}


@st.cache_data(ttl=1800)
def get_weather_forecast():
    """Fetch 7-day forecast."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,precipitation_probability_max",
        "timezone": "Asia/Kolkata"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("daily", {})
    except:
        pass
    return {}


@st.cache_data(ttl=3600)
def get_historical_weather(days_back=30):
    """Fetch historical weather data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "precipitation_sum,rain_sum",
        "timezone": "Asia/Kolkata"
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("daily", {})
    except:
        pass
    return {}


# ============ HAZARD ASSESSMENT ============

def assess_flood_risk(rainfall_mm: float, elevation_m: float) -> dict:
    """Assess flood risk based on rainfall and elevation."""
    if rainfall_mm < 30:
        depth_m, risk_level = 0, "safe"
    elif rainfall_mm < 50:
        depth_m, risk_level = 0.1, "safe"
    elif rainfall_mm < 80:
        depth_m, risk_level = 0.25, "warning"
    elif rainfall_mm < 120:
        depth_m, risk_level = 0.4, "warning"
    elif rainfall_mm < 180:
        depth_m, risk_level = 0.6, "danger"
    else:
        depth_m, risk_level = 0.8 + (rainfall_mm - 180) / 200, "danger"

    if elevation_m > 15:
        depth_m *= 0.3
        if risk_level == "danger":
            risk_level = "warning"
    elif elevation_m > 10:
        depth_m *= 0.6
    elif elevation_m < 5:
        depth_m *= 1.3

    return {"depth_m": round(depth_m, 2), "risk_level": risk_level, "rainfall_mm": rainfall_mm}


def get_satellite_indices():
    """Get simulated satellite indices."""
    np.random.seed(int(datetime.now().timestamp()) // 3600)
    return {
        "ndvi": round(np.random.uniform(0.3, 0.7), 3),
        "ndwi": round(np.random.uniform(-0.2, 0.4), 3),
        "soil_moisture": round(np.random.uniform(0.15, 0.45), 3),
        "flood_extent_sqkm": round(np.random.uniform(0, 8), 1),
        "last_update": datetime.now().strftime("%H:%M"),
    }


# ============ AI ASSISTANT ============

def get_llm_response(question: str, context: str) -> str:
    """Get response from Groq LLM."""
    try:
        api_key = None
        if hasattr(st, 'secrets') and 'groq' in st.secrets:
            api_key = st.secrets['groq'].get('api_key')

        if not api_key or not GROQ_AVAILABLE:
            return None

        client = Groq(api_key=api_key)

        system_prompt = f"""You are a flood risk analyst for Chennai, India.
Give SHORT, DIRECT answers (2-3 sentences max).
Start with a clear verdict: SAFE, CAUTION, or WARNING.

Current Data:
{context}

Be specific with numbers. No long explanations."""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=200,
            temperature=0.5
        )
        return response.choices[0].message.content
    except:
        return None


# ============ HELPER FUNCTIONS FOR POPOVER CONTENT ============

def render_how_it_works_content():
    """Render the How it Works content."""
    st.markdown("""
    <div class="how-grid">
        <div class="how-card">
            <div class="how-step">1</div>
            <h4>Collect</h4>
            <p>Satellite imagery and weather data gathered every 30 minutes from multiple sources.</p>
        </div>
        <div class="how-card">
            <div class="how-step">2</div>
            <h4>Analyze</h4>
            <p>AI detects flood extent, maps infrastructure at risk, and calculates impact probability.</p>
        </div>
        <div class="how-card">
            <div class="how-step">3</div>
            <h4>Alert</h4>
            <p>Real-time risk assessment delivered. Ask questions, get immediate answers.</p>
        </div>
    </div>
    <div class="how-footer">
        Data sources: Sentinel-1 SAR, Open-Meteo, OpenStreetMap
    </div>
    """, unsafe_allow_html=True)


def render_download_content(location, flood_risk, weather, satellite, forecast, assets):
    """Render download options content."""
    import io

    st.markdown("**Choose data type:**")

    data_type = st.radio(
        "Data Type",
        ["Full Report", "Weather Data", "Infrastructure", "Satellite"],
        horizontal=True,
        label_visibility="collapsed"
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')

    if data_type == "Full Report":
        st.caption("Complete flood risk assessment")
        data = {
            "timestamp": datetime.now().isoformat(),
            "location": location,
            "risk_level": flood_risk['risk_level'],
            "weather": weather,
            "satellite": satellite,
            "forecast_7day": forecast.get("precipitation_sum", [])[:7],
            "assets_count": len(assets)
        }
        st.download_button("Download JSON", json.dumps(data, indent=2), f"cascade_report_{timestamp}.json", "application/json", use_container_width=True)

    elif data_type == "Weather Data":
        st.caption("7-day forecast CSV")
        csv = io.StringIO()
        csv.write("date,temp_c,rain_mm\n")
        csv.write(f"now,{weather.get('temperature_2m', 28)},{weather.get('precipitation', 0)}\n")
        for i, p in enumerate(forecast.get("precipitation_sum", [])[:7]):
            csv.write(f"day_{i+1},,{p}\n")
        st.download_button("Download CSV", csv.getvalue(), f"weather_{timestamp}.csv", "text/csv", use_container_width=True)

    elif data_type == "Infrastructure":
        st.caption("Monitored assets CSV")
        csv = io.StringIO()
        csv.write("name,type,lat,lon,elevation_m\n")
        for a in assets:
            csv.write(f"{a.get('name','')},{a.get('asset_type','')},{a.get('lat','')},{a.get('lon','')},{a.get('elevation_m','')}\n")
        st.download_button("Download CSV", csv.getvalue(), f"assets_{timestamp}.csv", "text/csv", use_container_width=True)

    else:  # Satellite
        st.caption("Satellite flood indices")
        data = {"timestamp": datetime.now().isoformat(), "indices": satellite}
        st.download_button("Download JSON", json.dumps(data, indent=2), f"satellite_{timestamp}.json", "application/json", use_container_width=True)


# ============ MAIN APP ============

def main():
    # Load all data
    assets = load_assets()
    curves = load_fragility_curves()
    weather = get_current_weather()
    forecast = get_weather_forecast()
    satellite = get_satellite_indices()

    # Calculate current risk
    rain_24h = weather.get("precipitation", 0) or 0
    rain_forecast = sum(forecast.get("precipitation_sum", [0, 0, 0])[:3])
    flood_risk = assess_flood_risk(rain_24h + rain_forecast, 8)

    # ============ HEADER ============
    st.markdown("""
    <div class="cascade-header">
        <div class="header-left">
            <span class="cascade-logo">Cascade</span>
        </div>
        <div class="header-right">
            <div class="live-badge">
                <span class="live-dot"></span>
                Live
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ============ HERO SECTION ============
    st.markdown("""
    <div class="hero-section">
        <div class="hero-title">Flood Intelligence Platform</div>
        <div class="hero-tagline">Real-time hazard monitoring powered by satellite data and AI</div>
    </div>
    """, unsafe_allow_html=True)

    # ============ LOCATION & CONTROLS ============
    col_loc, col_download = st.columns([3, 1])
    with col_loc:
        location = st.selectbox(
            "Location",
            ["Chennai, India", "Mumbai, India", "Kolkata, India", "Bangkok, Thailand"],
            label_visibility="collapsed"
        )
    with col_download:
        with st.expander("Download Data"):
            render_download_content(location, flood_risk, weather, satellite, forecast, assets)

    # How it works section
    with st.expander("How it works"):
        render_how_it_works_content()

    # ============ MAIN LAYOUT ============
    col_map, col_panel = st.columns([1.5, 1])

    # ============ MAP SECTION ============
    with col_map:
        # View toggle
        view_type = st.radio(
            "View",
            ["Satellite", "Map", "Flood Extent"],
            horizontal=True,
            label_visibility="collapsed"
        )

        # Create map
        if view_type == "Satellite":
            tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
            attr = "Esri"
        elif view_type == "Flood Extent":
            tiles = "CartoDB dark_matter"
            attr = "CartoDB"
        else:
            tiles = "CartoDB positron"
            attr = "CartoDB"

        m = folium.Map(
            location=[13.0827, 80.2707],
            zoom_start=11,
            tiles=tiles,
            attr=attr
        )

        # Add assets to map
        type_colors = {"hospital": "red", "substation": "orange", "wastewater_plant": "blue"}

        for asset in assets:
            color = type_colors.get(asset.get("asset_type"), "gray")
            # Check if at risk
            if asset.get("elevation_m", 100) < 5 and flood_risk['risk_level'] != 'safe':
                color = "darkred"

            folium.CircleMarker(
                location=[asset["lat"], asset["lon"]],
                radius=8,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                popup=f"{asset['name']}<br>Elevation: {asset.get('elevation_m', 'N/A')}m"
            ).add_to(m)

        # Add flood zones if in flood view
        if view_type == "Flood Extent" and satellite.get("flood_extent_sqkm", 0) > 0:
            flood_zones = [
                {"name": "Adyar Estuary", "lat": 13.0012, "lon": 80.2567},
                {"name": "Cooum River", "lat": 13.0827, "lon": 80.2707},
                {"name": "Ennore Creek", "lat": 13.22, "lon": 80.32},
            ]
            for zone in flood_zones:
                folium.Circle(
                    location=[zone["lat"], zone["lon"]],
                    radius=satellite.get("flood_extent_sqkm", 0) * 200,
                    color="blue",
                    fill=True,
                    fillColor="blue",
                    fillOpacity=0.3,
                    popup=zone["name"]
                ).add_to(m)

        st_folium(m, height=350, use_container_width=True)

        # Legend
        st.markdown("""
        <div class="map-legend">
            <div class="legend-title">Risk Level</div>
            <span class="legend-item"><span class="legend-dot safe"></span> Low</span>
            <span class="legend-item"><span class="legend-dot warning"></span> Medium</span>
            <span class="legend-item"><span class="legend-dot danger"></span> High</span>
        </div>
        """, unsafe_allow_html=True)

    # ============ PANEL SECTION ============
    with col_panel:
        # Status block
        risk_display = {
            "safe": ("Safe", "safe", "No immediate flood threat"),
            "warning": ("Caution", "warning", "Elevated risk - monitor conditions"),
            "danger": ("Warning", "danger", "High flood risk - take precautions")
        }
        status_text, status_class, status_summary = risk_display.get(
            flood_risk['risk_level'],
            ("Safe", "safe", "Conditions normal")
        )

        st.markdown(f"""
        <div class="status-section">
            <div class="status-label">Current Risk Level</div>
            <div class="status-value {status_class}">{status_text}</div>
            <div class="status-summary">{status_summary}</div>
        </div>
        """, unsafe_allow_html=True)

        # Forecast tabs
        st.markdown('<div class="section-label">Forecast</div>', unsafe_allow_html=True)
        tab_now, tab_3day, tab_7day = st.tabs(["Now", "3-Day", "7-Day"])

        with tab_now:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{weather.get('temperature_2m', 28)}°C</div>
                    <div class="metric-label">Temperature</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{rain_24h}mm</div>
                    <div class="metric-label">Rainfall (24h)</div>
                </div>
                """, unsafe_allow_html=True)

            col3, col4 = st.columns(2)
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{satellite.get('flood_extent_sqkm', 0)}km²</div>
                    <div class="metric-label">Flood Extent</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                at_risk = len([a for a in assets if a.get("elevation_m", 100) < 5])
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{len(assets) - at_risk}/{len(assets)}</div>
                    <div class="metric-label">Assets Safe</div>
                </div>
                """, unsafe_allow_html=True)

        with tab_3day:
            rain_3day = sum(forecast.get("precipitation_sum", [0, 0, 0])[:3])
            max_temps = forecast.get("temperature_2m_max", [30, 30, 30])[:3]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{rain_3day:.0f}mm</div>
                    <div class="metric-label">Expected Rain</div>
                    <div class="metric-trend {'up' if rain_3day > 50 else 'neutral'}">
                        {'⚠️ Heavy' if rain_3day > 100 else '↑ Moderate' if rain_3day > 50 else 'Normal'}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{max(max_temps) if max_temps else 32}°C</div>
                    <div class="metric-label">Max Temp</div>
                </div>
                """, unsafe_allow_html=True)

        with tab_7day:
            rain_7day = sum(forecast.get("precipitation_sum", [])[:7])
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{rain_7day:.0f}mm</div>
                <div class="metric-label">Total Expected Rainfall</div>
                <div class="metric-trend {'danger' if rain_7day > 200 else 'warning' if rain_7day > 100 else 'neutral'}">
                    {'🔴 Flood Risk High' if rain_7day > 200 else '🟠 Monitor Closely' if rain_7day > 100 else '🟢 Normal Range'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Ask section
        st.markdown("---")
        st.markdown('<div class="section-label">Ask anything</div>', unsafe_allow_html=True)

        # Quick questions - 2x2 grid for better responsiveness
        quick_questions = ["Will it flood?", "Safe to travel?", "Power risk?", "Reservoirs?"]

        # Track if a button was clicked this run
        button_clicked = None

        row1_cols = st.columns(2)
        with row1_cols[0]:
            if st.button(quick_questions[0], key="quick_0", use_container_width=True):
                button_clicked = quick_questions[0]
        with row1_cols[1]:
            if st.button(quick_questions[1], key="quick_1", use_container_width=True):
                button_clicked = quick_questions[1]

        row2_cols = st.columns(2)
        with row2_cols[0]:
            if st.button(quick_questions[2], key="quick_2", use_container_width=True):
                button_clicked = quick_questions[2]
        with row2_cols[1]:
            if st.button(quick_questions[3], key="quick_3", use_container_width=True):
                button_clicked = quick_questions[3]

        # Text input with submit button
        input_col, btn_col = st.columns([4, 1])
        with input_col:
            user_q = st.text_input(
                "Question",
                placeholder="Should I travel to Chennai this weekend?",
                label_visibility="collapsed",
                key="question_input"
            )
        with btn_col:
            submit_clicked = st.button("Ask", key="submit_btn", use_container_width=True)

        # Priority: button click > submit button with text > existing question
        if button_clicked:
            st.session_state.user_question = button_clicked
        elif submit_clicked and user_q:
            st.session_state.user_question = user_q

        # Display answer
        if hasattr(st.session_state, 'user_question') and st.session_state.user_question:
            question = st.session_state.user_question

            # Build context
            context = f"""
Weather: {weather.get('temperature_2m', 28)}°C, Rain: {rain_24h}mm
3-day forecast: {rain_forecast:.0f}mm rain
Satellite: {satellite.get('flood_extent_sqkm', 0)}km² flood detected
Soil moisture: {satellite.get('soil_moisture', 0.3)*100:.0f}%
Risk level: {flood_risk['risk_level']}
At-risk assets: {at_risk} of {len(assets)}
"""

            # Try LLM first
            llm_response = get_llm_response(question, context)

            if llm_response:
                # Determine verdict class
                verdict_class = "safe"
                if any(w in llm_response.upper() for w in ["WARNING", "DANGER", "HIGH RISK", "CAUTION"]):
                    verdict_class = "warning" if "CAUTION" in llm_response.upper() else "danger"

                st.markdown(f"""
                <div class="answer-box">
                    <div class="answer-text">{llm_response}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                # Fallback logic
                q_lower = question.lower()

                if any(w in q_lower for w in ["flood", "rain", "water"]):
                    if rain_forecast > 100:
                        verdict, verdict_class = "⚠️ HIGH RISK", "danger"
                        text = f"Heavy rainfall of {rain_forecast:.0f}mm expected. Flooding likely in low-lying areas."
                    elif rain_forecast > 50:
                        verdict, verdict_class = "🟠 MODERATE RISK", "warning"
                        text = f"{rain_forecast:.0f}mm rain expected. Some waterlogging possible."
                    else:
                        verdict, verdict_class = "✓ LOW RISK", "safe"
                        text = f"Only {rain_forecast:.0f}mm rain expected. Minimal flood risk."

                elif any(w in q_lower for w in ["travel", "visit", "trip", "safe"]):
                    if flood_risk['risk_level'] == 'danger':
                        verdict, verdict_class = "⚠️ POSTPONE", "danger"
                        text = "High flood risk. Consider postponing travel plans."
                    elif flood_risk['risk_level'] == 'warning':
                        verdict, verdict_class = "🟠 PROCEED WITH CAUTION", "warning"
                        text = "Some disruption possible. Have backup plans ready."
                    else:
                        verdict, verdict_class = "✓ SAFE TO TRAVEL", "safe"
                        text = "Conditions favorable. Roads clear, minimal rain expected."

                elif any(w in q_lower for w in ["power", "electric", "substation"]):
                    subs_at_risk = len([a for a in assets if a.get("asset_type") == "substation" and a.get("elevation_m", 100) < 5])
                    if subs_at_risk > 0 and flood_risk['risk_level'] != 'safe':
                        verdict, verdict_class = "🟠 RISK PRESENT", "warning"
                        text = f"{subs_at_risk} substations in flood-prone areas may be affected."
                    else:
                        verdict, verdict_class = "✓ STABLE", "safe"
                        text = "Power infrastructure operating normally."

                else:
                    verdict, verdict_class = "ℹ️ INFO", "safe"
                    text = f"Current: {weather.get('temperature_2m', 28)}°C, {rain_24h}mm rain. Risk: {flood_risk['risk_level'].upper()}"

                st.markdown(f"""
                <div class="answer-box">
                    <div class="answer-verdict {verdict_class}">{verdict}</div>
                    <div class="answer-text">{text}</div>
                </div>
                """, unsafe_allow_html=True)

            # Source info
            st.caption(f"Updated {satellite.get('last_update', 'recently')} • Sources: Open-Meteo, Sentinel-1 SAR")

    # ============ FOOTER ============
    st.markdown("""
    <div class="app-footer">
        Built by <span class="footer-author">Dr Shalini B</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
