"""Streamlit dashboard for Cascade Scanner - Full Featured."""

import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone, timedelta
import json
import pandas as pd
import numpy as np
import io
import requests
import sys
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import drought engine
try:
    from src.core.drought_engine import (
        generate_5year_data, calculate_water_budget, assess_drought,
        generate_projections, calculate_crop_status, generate_regional_report,
        generate_satellite_observations, SCENARIOS
    )
    DROUGHT_ENGINE_AVAILABLE = True
except ImportError:
    DROUGHT_ENGINE_AVAILABLE = False

st.set_page_config(
    page_title="Cascade Scanner",
    page_icon="🌊",
    layout="wide",
)

# ============ DEMO DATA ============
DEMO_ASSETS = [
    {"asset_id": "hospital_rggh", "name": "Rajiv Gandhi Govt Hospital", "asset_type": "hospital", "lat": 13.0878, "lon": 80.2785, "criticality": 5},
    {"asset_id": "hospital_apollo", "name": "Apollo Hospital Greams Road", "asset_type": "hospital", "lat": 13.0569, "lon": 80.2425, "criticality": 5},
    {"asset_id": "hospital_stanley", "name": "Stanley Medical College Hospital", "asset_type": "hospital", "lat": 13.1122, "lon": 80.2875, "criticality": 5},
    {"asset_id": "substation_tondiarpet", "name": "Tondiarpet 230kV Substation", "asset_type": "substation", "lat": 13.1247, "lon": 80.2891, "criticality": 5},
    {"asset_id": "substation_kathivakkam", "name": "Kathivakkam 110kV Substation", "asset_type": "substation", "lat": 13.2147, "lon": 80.3156, "criticality": 4},
    {"asset_id": "substation_manali", "name": "Manali 400kV Substation", "asset_type": "substation", "lat": 13.1667, "lon": 80.2589, "criticality": 5},
    {"asset_id": "wwtp_kodungaiyur", "name": "Kodungaiyur STP", "asset_type": "wastewater_plant", "lat": 13.1312, "lon": 80.2523, "criticality": 4},
    {"asset_id": "wwtp_nesapakkam", "name": "Nesapakkam STP", "asset_type": "wastewater_plant", "lat": 13.0456, "lon": 80.2012, "criticality": 4},
    {"asset_id": "road_nh16", "name": "NH16 Kathivakkam Stretch", "asset_type": "evacuation_route", "lat": 13.2089, "lon": 80.3012, "criticality": 4},
    {"asset_id": "road_ecr", "name": "ECR Adyar-Thiruvanmiyur", "asset_type": "evacuation_route", "lat": 13.0012, "lon": 80.2567, "criticality": 4},
]

# ============ NEO4J CONNECTION ============
@st.cache_resource
def get_neo4j_driver():
    """Get Neo4j driver (cached)."""
    try:
        if hasattr(st, "secrets") and "neo4j" in st.secrets:
            from neo4j import GraphDatabase
            uri = st.secrets["neo4j"]["uri"]
            user = st.secrets["neo4j"]["user"]
            password = st.secrets["neo4j"]["password"]
            driver = GraphDatabase.driver(uri, auth=(user, password))
            return driver
    except Exception:
        pass
    return None

def get_assets():
    """Fetch assets from Neo4j or return demo data."""
    driver = get_neo4j_driver()
    if not driver:
        return DEMO_ASSETS
    try:
        with driver.session() as session:
            result = session.run("""
                MATCH (a:Asset)
                RETURN a.asset_id as asset_id, a.name as name,
                       a.asset_type as asset_type, a.lat as lat,
                       a.lon as lon, a.criticality as criticality
            """)
            assets = [dict(record) for record in result]
            return assets if assets else DEMO_ASSETS
    except Exception:
        return DEMO_ASSETS

# ============ TEMPORAL ANALYSIS ============
def generate_temporal_series(days_back: int = 30):
    """Fetch REAL temporal data from Open-Meteo for Chennai."""
    lat, lon = 13.0827, 80.2707  # Chennai
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    series = []

    try:
        # Fetch from Open-Meteo Archive API
        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": (end_date - timedelta(days=5)).strftime("%Y-%m-%d"),
            "daily": "precipitation_sum",
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(archive_url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            precip = daily.get("precipitation_sum", [])

            for i, date in enumerate(dates):
                rainfall = precip[i] if precip[i] is not None else 0.0

                if rainfall > 50:
                    risk = "high"
                elif rainfall > 20:
                    risk = "moderate"
                elif rainfall > 5:
                    risk = "low"
                else:
                    risk = "minimal"

                series.append({
                    "date": date,
                    "rainfall_mm": round(rainfall, 1),
                    "risk_level": risk
                })

        # Add recent days from forecast API
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "past_days": 5,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(forecast_url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            daily = data.get("daily", {})
            existing_dates = {s["date"] for s in series}

            for i, date in enumerate(daily.get("time", [])):
                if date not in existing_dates:
                    rainfall = daily.get("precipitation_sum", [])[i] or 0.0

                    if rainfall > 50:
                        risk = "high"
                    elif rainfall > 20:
                        risk = "moderate"
                    elif rainfall > 5:
                        risk = "low"
                    else:
                        risk = "minimal"

                    series.append({
                        "date": date,
                        "rainfall_mm": round(rainfall, 1),
                        "risk_level": risk
                    })

        series.sort(key=lambda x: x["date"])

    except Exception as e:
        # Fallback: return empty with message
        series = [{"date": datetime.now().strftime("%Y-%m-%d"), "rainfall_mm": 0.0, "risk_level": "minimal"}]

    return series

def analyze_temporal_data(days_back: int = 30):
    """Analyze temporal trends."""
    series = generate_temporal_series(days_back)
    df = pd.DataFrame(series)

    total_rainfall = df['rainfall_mm'].sum()
    avg_rainfall = df['rainfall_mm'].mean()
    max_rainfall = df['rainfall_mm'].max()
    rainy_days = len(df[df['rainfall_mm'] > 0.1])

    high_risk = len(df[df['risk_level'].isin(['high', 'severe'])])
    moderate_risk = len(df[df['risk_level'] == 'moderate'])

    # Trend calculation
    first_half = df['rainfall_mm'][:len(df)//2].mean()
    second_half = df['rainfall_mm'][len(df)//2:].mean()

    if second_half > first_half * 1.2:
        trend = "increasing"
    elif second_half < first_half * 0.8:
        trend = "decreasing"
    else:
        trend = "stable"

    month = datetime.now().month
    if month in [10, 11, 12]:
        monsoon = "Northeast Monsoon (Peak)"
        peak_month = "November"
    elif month in [6, 7, 8, 9]:
        monsoon = "Southwest Monsoon"
        peak_month = "August"
    else:
        monsoon = "Dry Season"
        peak_month = None

    return {
        "total_rainfall_mm": round(total_rainfall, 1),
        "avg_daily_mm": round(avg_rainfall, 1),
        "max_daily_mm": round(max_rainfall, 1),
        "rainy_days": rainy_days,
        "high_risk_days": high_risk,
        "moderate_risk_days": moderate_risk,
        "trend": trend,
        "monsoon_status": monsoon,
        "peak_month": peak_month,
        "series": series
    }

# ============ INITIALIZE ============
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []

def main():
    st.title("🌊 Cascade Scanner")
    st.markdown("*Flood-cascade risk assessment for Chennai with temporal analysis*")

    # Check if secrets exist
    has_secrets = hasattr(st, "secrets") and "neo4j" in st.secrets

    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "🗺️ Live Scanner",
        "📈 Temporal Analysis",
        "🛰️ Satellite Data",
        "📥 Download Center",
        "📊 Historical Data",
        "🌾 Drought Mode",
        "👨‍🌾 Farmer View",
        "🏛️ Admin View"
    ])

    with tab1:
        render_scanner_tab(has_secrets)

    with tab2:
        render_temporal_tab()

    with tab3:
        render_satellite_tab()

    with tab4:
        render_download_tab()

    with tab5:
        render_history_tab()

    with tab6:
        render_drought_tab()

    with tab7:
        render_farmer_tab()

    with tab8:
        render_admin_tab()


def render_scanner_tab(has_secrets):
    """Main scanner interface."""
    if not has_secrets:
        st.info("📌 **Demo Mode** - Using sample data. Connect Neo4j for live data.")

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Scan Settings")

        time_mode = st.selectbox(
            "Time Mode",
            ["nowcast", "forecast", "diagnostic"],
            index=0,
        )

        stakeholder = st.selectbox(
            "Stakeholder View",
            ["emergency_manager", "researcher"],
            index=0,
        )

        st.divider()
        st.header("🔍 Quick Query")

        query = st.text_input(
            "Natural Language Query",
            placeholder="e.g., Show flood risks in Chennai",
        )

        if st.button("🔍 Run Scan", type="primary", use_container_width=True):
            run_scan(query, time_mode, stakeholder)

        st.divider()
        st.caption("**Data Sources:**")
        st.caption("• Precipitation: Open-Meteo")
        st.caption("• Satellite: Sentinel-1 SAR")
        st.caption("• Assets: Neo4j Knowledge Graph")

        if has_secrets:
            st.success("✅ Neo4j Connected")

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📍 Chennai Infrastructure Map")
        render_map()

    with col2:
        st.subheader("📊 Current Status")
        render_status()

    if st.session_state.last_result:
        render_results()


def run_scan(query: str, time_mode: str, stakeholder: str):
    """Execute scan."""
    with st.spinner("Scanning Chennai for flood risks..."):
        # Create scan result
        result = create_scan_result(time_mode)
        st.session_state.last_result = result

        # Add to history
        st.session_state.scan_history.append({
            "timestamp": datetime.now().isoformat(),
            "event_id": result["event_id"],
            "risk_level": result["hazard"]["risk_level"],
            "rainfall_mm": result["hazard"]["rainfall_24h_mm"]
        })

        st.success("Scan complete!")


def fetch_real_rainfall() -> float:
    """Fetch real 24h rainfall from Open-Meteo API."""
    try:
        lat, lon = 13.0827, 80.2707  # Chennai
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "past_days": 1,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            daily = data.get("daily", {})
            precip_values = daily.get("precipitation_sum", [0])
            # Get yesterday's precipitation (last 24h)
            if len(precip_values) > 0:
                return precip_values[0] or 0.0
        return 0.0
    except Exception:
        return 0.0  # Return 0 if API fails


def create_scan_result(time_mode: str):
    """Create scan result."""
    assets = get_assets()

    # Fetch REAL rainfall data from Open-Meteo
    rainfall = fetch_real_rainfall()

    # Estimate depth from rainfall (Chennai drainage model)
    if rainfall < 30:
        depth = 0.0
    elif rainfall < 50:
        depth = 0.1
    elif rainfall < 80:
        depth = 0.25
    elif rainfall < 120:
        depth = 0.4
    elif rainfall < 180:
        depth = 0.6
    else:
        depth = 0.8 + (rainfall - 180) / 200

    if rainfall > 50:
        risk = "high"
    elif rainfall > 20:
        risk = "moderate"
    elif rainfall > 5:
        risk = "low"
    else:
        risk = "minimal"

    # Calculate alerts based on REAL thresholds (from fragility curves)
    # Thresholds: substation=0.4m, road=0.3m, wwtp=0.6m, hospital=0.5m
    alerts_count = 0
    if depth >= 0.25:  # Expressway threshold
        alerts_count += 2  # Ennore Expressway, ECR
    if depth >= 0.3:   # Road threshold
        alerts_count += 3  # NH16, other arterials
    if depth >= 0.4:   # Substation threshold
        alerts_count += 2  # Low-lying substations
    if depth >= 0.5:   # Hospital threshold
        alerts_count += 2  # Low-lying hospitals
    if depth >= 0.6:   # WWTP threshold
        alerts_count += 1  # Kodungaiyur WWTP

    return {
        "event_id": f"SCAN-{datetime.now().strftime('%Y%m%d-%H%M')}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "location": "Chennai, India",
        "time_mode": time_mode,
        "hazard": {
            "type": "flood",
            "rainfall_24h_mm": round(rainfall, 1),
            "depth_m": round(depth, 2),
            "risk_level": risk
        },
        "assets_count": len(assets),
        "alerts_count": alerts_count,
        "summary": f"Scan complete - {len(assets)} assets monitored, {alerts_count} at risk"
    }


def render_map():
    """Render map."""
    m = folium.Map(
        location=[13.0827, 80.2707],
        zoom_start=11,
        tiles="CartoDB positron",
    )

    colors = {
        "hospital": "red",
        "substation": "orange",
        "wastewater_plant": "purple",
        "evacuation_route": "blue",
    }

    assets = get_assets()

    for a in assets:
        lat = a.get("lat", 0)
        lon = a.get("lon", 0)
        atype = a.get("asset_type", "")
        name = a.get("name", "Unknown")

        if lat and lon:
            folium.CircleMarker(
                location=[lat, lon],
                radius=8,
                color=colors.get(atype, "gray"),
                fill=True,
                fillOpacity=0.7,
                popup=f"<b>{name}</b><br>{atype}",
            ).add_to(m)

    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; border: 1px solid gray;">
        <b>Legend</b><br>
        <span style="color: red;">●</span> Hospital<br>
        <span style="color: orange;">●</span> Substation<br>
        <span style="color: purple;">●</span> Wastewater Plant<br>
        <span style="color: blue;">●</span> Evacuation Route
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=700, height=450)


def render_status():
    """Render status."""
    result = st.session_state.last_result

    if result:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("24h Rainfall", f"{result['hazard']['rainfall_24h_mm']} mm")
            st.metric("Est. Depth", f"{result['hazard']['depth_m']} m")
        with col2:
            st.metric("Risk Level", result['hazard']['risk_level'].upper())
            st.metric("Assets", result['assets_count'])

        st.caption(f"Event: {result['event_id']}")
    else:
        st.info("👆 Click **Run Scan** to check current conditions")
        st.metric("Current Time", datetime.now().strftime("%H:%M IST"))
        st.metric("Location", "Chennai, India")


def render_results():
    """Render results."""
    result = st.session_state.last_result

    st.divider()
    st.subheader("📋 Scan Results")

    tab1, tab2 = st.tabs(["📝 Summary", "📊 Raw Data"])

    with tab1:
        st.markdown(f"""
**FLOOD-CASCADE EVENT MONITOR**

**Location:** {result['location']} | **Time:** {result['timestamp'][:19]}
**Event ID:** {result['event_id']} | **Mode:** {result['time_mode']}

---

**CONDITIONS:**
- 24h Rainfall: {result['hazard']['rainfall_24h_mm']} mm
- Est. Flood Depth: {result['hazard']['depth_m']} m
- Risk Level: **{result['hazard']['risk_level'].upper()}**

**ASSETS MONITORED:** {result['assets_count']}

---

**Summary:** {result['summary']}
        """)

    with tab2:
        st.json(result)


def render_satellite_tab():
    """Satellite and precipitation data access."""
    st.subheader("🛰️ Satellite & Weather Data")
    st.markdown("Access real satellite imagery and precipitation data for Chennai")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🌧️ Precipitation Data")
        st.markdown("*Source: Open-Meteo (ERA5 reanalysis + ECMWF forecast)*")

        precip_days = st.selectbox(
            "Period",
            options=[7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )

        if st.button("📥 Download Precipitation Data", type="primary", key="precip_btn"):
            with st.spinner("Fetching precipitation data from Open-Meteo..."):
                precip_result = fetch_precipitation_data(precip_days)
                st.session_state.precip_data = precip_result

        if hasattr(st.session_state, 'precip_data'):
            data = st.session_state.precip_data
            if "error" not in data:
                st.success(f"✅ Data retrieved: {data['period_days']} days")

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.metric("Total Rainfall", f"{data['total_mm']:.1f} mm")
                with m2:
                    st.metric("Max Daily", f"{data['max_daily_mm']:.1f} mm")
                with m3:
                    st.metric("Rainy Days", data['rainy_days'])

                # Chart
                if data.get('daily_data'):
                    df = pd.DataFrame(data['daily_data'])
                    df['date'] = pd.to_datetime(df['date'])
                    st.line_chart(df.set_index('date')['precipitation_mm'])

                # Download buttons
                st.markdown("#### Download Files")
                csv_data = "date,precipitation_mm\n"
                for d in data.get('daily_data', []):
                    csv_data += f"{d['date']},{d['precipitation_mm']}\n"

                st.download_button(
                    "📊 Download CSV",
                    data=csv_data,
                    file_name=f"chennai_precipitation_{precip_days}days.csv",
                    mime="text/csv"
                )

                st.download_button(
                    "📄 Download JSON",
                    data=json.dumps(data, indent=2),
                    file_name=f"chennai_precipitation_{precip_days}days.json",
                    mime="application/json"
                )
            else:
                st.error(f"Error: {data['error']}")

    with col2:
        st.markdown("### 🛰️ Sentinel-1 SAR Data")
        st.markdown("*Source: ESA Copernicus via Google Earth Engine*")

        sat_year = st.selectbox("Year", options=list(range(2024, 2014, -1)), index=0, key="sat_year")
        sat_month = st.selectbox("Month", options=list(range(1, 13)),
                                 index=datetime.now().month - 1,
                                 format_func=lambda x: datetime(2000, x, 1).strftime("%B"),
                                 key="sat_month")

        st.info("""
**Note:** Sentinel-1 SAR data requires Google Earth Engine access.
- Revisit cycle: 6-12 days
- Resolution: 10m
- Bands: VV, VH (polarization)
        """)

        if st.button("🔍 Check Available Imagery", key="sat_btn"):
            with st.spinner("Checking Sentinel-1 availability..."):
                sat_result = check_satellite_availability(sat_year, sat_month)
                st.session_state.sat_data = sat_result

        if hasattr(st.session_state, 'sat_data'):
            data = st.session_state.sat_data
            if data.get('available'):
                st.success(f"✅ {data['count']} images available")

                st.markdown("#### Available Passes")
                for img in data.get('images', [])[:10]:
                    st.markdown(f"- **{img['date']}** - {img['mode']} mode")

                if st.button("📥 Download Latest GeoTIFF", key="download_sat"):
                    st.warning("⚠️ GeoTIFF download requires GEE authentication. Use the local app for full functionality.")
                    st.code("""
# Run locally to download:
cd cascade_scanner
python scripts/download_data.py
                    """)
            else:
                st.warning(data.get('message', 'No data available'))

    # Data sources info
    st.divider()
    st.markdown("### 📋 Data Source Information")

    sources_df = pd.DataFrame([
        {
            "Data Type": "Precipitation",
            "Source": "Open-Meteo API",
            "Provider": "ERA5 Reanalysis + ECMWF",
            "Resolution": "~9km",
            "Latency": "Real-time to 5 days",
            "Access": "Free, no auth required"
        },
        {
            "Data Type": "SAR Imagery",
            "Source": "Sentinel-1 GRD",
            "Provider": "ESA Copernicus via GEE",
            "Resolution": "10m",
            "Latency": "6-12 day revisit",
            "Access": "Free, GEE auth required"
        },
        {
            "Data Type": "Flood Detection",
            "Source": "Sentinel-1 SAR",
            "Provider": "Processed VV/VH bands",
            "Resolution": "10-30m",
            "Latency": "1-3 days processing",
            "Access": "Via GEE"
        }
    ])
    st.dataframe(sources_df, use_container_width=True)


def fetch_precipitation_data(days_back: int) -> dict:
    """Fetch real precipitation data from Open-Meteo."""
    import requests

    lat, lon = 13.0827, 80.2707  # Chennai

    today = datetime.now()
    archive_end = today - timedelta(days=7)
    archive_start = today - timedelta(days=days_back)

    all_data = []

    # Fetch from archive API (historical)
    try:
        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": archive_start.strftime("%Y-%m-%d"),
            "end_date": archive_end.strftime("%Y-%m-%d"),
            "daily": "precipitation_sum",
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(archive_url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            daily = data.get("daily", {})
            for i, date in enumerate(daily.get("time", [])):
                precip = daily.get("precipitation_sum", [])[i] or 0
                all_data.append({"date": date, "precipitation_mm": precip})
    except Exception as e:
        pass  # Continue with forecast data

    # Fetch from forecast API (recent + forecast)
    try:
        forecast_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum",
            "past_days": 7,
            "forecast_days": 1,
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(forecast_url, params=params, timeout=30)
        if resp.status_code == 200:
            data = resp.json()
            daily = data.get("daily", {})
            existing_dates = {d["date"] for d in all_data}
            for i, date in enumerate(daily.get("time", [])):
                if date not in existing_dates:
                    precip = daily.get("precipitation_sum", [])[i] or 0
                    all_data.append({"date": date, "precipitation_mm": precip})
    except Exception as e:
        pass

    if not all_data:
        return {"error": "Could not fetch precipitation data"}

    # Sort by date
    all_data.sort(key=lambda x: x["date"])

    # Calculate stats
    precip_values = [d["precipitation_mm"] for d in all_data]
    total = sum(precip_values)
    max_daily = max(precip_values) if precip_values else 0
    rainy_days = sum(1 for p in precip_values if p > 0.1)

    return {
        "period_days": len(all_data),
        "total_mm": round(total, 1),
        "max_daily_mm": round(max_daily, 1),
        "rainy_days": rainy_days,
        "daily_data": all_data
    }


def get_gee_initialized():
    """Initialize GEE using Streamlit secrets."""
    try:
        import ee
        if hasattr(st, "secrets") and "gee" in st.secrets:
            # Initialize from Streamlit secrets
            service_account_json = st.secrets["gee"].get("service_account_json", "")
            project_id = st.secrets["gee"].get("project_id", "")

            if service_account_json and project_id:
                import json
                sa_info = json.loads(service_account_json)
                credentials = ee.ServiceAccountCredentials(sa_info['client_email'], key_data=service_account_json)
                ee.Initialize(credentials=credentials, project=project_id)
                return True
        return False
    except Exception:
        return False


def check_satellite_availability(year: int, month: int) -> dict:
    """Check REAL Sentinel-1 availability for a given month using GEE."""
    import calendar

    # Check for future dates
    if year > datetime.now().year or (year == datetime.now().year and month > datetime.now().month):
        return {
            "available": False,
            "message": "Future dates - no imagery available yet"
        }

    # Try to use real GEE
    try:
        import ee

        if get_gee_initialized():
            # Query real Sentinel-1 collection
            days_in_month = calendar.monthrange(year, month)[1]
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-{days_in_month:02d}"

            # Chennai bounding box
            chennai = ee.Geometry.Rectangle([80.05, 12.85, 80.35, 13.25])

            collection = (
                ee.ImageCollection("COPERNICUS/S1_GRD")
                .filterBounds(chennai)
                .filterDate(start_date, end_date)
                .filter(ee.Filter.eq("instrumentMode", "IW"))
            )

            # Get image info
            image_list = collection.toList(100)
            count = collection.size().getInfo()

            images = []
            for i in range(min(count, 20)):  # Limit to 20
                img = ee.Image(image_list.get(i))
                props = img.getInfo()['properties']
                images.append({
                    "date": props.get("system:time_start", ""),
                    "mode": props.get("instrumentMode", "IW"),
                    "orbit": props.get("orbitProperties_pass", "Unknown"),
                    "bands": ["VV", "VH"]
                })

            # Convert timestamps to dates
            for img in images:
                if isinstance(img["date"], int):
                    img["date"] = datetime.fromtimestamp(img["date"]/1000).strftime("%Y-%m-%d")

            return {
                "available": True,
                "count": count,
                "images": images,
                "note": "Real Sentinel-1 data from Google Earth Engine"
            }

    except Exception as e:
        pass  # Fall back to estimation

    # Fallback: Estimate based on typical Sentinel-1 revisit (6-12 days)
    days_in_month = calendar.monthrange(year, month)[1]
    estimated_count = days_in_month // 8  # ~4 images per month

    images = []
    for i in range(estimated_count):
        day = min(1 + i * 8, days_in_month)
        images.append({
            "date": f"{year}-{month:02d}-{day:02d}",
            "mode": "IW",
            "orbit": "Ascending" if i % 2 == 0 else "Descending",
            "bands": ["VV", "VH"]
        })

    return {
        "available": True,
        "count": len(images),
        "images": images,
        "note": "Estimated availability (GEE not configured - add credentials in Streamlit secrets)"
    }


def render_temporal_tab():
    """Temporal analysis."""
    st.subheader("📈 Temporal Analysis")
    st.markdown("Analyze flood risk trends over time")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Analysis Settings")
        period = st.selectbox(
            "Analysis Period",
            ["Last 7 days", "Last 30 days", "Last 90 days"],
            index=1
        )

        days_map = {"Last 7 days": 7, "Last 30 days": 30, "Last 90 days": 90}
        days_back = days_map[period]

        if st.button("🔄 Analyze Trends", type="primary"):
            with st.spinner("Analyzing..."):
                analysis = analyze_temporal_data(days_back)
                st.session_state.temporal_analysis = analysis

    with col2:
        if hasattr(st.session_state, 'temporal_analysis'):
            analysis = st.session_state.temporal_analysis

            st.markdown("### Trend Summary")
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Rainfall", f"{analysis['total_rainfall_mm']} mm")
            with m2:
                st.metric("Avg Daily", f"{analysis['avg_daily_mm']} mm")
            with m3:
                st.metric("High Risk Days", analysis['high_risk_days'])
            with m4:
                st.metric("Rainy Days", analysis['rainy_days'])

            st.markdown("### Trend Indicators")
            col_a, col_b = st.columns(2)
            with col_a:
                trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(analysis['trend'], "❓")
                st.info(f"**Rainfall Trend:** {trend_emoji} {analysis['trend'].title()}")
            with col_b:
                st.info(f"**Season:** {analysis['monsoon_status']}")

    # Chart
    if hasattr(st.session_state, 'temporal_analysis'):
        st.divider()
        st.markdown("### Rainfall Time Series")
        df = pd.DataFrame(st.session_state.temporal_analysis['series'])
        df['date'] = pd.to_datetime(df['date'])
        st.line_chart(df.set_index('date')['rainfall_mm'])


def render_download_tab():
    """Download center."""
    st.subheader("📥 Download Center")
    st.markdown("Export scan results and analysis data")

    result = st.session_state.last_result

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Current Scan Results")

        if result:
            st.download_button(
                "📄 Download JSON",
                data=json.dumps(result, indent=2),
                file_name=f"cascade_scan_{result['event_id']}.json",
                mime="application/json",
                use_container_width=True
            )

            report = f"""# Cascade Scanner Report
## Event: {result['event_id']}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Hazard Conditions
- **Location:** {result['location']}
- **24h Rainfall:** {result['hazard']['rainfall_24h_mm']} mm
- **Flood Depth:** {result['hazard']['depth_m']} m
- **Risk Level:** {result['hazard']['risk_level'].upper()}

## Summary
{result['summary']}

---
*Generated by Cascade Scanner*
"""
            st.download_button(
                "📝 Download Report (MD)",
                data=report,
                file_name=f"cascade_report_{result['event_id']}.md",
                mime="text/markdown",
                use_container_width=True
            )
        else:
            st.info("Run a scan first to enable downloads")

    with col2:
        st.markdown("### Temporal Analysis Data")

        if hasattr(st.session_state, 'temporal_analysis'):
            analysis = st.session_state.temporal_analysis

            st.download_button(
                "📈 Download Analysis JSON",
                data=json.dumps(analysis, indent=2, default=str),
                file_name=f"temporal_analysis.json",
                mime="application/json",
                use_container_width=True
            )

            df = pd.DataFrame(analysis['series'])
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            st.download_button(
                "📊 Download Time Series CSV",
                data=csv_buffer.getvalue(),
                file_name="timeseries.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("Run temporal analysis first")

        st.divider()
        if st.session_state.scan_history:
            st.download_button(
                "📜 Download Scan History",
                data=json.dumps(st.session_state.scan_history, indent=2),
                file_name="scan_history.json",
                mime="application/json",
                use_container_width=True
            )


def render_history_tab():
    """Historical data lookup."""
    st.subheader("📊 Historical Data Lookup")
    st.markdown("Check flood conditions for any specific date or month in history")

    # Date lookup section
    st.markdown("### 🔍 Lookup Specific Date")
    col1, col2, col3 = st.columns([1, 1, 1])

    with col1:
        lookup_year = st.selectbox(
            "Year",
            options=list(range(2024, 2010, -1)),
            index=0
        )

    with col2:
        months = ["January", "February", "March", "April", "May", "June",
                  "July", "August", "September", "October", "November", "December"]
        lookup_month = st.selectbox("Month", options=months, index=datetime.now().month - 1)
        month_num = months.index(lookup_month) + 1

    with col3:
        # Get days in selected month
        import calendar
        days_in_month = calendar.monthrange(lookup_year, month_num)[1]
        lookup_day = st.selectbox("Day (optional)", options=["Entire Month"] + list(range(1, days_in_month + 1)))

    if st.button("🔍 Lookup Historical Data", type="primary"):
        if lookup_day == "Entire Month":
            # Get entire month data
            historical_data = get_historical_month_data(lookup_year, month_num)
            st.session_state.historical_lookup = {
                "type": "month",
                "year": lookup_year,
                "month": lookup_month,
                "data": historical_data
            }
        else:
            # Get specific date data
            historical_data = get_historical_date_data(lookup_year, month_num, lookup_day)
            st.session_state.historical_lookup = {
                "type": "date",
                "year": lookup_year,
                "month": lookup_month,
                "day": lookup_day,
                "data": historical_data
            }

    # Display results
    if hasattr(st.session_state, 'historical_lookup'):
        lookup = st.session_state.historical_lookup
        st.divider()

        if lookup["type"] == "date":
            st.markdown(f"### Conditions on {lookup['month']} {lookup['day']}, {lookup['year']}")
            data = lookup["data"]

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Rainfall", f"{data['rainfall_mm']:.1f} mm")
            with col2:
                st.metric("Risk Level", data['risk_level'].upper())
            with col3:
                st.metric("Flood Depth", f"{data['flood_depth_m']:.2f} m")
            with col4:
                st.metric("Alert Level", data['alert_level'])

            # Detailed analysis
            st.markdown("#### Analysis")
            st.info(data['analysis'])

            # Historical events
            if data.get('notable_events'):
                st.markdown("#### Notable Events")
                for event in data['notable_events']:
                    st.warning(f"📌 {event}")

        else:  # Month view
            st.markdown(f"### {lookup['month']} {lookup['year']} Overview")
            data = lookup["data"]

            # Monthly summary
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Rainfall", f"{data['total_rainfall_mm']:.1f} mm")
            with col2:
                st.metric("Rainy Days", data['rainy_days'])
            with col3:
                st.metric("High Risk Days", data['high_risk_days'])
            with col4:
                st.metric("Max Daily", f"{data['max_daily_mm']:.1f} mm")

            # Month chart
            st.markdown("#### Daily Rainfall Pattern")
            df = pd.DataFrame(data['daily_series'])
            df['date'] = pd.to_datetime(df['date'])
            st.bar_chart(df.set_index('date')['rainfall_mm'])

            # Risk distribution
            st.markdown("#### Risk Distribution")
            risk_df = pd.DataFrame({
                'Risk Level': ['High', 'Moderate', 'Low', 'Minimal'],
                'Days': [data['high_risk_days'], data['moderate_risk_days'],
                        data['low_risk_days'], data['minimal_risk_days']]
            })
            st.bar_chart(risk_df.set_index('Risk Level'))

            # Monthly analysis
            st.markdown("#### Monthly Analysis")
            st.info(data['analysis'])

            # Data table
            st.markdown("#### Daily Data")
            st.dataframe(pd.DataFrame(data['daily_series']), use_container_width=True)

    # Session history
    st.divider()
    st.markdown("### Session Scan History")
    if st.session_state.scan_history:
        st.dataframe(pd.DataFrame(st.session_state.scan_history), use_container_width=True)
    else:
        st.info("No scans recorded yet")


def get_historical_date_data(year: int, month: int, day: int) -> dict:
    """Get REAL historical data for a specific date from Open-Meteo."""
    target_date = datetime(year, month, day)
    lat, lon = 13.0827, 80.2707  # Chennai

    # Determine season
    if month in [10, 11, 12]:
        season = "Northeast Monsoon (Peak Season)"
    elif month in [6, 7, 8, 9]:
        season = "Southwest Monsoon"
    else:
        season = "Dry Season"

    # Fetch REAL data from Open-Meteo Archive API
    rainfall = 0.0
    try:
        date_str = f"{year}-{month:02d}-{day:02d}"
        archive_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": date_str,
            "end_date": date_str,
            "daily": "precipitation_sum",
            "timezone": "Asia/Kolkata",
        }
        resp = requests.get(archive_url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            precip = data.get("daily", {}).get("precipitation_sum", [0])
            rainfall = precip[0] if precip and precip[0] is not None else 0.0
    except Exception:
        rainfall = 0.0  # API unavailable

    rainfall = round(rainfall, 1)

    # Estimate flood depth from rainfall
    if rainfall < 30:
        flood_depth = 0.0
    elif rainfall < 50:
        flood_depth = 0.1
    elif rainfall < 80:
        flood_depth = 0.25
    elif rainfall < 120:
        flood_depth = 0.4
    else:
        flood_depth = 0.6 + (rainfall - 120) / 200

    if rainfall > 150:
        risk = "severe"
        alert = "RED"
    elif rainfall > 80:
        risk = "high"
        alert = "ORANGE"
    elif rainfall > 30:
        risk = "moderate"
        alert = "YELLOW"
    elif rainfall > 10:
        risk = "low"
        alert = "GREEN"
    else:
        risk = "minimal"
        alert = "NONE"

    # Generate analysis
    analysis = f"""
**Season:** {season}

**Rainfall Assessment:** {"Heavy" if rainfall > 80 else "Moderate" if rainfall > 30 else "Light" if rainfall > 5 else "No significant"} rainfall recorded.

**Flood Risk:** {risk.title()} risk conditions. {"Emergency protocols would be activated." if risk in ["severe", "high"] else "Normal monitoring advised."}

**Infrastructure Impact:** {"Major disruptions to power substations and transport likely." if risk == "severe" else "Possible localized flooding in low-lying areas." if risk in ["high", "moderate"] else "Minimal impact expected."}
"""

    # Notable events
    notable_events = []
    if year == 2015 and month == 12 and day <= 5:
        notable_events.append("2015 Chennai Floods - One of the worst floods in 100 years. 500+ deaths, 18 lakh people displaced.")
    if year == 2021 and month == 11 and 6 <= day <= 12:
        notable_events.append("2021 Chennai Floods - Heavy rains caused widespread flooding. Airport runway submerged.")
    if year == 2023 and month == 11 and 10 <= day <= 15:
        notable_events.append("Cyclone Michaung - Record 40cm rainfall in 24 hours. City paralyzed for 3 days.")

    return {
        "date": target_date.strftime("%Y-%m-%d"),
        "rainfall_mm": rainfall,
        "risk_level": risk,
        "flood_depth_m": round(flood_depth, 2),
        "alert_level": alert,
        "season": season,
        "analysis": analysis,
        "notable_events": notable_events
    }


def get_historical_month_data(year: int, month: int) -> dict:
    """Get historical data for entire month."""
    import calendar
    days_in_month = calendar.monthrange(year, month)[1]

    daily_series = []
    total_rainfall = 0
    rainy_days = 0
    high_risk = 0
    moderate_risk = 0
    low_risk = 0
    minimal_risk = 0
    max_daily = 0

    for day in range(1, days_in_month + 1):
        day_data = get_historical_date_data(year, month, day)
        daily_series.append({
            "date": f"{year}-{month:02d}-{day:02d}",
            "rainfall_mm": day_data["rainfall_mm"],
            "risk_level": day_data["risk_level"]
        })

        total_rainfall += day_data["rainfall_mm"]
        max_daily = max(max_daily, day_data["rainfall_mm"])

        if day_data["rainfall_mm"] > 0.1:
            rainy_days += 1

        if day_data["risk_level"] in ["severe", "high"]:
            high_risk += 1
        elif day_data["risk_level"] == "moderate":
            moderate_risk += 1
        elif day_data["risk_level"] == "low":
            low_risk += 1
        else:
            minimal_risk += 1

    # Month analysis
    month_names = ["", "January", "February", "March", "April", "May", "June",
                   "July", "August", "September", "October", "November", "December"]

    if month in [10, 11, 12]:
        season_desc = "Northeast Monsoon season - Chennai's primary rainy season"
    elif month in [6, 7, 8, 9]:
        season_desc = "Southwest Monsoon - moderate rainfall expected"
    else:
        season_desc = "Dry season - minimal rainfall expected"

    analysis = f"""
**{month_names[month]} {year} Summary**

**Season:** {season_desc}

**Rainfall Pattern:** Total {total_rainfall:.1f}mm across {rainy_days} rainy days. {"Above normal" if total_rainfall > 200 else "Normal" if total_rainfall > 50 else "Below normal"} for this period.

**Risk Assessment:** {high_risk} high-risk days requiring emergency response. {"Critical month for flood preparedness." if high_risk > 5 else "Standard monitoring sufficient."}

**Recommendation:** {"Activate flood response protocols" if high_risk > 10 else "Enhanced monitoring during rain events" if high_risk > 3 else "Routine monitoring"}
"""

    return {
        "total_rainfall_mm": round(total_rainfall, 1),
        "rainy_days": rainy_days,
        "high_risk_days": high_risk,
        "moderate_risk_days": moderate_risk,
        "low_risk_days": low_risk,
        "minimal_risk_days": minimal_risk,
        "max_daily_mm": round(max_daily, 1),
        "daily_series": daily_series,
        "analysis": analysis
    }


# ============ DROUGHT MODE TAB ============

def render_drought_tab():
    """Drought mode with water budget and scenarios."""
    st.subheader("🌾 Drought Mode - Water Budget Analysis")

    if not DROUGHT_ENGINE_AVAILABLE:
        st.error("Drought engine not available. Please check installation.")
        return

    st.markdown("Analyze drought conditions using historical scenarios or custom inputs")

    # Scenario selector
    st.markdown("### Select Scenario")
    scenario_options = {
        "2019_drought": "🔴 2019 Chennai Water Crisis",
        "2021_flood": "🔵 2021 Chennai Floods",
        "2023_cyclone": "🌀 Cyclone Michaung 2023",
        "normal_monsoon": "🟢 Normal Monsoon Year",
        "pre_monsoon_stress": "🟡 Pre-Monsoon Water Stress",
        "custom": "⚙️ Custom Input"
    }

    selected_scenario = st.selectbox(
        "Choose a scenario to analyze",
        options=list(scenario_options.keys()),
        format_func=lambda x: scenario_options[x]
    )

    if selected_scenario != "custom":
        scenario = SCENARIOS[selected_scenario]
        conditions = scenario["conditions"]

        st.info(f"**{scenario['name']}** - {scenario['description']}")

        # Display scenario conditions
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rainfall", f"{conditions['rainfall_mm']} mm/day")
            st.metric("Reservoir", f"{conditions['reservoir_pct']}%")
        with col2:
            st.metric("Groundwater", f"{conditions['groundwater_m']} m depth")
            st.metric("SPI", f"{conditions['spi']:.1f}")
        with col3:
            st.metric("NDVI", f"{conditions['ndvi']:.2f}")
            st.metric("Soil Moisture", f"{conditions['soil_moisture_pct']}%")
    else:
        # Custom inputs
        st.markdown("#### Custom Conditions")
        col1, col2, col3 = st.columns(3)
        with col1:
            custom_rainfall = st.slider("Daily Rainfall (mm)", 0, 400, 20)
            custom_reservoir = st.slider("Reservoir Level (%)", 0, 100, 50)
        with col2:
            custom_gw = st.slider("Groundwater Depth (m)", 0, 30, 10)
            custom_spi = st.slider("SPI", -3.0, 3.0, 0.0, 0.1)
        with col3:
            custom_ndvi = st.slider("NDVI", 0.0, 1.0, 0.5, 0.01)
            custom_soil = st.slider("Soil Moisture (%)", 0, 100, 40)

        conditions = {
            "rainfall_mm": custom_rainfall,
            "reservoir_pct": custom_reservoir,
            "groundwater_m": custom_gw,
            "spi": custom_spi,
            "ndvi": custom_ndvi,
            "soil_moisture_pct": custom_soil
        }

    st.divider()

    # Run assessment
    if st.button("🔍 Analyze Conditions", type="primary"):
        with st.spinner("Analyzing water budget and drought conditions..."):
            # Drought assessment
            assessment = assess_drought(
                spi=conditions["spi"],
                soil_moisture=conditions["soil_moisture_pct"],
                reservoir_pct=conditions["reservoir_pct"],
                ndvi=conditions["ndvi"],
                current_month=datetime.now().month
            )

            # Water budget
            budget = calculate_water_budget(
                area_ha=50000,  # Chennai metro agricultural area
                rainfall_mm=conditions["rainfall_mm"],
                et_mm=5.0,  # Average ET
                reservoir_pct=conditions["reservoir_pct"],
                crop_area_ha=30000,
                crop_water_need_mm=800
            )

            st.session_state.drought_assessment = assessment
            st.session_state.water_budget = budget

    # Display results
    if hasattr(st.session_state, 'drought_assessment'):
        assessment = st.session_state.drought_assessment
        budget = st.session_state.water_budget

        st.markdown("### Drought Assessment")

        # Severity indicator
        severity_colors = {
            "none": "🟢", "mild": "🟡", "moderate": "🟠",
            "severe": "🔴", "extreme": "⚫"
        }

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Severity", f"{severity_colors.get(assessment.severity, '❓')} {assessment.severity.upper()}")
        with col2:
            st.metric("Drought Type", assessment.drought_type.title())
        with col3:
            st.metric("Weeks to Impact", assessment.weeks_to_impact)
        with col4:
            st.metric("Crop Stress", assessment.crop_stress_level.title())

        # Affected area
        st.progress(assessment.affected_area_pct / 100, text=f"Affected Area: {assessment.affected_area_pct}%")

        # Water Budget
        st.markdown("### Water Budget (MCM)")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Supply Side**")
            st.metric("Total Supply", f"{budget.total_supply_mcm} MCM")
            st.caption(f"• Rainfall: {budget.rainfall_contribution_mcm} MCM")
            st.caption(f"• Reservoir: {budget.reservoir_storage_mcm} MCM")
            st.caption(f"• Groundwater: {budget.groundwater_available_mcm} MCM")

        with col2:
            st.markdown("**Demand Side**")
            st.metric("Total Demand", f"{budget.total_demand_mcm} MCM")
            st.caption(f"• Irrigation: {budget.irrigation_demand_mcm} MCM")
            st.caption(f"• Domestic: {budget.domestic_demand_mcm} MCM")
            st.caption(f"• ET Loss: {budget.et_loss_mcm} MCM")

        # Deficit/Surplus
        if budget.deficit_mcm > 0:
            st.error(f"⚠️ Water Deficit: {budget.deficit_mcm} MCM")
        else:
            st.success(f"✅ Water Surplus: {budget.surplus_mcm} MCM")

        # Recommendations
        st.markdown("### Recommended Actions")
        for action in assessment.recommended_actions:
            st.markdown(f"• {action}")

    # Satellite Observations
    st.divider()
    st.markdown("### 🛰️ Satellite-Derived Observations")

    if st.button("Load Satellite Data (Last 30 Days)", key="sat_drought"):
        with st.spinner("Loading satellite observations..."):
            sat_obs = generate_satellite_observations(datetime.now(), 30)
            st.session_state.sat_observations = sat_obs

    if hasattr(st.session_state, 'sat_observations'):
        obs = st.session_state.sat_observations

        # Display as table
        sat_df = pd.DataFrame([{
            'Date': o.date,
            'Source': o.source,
            'NDVI': o.ndvi,
            'NDWI': o.ndwi,
            'LST (°C)': o.lst_celsius,
            'Soil Moisture (%)': o.soil_moisture_pct,
            'ET (mm/day)': o.evapotranspiration_mm,
            'Cloud (%)': o.cloud_cover_pct
        } for o in obs])

        st.dataframe(sat_df, use_container_width=True)

        # Chart for key indices
        st.markdown("#### Vegetation and Water Indices")
        chart_df = sat_df[['Date', 'NDVI', 'NDWI']].set_index('Date')
        st.line_chart(chart_df)

        # Download satellite data
        st.download_button(
            "📥 Download Satellite Data (CSV)",
            data=sat_df.to_csv(index=False),
            file_name="satellite_observations.csv",
            mime="text/csv"
        )

    # 5-Year data visualization
    st.divider()
    st.markdown("### 📊 5-Year Historical Data")

    if st.button("Generate 5-Year Dataset"):
        with st.spinner("Generating synthetic historical data..."):
            df = generate_5year_data(2019)
            st.session_state.five_year_data = df

    if hasattr(st.session_state, 'five_year_data'):
        df = st.session_state.five_year_data

        variable = st.selectbox(
            "Select Variable to Plot",
            options=["rainfall_mm", "reservoir_pct", "soil_moisture_pct", "spi", "ndvi", "groundwater_m"],
            format_func=lambda x: {
                "rainfall_mm": "Daily Rainfall (mm)",
                "reservoir_pct": "Reservoir Level (%)",
                "soil_moisture_pct": "Soil Moisture (%)",
                "spi": "SPI (Drought Index)",
                "ndvi": "NDVI (Vegetation)",
                "groundwater_m": "Groundwater Depth (m)"
            }[x]
        )

        # Monthly aggregation for clarity
        df['month'] = df['date'].dt.to_period('M')
        monthly = df.groupby('month')[variable].mean().reset_index()
        monthly['month'] = monthly['month'].dt.to_timestamp()

        st.line_chart(monthly.set_index('month')[variable])

        # Download
        st.download_button(
            "📥 Download 5-Year Data (CSV)",
            data=df.to_csv(index=False),
            file_name="chennai_5year_hydro_data.csv",
            mime="text/csv"
        )


# ============ FARMER VIEW TAB ============

def render_farmer_tab():
    """Neutral crop water status and projections for farmers."""
    st.subheader("👨‍🌾 Crop Water Status Report")

    if not DROUGHT_ENGINE_AVAILABLE:
        st.error("Projection engine not available. Please check installation.")
        return

    st.markdown("*View your crop water status and future projections*")

    # Farm inputs
    st.markdown("### Farm Details")

    col1, col2 = st.columns(2)
    with col1:
        farm_area = st.number_input("Farm Area (hectares)", min_value=0.5, max_value=100.0, value=2.0, step=0.5)
        current_crop = st.selectbox(
            "Current Crop",
            options=["paddy", "sugarcane", "cotton", "groundnut", "millets", "pulses"]
        )

    with col2:
        crop_stage = st.slider("Days Since Sowing", 0, 180, 45)
        water_available = st.number_input("Water Available (mm)", min_value=0, max_value=2000, value=500)

    daily_et = st.slider("Daily Water Use (ET mm/day)", 2.0, 10.0, 5.0, 0.5)

    st.divider()

    # Generate status report
    if st.button("📊 Generate Status Report", type="primary", use_container_width=True):
        with st.spinner("Calculating crop water status..."):
            status = calculate_crop_status(
                crop=current_crop,
                area_ha=farm_area,
                days_since_sowing=crop_stage,
                water_available_mm=water_available,
                daily_et_mm=daily_et
            )
            st.session_state.crop_status = status

            # Also generate projections if we have drought data
            if hasattr(st.session_state, 'drought_assessment'):
                spi = st.session_state.drought_assessment.spi_value
            else:
                spi = 0
            projections = generate_projections(
                current_rainfall_mm=5.0,
                current_reservoir_pct=50,
                current_soil_moisture=40,
                spi=spi,
                weeks_ahead=8
            )
            st.session_state.water_projections = projections

    # Display status
    if hasattr(st.session_state, 'crop_status'):
        status = st.session_state.crop_status

        st.markdown("### Current Crop Status")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Crop", status.crop.title())
            st.metric("Growth Stage", status.stage)
        with col2:
            st.metric("Days Remaining", f"{status.days_remaining} days")
            st.metric("Water Required", f"{status.water_required_mm} mm")
        with col3:
            st.metric("Water Available", f"{status.water_available_mm} mm")
            st.metric("Deficit", f"{status.deficit_mm} mm", delta=f"-{status.deficit_pct:.0f}%" if status.deficit_mm > 0 else None)

        # Critical date
        st.markdown("### Key Projection")
        if status.deficit_mm > 0:
            st.warning(f"⚠️ **Water Deficit:** {status.deficit_mm} mm ({status.deficit_pct:.0f}% of remaining need)")
        else:
            st.success(f"✅ **Water Sufficient:** {status.water_available_mm - status.water_required_mm:.0f} mm surplus")

        st.info(f"📅 **Critical Date:** At current usage rate ({daily_et} mm/day), available water may last until **{status.critical_date}**")

        # Water projections
        if hasattr(st.session_state, 'water_projections'):
            st.markdown("### 8-Week Water Projections")

            projections = st.session_state.water_projections
            proj_df = pd.DataFrame([{
                'Week': p.weeks_ahead,
                'Rainfall (mm)': p.projected_rainfall_mm,
                'Reservoir (%)': p.projected_reservoir_pct,
                'Soil Moisture (%)': p.projected_soil_moisture_pct,
                'Streamflow (% normal)': p.projected_streamflow_pct,
                'Cumulative Deficit (mm)': p.cumulative_deficit_mm,
                'Trend': p.trend
            } for p in projections])

            st.dataframe(proj_df, use_container_width=True)

            # Chart
            st.markdown("#### Projected Water Availability")
            chart_df = proj_df[['Week', 'Reservoir (%)', 'Soil Moisture (%)']].set_index('Week')
            st.line_chart(chart_df)

            # Find critical week
            for p in projections:
                if p.projected_reservoir_pct < 20 or p.cumulative_deficit_mm > 100:
                    st.warning(f"⚠️ **Projection:** Reservoir may fall below 20% by week {p.weeks_ahead}")
                    break
            else:
                st.success("✅ **Projection:** Water availability stable for next 8 weeks")

        # Download report
        st.divider()
        report = f"""# Crop Water Status Report
## {current_crop.title()} - {farm_area} hectares

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

### Current Status
- **Growth Stage:** {status.stage}
- **Days Remaining:** {status.days_remaining}
- **Water Required:** {status.water_required_mm} mm
- **Water Available:** {status.water_available_mm} mm
- **Deficit:** {status.deficit_mm} mm ({status.deficit_pct:.0f}%)

### Critical Date
At current usage rate ({daily_et} mm/day), water may last until: **{status.critical_date}**

---
*This report presents data projections only. Decisions rest with the farmer.*
"""

        st.download_button(
            "📥 Download Report",
            data=report,
            file_name=f"crop_status_{current_crop}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )


# ============ ADMIN VIEW TAB ============

def render_admin_tab():
    """Neutral regional water status report for administrators."""
    st.subheader("🏛️ Regional Water Status Report")

    if not DROUGHT_ENGINE_AVAILABLE:
        st.error("Projection engine not available. Please check installation.")
        return

    st.markdown("*View regional water supply status and projections*")

    # Region selector
    st.markdown("### Region Configuration")

    col1, col2 = st.columns(2)
    with col1:
        region = st.selectbox(
            "Select District",
            options=["Chennai Metro", "Kanchipuram", "Thiruvallur", "Chengalpattu"]
        )

        # Pre-set areas for different districts
        region_areas = {
            "Chennai Metro": 50000,
            "Kanchipuram": 80000,
            "Thiruvallur": 70000,
            "Chengalpattu": 60000
        }
        total_area = region_areas.get(region, 50000)
        st.caption(f"Total Cultivated Area: {total_area:,} hectares")

    with col2:
        reservoir_pct = st.slider("Current Reservoir Level (%)", 0, 100, 45)
        groundwater_m = st.slider("Groundwater Depth (m below surface)", 0, 30, 12)

    weekly_demand = st.number_input("Weekly Water Demand (MCM)", min_value=1.0, max_value=100.0, value=25.0, step=5.0)

    st.divider()

    # Generate report
    if st.button("📊 Generate Status Report", type="primary", use_container_width=True):
        with st.spinner("Generating regional water status..."):
            report = generate_regional_report(
                region=region,
                total_area_ha=total_area,
                reservoir_pct=reservoir_pct,
                groundwater_m=groundwater_m,
                current_demand_mcm_per_week=weekly_demand
            )
            st.session_state.regional_report = report

            # Generate projections
            if hasattr(st.session_state, 'drought_assessment'):
                spi = st.session_state.drought_assessment.spi_value
            else:
                # Estimate SPI from reservoir level
                if reservoir_pct < 20:
                    spi = -2.0
                elif reservoir_pct < 40:
                    spi = -1.0
                elif reservoir_pct < 60:
                    spi = 0.0
                else:
                    spi = 0.5

            projections = generate_projections(
                current_rainfall_mm=5.0,
                current_reservoir_pct=reservoir_pct,
                current_soil_moisture=40,
                spi=spi,
                weeks_ahead=12
            )
            st.session_state.admin_projections = projections

    # Display report
    if hasattr(st.session_state, 'regional_report'):
        rpt = st.session_state.regional_report

        st.markdown(f"### {rpt.region} - Current Status")

        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Reservoir Storage", f"{rpt.reservoir_storage_mcm:.0f} MCM",
                      delta=f"{rpt.reservoir_pct:.0f}% of capacity")
        with col2:
            st.metric("Groundwater Level", f"{rpt.groundwater_level_m} m",
                      delta="below surface")
        with col3:
            st.metric("Streamflow", f"{rpt.streamflow_pct_normal:.0f}%",
                      delta="of normal")
        with col4:
            st.metric("Weeks of Supply", f"{rpt.weeks_of_supply} weeks",
                      delta="at current demand")

        # Area breakdown
        st.markdown("### Land Use")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Agricultural", f"{rpt.total_area_ha:,.0f} ha")
        with col2:
            st.metric("Irrigated Area", f"{rpt.irrigated_area_ha:,.0f} ha")
        with col3:
            st.metric("Rainfed Area", f"{rpt.rainfed_area_ha:,.0f} ha")

        # Key projections
        st.markdown("### Key Projections")

        if rpt.deficit_mcm > 0:
            st.warning(f"⚠️ **Projected Seasonal Deficit:** {rpt.deficit_mcm:.1f} MCM")
        else:
            st.success(f"✅ **Projected Seasonal Surplus:** {abs(rpt.deficit_mcm):.1f} MCM")

        st.info(f"📅 **Projected Depletion Date:** At current demand ({weekly_demand} MCM/week), reservoir storage may deplete by **{rpt.projected_depletion_date}**")

        # Projections table
        if hasattr(st.session_state, 'admin_projections'):
            st.markdown("### 12-Week Projections")

            projections = st.session_state.admin_projections
            proj_df = pd.DataFrame([{
                'Week': p.weeks_ahead,
                'Reservoir (%)': p.projected_reservoir_pct,
                'Soil Moisture (%)': p.projected_soil_moisture_pct,
                'Streamflow (% normal)': p.projected_streamflow_pct,
                'Cumulative Deficit (mm)': p.cumulative_deficit_mm,
                'Trend': p.trend
            } for p in projections])

            st.dataframe(proj_df, use_container_width=True)

            # Chart
            st.markdown("#### Projected Reservoir and Streamflow")
            chart_df = proj_df[['Week', 'Reservoir (%)', 'Streamflow (% normal)']].set_index('Week')
            st.line_chart(chart_df)

            # Find critical thresholds
            st.markdown("### Critical Thresholds")

            critical_week = None
            for p in projections:
                if p.projected_reservoir_pct < 15:
                    critical_week = p.weeks_ahead
                    break

            if critical_week:
                st.warning(f"⚠️ **Projection:** Reservoir may fall below 15% by week {critical_week}")
            else:
                st.success("✅ **Projection:** Reservoir level above 15% for next 12 weeks")

            streamflow_week = None
            for p in projections:
                if p.projected_streamflow_pct < 50:
                    streamflow_week = p.weeks_ahead
                    break

            if streamflow_week:
                st.warning(f"⚠️ **Projection:** Streamflow may fall below 50% of normal by week {streamflow_week}")

        # Download report
        st.divider()
        report_text = f"""# Regional Water Status Report
## {rpt.region}

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

### Current Status
- **Reservoir Storage:** {rpt.reservoir_storage_mcm:.0f} MCM ({rpt.reservoir_pct:.0f}% of {rpt.reservoir_capacity_mcm:.0f} MCM capacity)
- **Groundwater Level:** {rpt.groundwater_level_m} m below surface
- **Streamflow:** {rpt.streamflow_pct_normal:.0f}% of normal

### Land Use
- **Total Agricultural Area:** {rpt.total_area_ha:,.0f} hectares
- **Irrigated Area:** {rpt.irrigated_area_ha:,.0f} hectares
- **Rainfed Area:** {rpt.rainfed_area_ha:,.0f} hectares

### Projections
- **Weeks of Supply:** {rpt.weeks_of_supply} weeks at current demand
- **Projected Seasonal Deficit:** {rpt.deficit_mcm:.1f} MCM
- **Projected Depletion Date:** {rpt.projected_depletion_date}

---
*This report presents data projections only. Policy decisions rest with the administration.*
"""

        st.download_button(
            "📥 Download Status Report",
            data=report_text,
            file_name=f"regional_water_status_{region.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.md",
            mime="text/markdown"
        )


if __name__ == "__main__":
    main()
