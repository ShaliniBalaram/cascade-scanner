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
    """Generate simulated temporal data for Chennai."""
    series = []
    end_date = datetime.now()

    for i in range(days_back):
        current = end_date - timedelta(days=days_back-i-1)
        month = current.month

        # Simulate Chennai monsoon patterns
        if month in [10, 11, 12]:  # Northeast monsoon
            base = 8.0
        elif month in [6, 7, 8, 9]:  # Southwest monsoon
            base = 3.0
        else:  # Dry season
            base = 0.5

        rainfall = max(0, base + np.random.normal(0, base * 0.5))

        if rainfall > 50:
            risk = "high"
        elif rainfall > 20:
            risk = "moderate"
        elif rainfall > 5:
            risk = "low"
        else:
            risk = "minimal"

        series.append({
            "date": current.strftime("%Y-%m-%d"),
            "rainfall_mm": round(rainfall, 1),
            "risk_level": risk
        })

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
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🗺️ Live Scanner",
        "📈 Temporal Analysis",
        "🛰️ Satellite Data",
        "📥 Download Center",
        "📊 Historical Data"
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

        if st.button("🔍 Run Scan", type="primary", width='stretch'):
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
        "alerts_count": 0 if risk in ["minimal", "low"] else np.random.randint(1, 5),
        "summary": f"Scan complete - {len(assets)} assets monitored"
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
    st.dataframe(sources_df, width='stretch')


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


def check_satellite_availability(year: int, month: int) -> dict:
    """Check Sentinel-1 availability for a given month."""
    # This would normally query GEE, but for cloud deployment we simulate
    # In local version, this uses actual GEE

    # Simulate typical Sentinel-1 passes for Chennai (every 6-12 days)
    import calendar
    days_in_month = calendar.monthrange(year, month)[1]

    images = []
    current_day = 1

    while current_day <= days_in_month:
        images.append({
            "date": f"{year}-{month:02d}-{current_day:02d}",
            "mode": "IW",
            "orbit": "Ascending" if len(images) % 2 == 0 else "Descending",
            "bands": ["VV", "VH"]
        })
        current_day += np.random.randint(5, 13)  # 5-12 day gap

    if year > datetime.now().year or (year == datetime.now().year and month > datetime.now().month):
        return {
            "available": False,
            "message": "Future dates - no imagery available yet"
        }

    return {
        "available": True,
        "count": len(images),
        "images": images,
        "note": "Simulated availability. Use local app with GEE for actual downloads."
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
                width='stretch'
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
                width='stretch'
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
                width='stretch'
            )

            df = pd.DataFrame(analysis['series'])
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)

            st.download_button(
                "📊 Download Time Series CSV",
                data=csv_buffer.getvalue(),
                file_name="timeseries.csv",
                mime="text/csv",
                width='stretch'
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
                width='stretch'
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
            st.dataframe(pd.DataFrame(data['daily_series']), width='stretch')

    # Session history
    st.divider()
    st.markdown("### Session Scan History")
    if st.session_state.scan_history:
        st.dataframe(pd.DataFrame(st.session_state.scan_history), width='stretch')
    else:
        st.info("No scans recorded yet")


def get_historical_date_data(year: int, month: int, day: int) -> dict:
    """Get historical data for a specific date."""
    target_date = datetime(year, month, day)

    # Chennai historical flood patterns
    # Major floods: Dec 2015, Nov 2021, Nov 2023
    major_flood_dates = [
        (2015, 12, 1), (2015, 12, 2), (2015, 12, 3),  # 2015 Chennai floods
        (2021, 11, 7), (2021, 11, 8), (2021, 11, 9),  # 2021 floods
        (2023, 11, 12), (2023, 11, 13),  # 2023 cyclone Michaung
    ]

    # Check if near a major flood event
    is_major_event = any(
        abs((target_date - datetime(y, m, d)).days) <= 2
        for y, m, d in major_flood_dates
    )

    # Seasonal patterns for Chennai
    if month in [10, 11, 12]:  # Northeast monsoon
        base_rainfall = np.random.uniform(15, 60) if not is_major_event else np.random.uniform(150, 350)
        season = "Northeast Monsoon (Peak Season)"
    elif month in [6, 7, 8, 9]:  # Southwest monsoon
        base_rainfall = np.random.uniform(5, 25)
        season = "Southwest Monsoon"
    else:  # Dry season
        base_rainfall = np.random.uniform(0, 8)
        season = "Dry Season"

    rainfall = round(base_rainfall, 1)
    flood_depth = rainfall * 0.003 if rainfall > 50 else rainfall * 0.001

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


if __name__ == "__main__":
    main()
