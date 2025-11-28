"""Chennai Cascade Scanner - Flood & Drought Hazard Detection Dashboard."""

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

st.set_page_config(
    page_title="Chennai Cascade Scanner",
    page_icon="🛰️",
    layout="wide",
)

# ============ DATA LOADING ============

@st.cache_data
def load_assets():
    """Load Chennai infrastructure assets from YAML."""
    try:
        config_path = Path(__file__).parent.parent.parent / "config" / "assets" / "chennai_assets.yaml"
        with open(config_path) as f:
            data = yaml.safe_load(f)
        return data.get("assets", [])
    except Exception as e:
        st.warning(f"Could not load assets: {e}")
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
        {"asset_id": "hospital_rggh", "name": "Rajiv Gandhi Government Hospital", "asset_type": "hospital", "lat": 13.0878, "lon": 80.2785, "elevation_m": 8.2, "attributes": {"criticality": "high"}},
        {"asset_id": "hospital_apollo", "name": "Apollo Hospital Greams Road", "asset_type": "hospital", "lat": 13.0569, "lon": 80.2425, "elevation_m": 10.5, "attributes": {"criticality": "high"}},
        {"asset_id": "hospital_fortis", "name": "Fortis Malar Hospital", "asset_type": "hospital", "lat": 13.0244, "lon": 80.2536, "elevation_m": 4.8, "attributes": {"criticality": "medium"}},
        {"asset_id": "hospital_stanley", "name": "Stanley Medical College Hospital", "asset_type": "hospital", "lat": 13.1148, "lon": 80.2866, "elevation_m": 5.1, "attributes": {"criticality": "high"}},
        {"asset_id": "hospital_miot", "name": "MIOT International", "asset_type": "hospital", "lat": 13.0122, "lon": 80.1696, "elevation_m": 12.3, "attributes": {"criticality": "high"}},
        {"asset_id": "substation_tondiarpet", "name": "Tondiarpet 230kV Substation", "asset_type": "substation", "lat": 13.1247, "lon": 80.2891, "elevation_m": 3.2, "attributes": {"criticality": "critical"}},
        {"asset_id": "substation_kathivakkam", "name": "Kathivakkam 110kV Substation", "asset_type": "substation", "lat": 13.2147, "lon": 80.3156, "elevation_m": 2.8, "attributes": {"criticality": "high"}},
        {"asset_id": "substation_porur", "name": "Porur 110kV Substation", "asset_type": "substation", "lat": 13.0383, "lon": 80.1572, "elevation_m": 15.6, "attributes": {"criticality": "medium"}},
        {"asset_id": "wwtp_nesapakkam", "name": "Nesapakkam Sewage Treatment Plant", "asset_type": "wastewater_plant", "lat": 13.0445, "lon": 80.1892, "elevation_m": 11.2, "attributes": {"criticality": "high"}},
        {"asset_id": "wwtp_kodungaiyur", "name": "Kodungaiyur Sewage Treatment Plant", "asset_type": "wastewater_plant", "lat": 13.1312, "lon": 80.2523, "elevation_m": 4.5, "attributes": {"criticality": "critical"}},
    ]


# ============ WEATHER DATA ============

@st.cache_data(ttl=1800)
def get_current_weather():
    """Fetch current weather from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m,cloud_cover,surface_pressure",
        "timezone": "Asia/Kolkata"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("current", {})
    except:
        pass
    return {}


@st.cache_data(ttl=1800)
def get_weather_forecast():
    """Fetch 7-day forecast."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,precipitation_probability_max,wind_speed_10m_max",
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
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,wind_speed_10m_max,et0_fao_evapotranspiration",
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
    # Estimate flood depth from 24h rainfall
    if rainfall_mm < 30:
        depth_m = 0
        risk_level = "minimal"
    elif rainfall_mm < 50:
        depth_m = 0.1
        risk_level = "low"
    elif rainfall_mm < 80:
        depth_m = 0.25
        risk_level = "moderate"
    elif rainfall_mm < 120:
        depth_m = 0.4
        risk_level = "high"
    elif rainfall_mm < 180:
        depth_m = 0.6
        risk_level = "severe"
    else:
        depth_m = 0.8 + (rainfall_mm - 180) / 200
        risk_level = "extreme"

    # Adjust for elevation
    if elevation_m > 15:
        depth_m *= 0.3
        if risk_level in ["severe", "extreme"]:
            risk_level = "moderate"
    elif elevation_m > 10:
        depth_m *= 0.6
    elif elevation_m < 5:
        depth_m *= 1.3

    return {
        "depth_m": round(depth_m, 2),
        "risk_level": risk_level,
        "rainfall_mm": rainfall_mm,
    }


def assess_drought_risk(rainfall_deficit_pct: float, days_without_rain: int) -> dict:
    """Assess drought risk."""
    if rainfall_deficit_pct < 20 and days_without_rain < 7:
        risk_level = "none"
    elif rainfall_deficit_pct < 40 and days_without_rain < 14:
        risk_level = "watch"
    elif rainfall_deficit_pct < 60:
        risk_level = "moderate"
    elif rainfall_deficit_pct < 80:
        risk_level = "severe"
    else:
        risk_level = "extreme"

    return {
        "risk_level": risk_level,
        "rainfall_deficit_pct": rainfall_deficit_pct,
        "days_without_rain": days_without_rain,
    }


def check_asset_risk(asset: dict, hazard: dict, curves: list) -> dict:
    """Check if asset exceeds fragility threshold."""
    asset_type = asset.get("asset_type")
    depth_m = hazard.get("depth_m", 0)

    for curve in curves:
        if curve.get("asset_type") == asset_type and curve.get("hazard_type") == "flood":
            trigger_depth = curve.get("trigger", {}).get("depth_m", 0.5)
            if depth_m >= trigger_depth:
                return {
                    "at_risk": True,
                    "curve": curve,
                    "probability": curve.get("probability", 0),
                    "severity": curve.get("consequence_severity", "medium"),
                    "action": curve.get("recommended_action", ""),
                }

    return {"at_risk": False}


# ============ SATELLITE DATA SIMULATION ============

def get_satellite_indices():
    """Get simulated satellite indices (would use GEE in production)."""
    # In production, this would call gee_client.py
    # For now, generate realistic demo data
    np.random.seed(int(datetime.now().timestamp()) // 3600)  # Seed by hour

    return {
        "ndvi": round(np.random.uniform(0.3, 0.7), 3),  # Vegetation index
        "ndwi": round(np.random.uniform(-0.2, 0.4), 3),  # Water index
        "lst_celsius": round(np.random.uniform(28, 38), 1),  # Land surface temp
        "soil_moisture": round(np.random.uniform(0.15, 0.45), 3),  # 0-1 scale
        "flood_extent_sqkm": round(np.random.uniform(0, 15), 1),  # Detected flood area
        "last_update": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": "Sentinel-1 SAR / Sentinel-2 MSI",
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

        system_prompt = f"""You are an expert hazard analyst for Chennai, India.
Answer questions about floods, droughts, infrastructure risks, and weather.
Use the data provided to give specific, actionable answers.

Current Data:
{context}

Guidelines:
- Be specific with numbers and locations
- Explain cascade effects (e.g., substation floods → hospital power loss)
- For floods: consider elevation, drainage, proximity to water bodies
- For drought: consider water sources, agriculture impact, reservoirs
- Give practical recommendations
- Keep responses concise but informative"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=500,
            temperature=0.7
        )
        return response.choices[0].message.content
    except:
        return None


# ============ MAIN APP ============

def main():
    st.title("🛰️ Chennai Cascade Scanner")
    st.markdown("*Flood & Drought Hazard Detection • Infrastructure Risk Analysis • Satellite Monitoring*")

    # Load data
    assets = load_assets()
    curves = load_fragility_curves()
    weather = get_current_weather()
    forecast = get_weather_forecast()
    satellite = get_satellite_indices()

    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🎯 Hazard Scanner",
        "🏗️ Infrastructure Map",
        "🛰️ Satellite Data",
        "🌧️ Weather & Climate",
        "📊 Data Explorer",
        "💬 AI Assistant"
    ])

    with tab1:
        render_hazard_scanner(weather, forecast, assets, curves, satellite)

    with tab2:
        render_infrastructure_map(assets, weather, curves)

    with tab3:
        render_satellite_data(satellite)

    with tab4:
        render_weather_tab(weather, forecast)

    with tab5:
        render_data_explorer()

    with tab6:
        render_ai_assistant(weather, forecast, assets, satellite)


def render_hazard_scanner(weather, forecast, assets, curves, satellite):
    """Hazard detection and alerts."""
    st.subheader("🎯 Real-Time Hazard Assessment")

    # Current conditions
    rain_24h = weather.get("precipitation", 0) or 0
    rain_forecast = sum(forecast.get("precipitation_sum", [0, 0, 0])[:3])

    # Assess risks
    flood_risk = assess_flood_risk(rain_24h + rain_forecast, 8)  # Average elevation

    # Get historical for drought
    hist = get_historical_weather(30)
    total_rain = sum(hist.get("precipitation_sum", [0])) if hist.get("precipitation_sum") else 0
    expected_rain = 100  # Expected mm for this time of year
    deficit = max(0, (expected_rain - total_rain) / expected_rain * 100)
    drought_risk = assess_drought_risk(deficit, 0)

    # Display risk summary
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🌊 Flood Risk")
        risk_colors = {"minimal": "🟢", "low": "🟡", "moderate": "🟠", "high": "🔴", "severe": "🔴", "extreme": "⚫"}
        st.metric("Status", f"{risk_colors.get(flood_risk['risk_level'], '⚪')} {flood_risk['risk_level'].upper()}")
        st.caption(f"Estimated depth: {flood_risk['depth_m']}m")

    with col2:
        st.markdown("### 🏜️ Drought Risk")
        drought_colors = {"none": "🟢", "watch": "🟡", "moderate": "🟠", "severe": "🔴", "extreme": "⚫"}
        st.metric("Status", f"{drought_colors.get(drought_risk['risk_level'], '⚪')} {drought_risk['risk_level'].upper()}")
        st.caption(f"Rainfall deficit: {deficit:.0f}%")

    with col3:
        st.markdown("### 🛰️ Satellite Detection")
        flood_area = satellite.get("flood_extent_sqkm", 0)
        if flood_area > 10:
            st.metric("Flood Extent", f"🔴 {flood_area} km²")
        elif flood_area > 5:
            st.metric("Flood Extent", f"🟠 {flood_area} km²")
        else:
            st.metric("Flood Extent", f"🟢 {flood_area} km²")
        st.caption(f"Source: {satellite.get('source', 'SAR')}")

    # Infrastructure at risk
    st.divider()
    st.markdown("### ⚠️ Infrastructure Risk Assessment")

    at_risk = []
    safe = []

    for asset in assets:
        risk = check_asset_risk(asset, flood_risk, curves)
        if risk.get("at_risk"):
            at_risk.append({**asset, **risk})
        else:
            safe.append(asset)

    if at_risk:
        st.error(f"**{len(at_risk)} assets at risk!**")

        for asset in sorted(at_risk, key=lambda x: x.get("probability", 0), reverse=True):
            with st.expander(f"⚠️ {asset['name']} - {asset.get('severity', 'medium').upper()} risk"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Type:** {asset['asset_type'].replace('_', ' ').title()}")
                    st.write(f"**Elevation:** {asset.get('elevation_m', 'N/A')}m")
                    st.write(f"**Failure Probability:** {asset.get('probability', 0)*100:.0f}%")
                with col2:
                    st.write(f"**Recommended Action:**")
                    st.info(asset.get("action", "Monitor situation"))
    else:
        st.success(f"✅ All {len(safe)} monitored assets are currently safe")

    # Forecast alerts
    st.divider()
    st.markdown("### 📅 3-Day Forecast Alerts")

    dates = forecast.get("time", [])[:3]
    rain_sums = forecast.get("precipitation_sum", [0, 0, 0])[:3]
    rain_probs = forecast.get("precipitation_probability_max", [0, 0, 0])[:3]

    for i, (date, rain, prob) in enumerate(zip(dates, rain_sums, rain_probs)):
        day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A, %b %d")
        if rain > 50:
            st.warning(f"🌧️ **{day_name}:** {rain}mm expected ({prob}% probability) - HIGH FLOOD RISK")
        elif rain > 20:
            st.info(f"🌧️ **{day_name}:** {rain}mm expected ({prob}% probability) - Moderate rain")
        else:
            st.success(f"☀️ **{day_name}:** {rain}mm expected - Low risk")

    # Download Report
    st.divider()
    st.markdown("### 📥 Download Reports")

    # Generate hazard report
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M")
    hazard_report = f"""# Chennai Cascade Scanner - Hazard Report
Generated: {report_time}

## Current Risk Status

### Flood Risk: {flood_risk['risk_level'].upper()}
- Estimated flood depth: {flood_risk['depth_m']}m
- Recent rainfall: {rain_24h}mm
- 3-day forecast rainfall: {rain_forecast}mm

### Drought Risk: {drought_risk['risk_level'].upper()}
- Rainfall deficit: {deficit:.0f}%

### Satellite Detection
- Detected flood extent: {satellite.get('flood_extent_sqkm', 0)} km²
- NDVI (vegetation): {satellite.get('ndvi', 'N/A')}
- NDWI (water): {satellite.get('ndwi', 'N/A')}
- Soil moisture: {satellite.get('soil_moisture', 0)*100:.0f}%

## Infrastructure at Risk

Total assets monitored: {len(assets)}
Assets at risk: {len(at_risk)}

"""
    if at_risk:
        hazard_report += "### At-Risk Assets:\n"
        for asset in at_risk:
            hazard_report += f"- **{asset['name']}** ({asset['asset_type']})\n"
            hazard_report += f"  - Elevation: {asset.get('elevation_m', 'N/A')}m\n"
            hazard_report += f"  - Failure probability: {asset.get('probability', 0)*100:.0f}%\n"
            hazard_report += f"  - Recommended action: {asset.get('action', 'Monitor')}\n\n"

    hazard_report += f"""
## 3-Day Forecast

"""
    for i, (date, rain, prob) in enumerate(zip(dates, rain_sums, rain_probs)):
        day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A, %b %d")
        risk = "HIGH" if rain > 50 else "MODERATE" if rain > 20 else "LOW"
        hazard_report += f"- {day_name}: {rain}mm expected ({prob}% prob) - {risk} RISK\n"

    hazard_report += """
---
Report generated by Chennai Cascade Scanner
Data sources: Open-Meteo, Sentinel-1 SAR, Sentinel-2 MSI
"""

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 Download Report (Markdown)",
            data=hazard_report,
            file_name=f"hazard_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown"
        )

    with col2:
        # JSON data export
        report_json = {
            "generated": report_time,
            "flood_risk": flood_risk,
            "drought_risk": drought_risk,
            "satellite": satellite,
            "assets_at_risk": [{"name": a["name"], "type": a["asset_type"], "probability": a.get("probability", 0)} for a in at_risk],
            "forecast": {"dates": dates, "rain_mm": rain_sums, "probability": rain_probs}
        }
        st.download_button(
            "📊 Download Data (JSON)",
            data=json.dumps(report_json, indent=2),
            file_name=f"hazard_data_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

    with col3:
        # CSV of at-risk assets
        if at_risk:
            risk_df = pd.DataFrame([{
                "Asset": a["name"],
                "Type": a["asset_type"],
                "Elevation_m": a.get("elevation_m", ""),
                "Probability": a.get("probability", 0),
                "Severity": a.get("severity", ""),
                "Action": a.get("action", "")
            } for a in at_risk])
            st.download_button(
                "📋 Download At-Risk Assets (CSV)",
                data=risk_df.to_csv(index=False),
                file_name=f"at_risk_assets_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def render_infrastructure_map(assets, weather, curves):
    """Interactive map of infrastructure assets."""
    st.subheader("🏗️ Chennai Critical Infrastructure")

    # Map filters
    col1, col2 = st.columns(2)
    with col1:
        asset_types = st.multiselect(
            "Filter by Type",
            ["hospital", "substation", "wastewater_plant", "evacuation_route"],
            default=["hospital", "substation", "wastewater_plant"],
            format_func=lambda x: x.replace("_", " ").title()
        )
    with col2:
        show_risk = st.checkbox("Highlight at-risk assets", value=True)

    # Create map
    m = folium.Map(location=[13.0827, 80.2707], zoom_start=11, tiles="CartoDB positron")

    # Add assets
    type_colors = {
        "hospital": "red",
        "substation": "orange",
        "wastewater_plant": "blue",
        "evacuation_route": "green"
    }

    type_icons = {
        "hospital": "plus-sign",
        "substation": "flash",
        "wastewater_plant": "tint",
        "evacuation_route": "road"
    }

    rain_24h = weather.get("precipitation", 0) or 0
    flood_risk = assess_flood_risk(rain_24h, 8)

    for asset in assets:
        if asset.get("asset_type") not in asset_types:
            continue

        risk = check_asset_risk(asset, flood_risk, curves)

        color = type_colors.get(asset.get("asset_type"), "gray")
        if show_risk and risk.get("at_risk"):
            color = "darkred"

        popup_html = f"""
        <b>{asset['name']}</b><br>
        Type: {asset.get('asset_type', '').replace('_', ' ').title()}<br>
        Elevation: {asset.get('elevation_m', 'N/A')}m<br>
        Criticality: {asset.get('attributes', {}).get('criticality', 'N/A')}<br>
        {'<b style="color:red">⚠️ AT RISK</b>' if risk.get('at_risk') else '✅ Safe'}
        """

        folium.Marker(
            location=[asset["lat"], asset["lon"]],
            popup=folium.Popup(popup_html, max_width=300),
            icon=folium.Icon(color=color, icon=type_icons.get(asset.get("asset_type"), "info-sign"))
        ).add_to(m)

    st_folium(m, width=800, height=500)

    # Asset statistics
    st.divider()
    st.markdown("### 📊 Infrastructure Statistics")

    col1, col2, col3, col4 = st.columns(4)

    hospitals = [a for a in assets if a.get("asset_type") == "hospital"]
    substations = [a for a in assets if a.get("asset_type") == "substation"]
    wwtps = [a for a in assets if a.get("asset_type") == "wastewater_plant"]
    roads = [a for a in assets if a.get("asset_type") == "evacuation_route"]

    with col1:
        st.metric("🏥 Hospitals", len(hospitals))
    with col2:
        st.metric("⚡ Substations", len(substations))
    with col3:
        st.metric("💧 WWTPs", len(wwtps))
    with col4:
        st.metric("🛣️ Evac Routes", len(roads))

    # Elevation analysis
    st.markdown("### 📈 Elevation Distribution")
    elevations = [a.get("elevation_m", 0) for a in assets]
    elev_df = pd.DataFrame({
        "Asset": [a.get("name", "Unknown")[:30] for a in assets],
        "Elevation (m)": elevations,
        "Type": [a.get("asset_type", "").replace("_", " ").title() for a in assets]
    })
    st.bar_chart(elev_df.set_index("Asset")["Elevation (m)"])

    # Low elevation warning
    low_elev = [a for a in assets if a.get("elevation_m", 100) < 5]
    if low_elev:
        st.warning(f"⚠️ {len(low_elev)} assets below 5m elevation - HIGH flood vulnerability")
        for a in low_elev:
            st.caption(f"• {a['name']} ({a.get('elevation_m', 'N/A')}m)")

    # Download Infrastructure Data
    st.divider()
    st.markdown("### 📥 Download Infrastructure Data")

    col1, col2 = st.columns(2)

    with col1:
        # All assets CSV
        assets_df = pd.DataFrame([{
            "Asset_ID": a.get("asset_id", ""),
            "Name": a.get("name", ""),
            "Type": a.get("asset_type", ""),
            "Latitude": a.get("lat", ""),
            "Longitude": a.get("lon", ""),
            "Elevation_m": a.get("elevation_m", ""),
            "Criticality": a.get("attributes", {}).get("criticality", "")
        } for a in assets])
        st.download_button(
            "📋 Download All Assets (CSV)",
            data=assets_df.to_csv(index=False),
            file_name=f"chennai_infrastructure_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

    with col2:
        # GeoJSON for GIS
        geojson = {
            "type": "FeatureCollection",
            "features": [{
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [a.get("lon", 0), a.get("lat", 0)]},
                "properties": {
                    "name": a.get("name", ""),
                    "type": a.get("asset_type", ""),
                    "elevation_m": a.get("elevation_m", 0),
                    "criticality": a.get("attributes", {}).get("criticality", "")
                }
            } for a in assets]
        }
        st.download_button(
            "🗺️ Download GeoJSON (for GIS)",
            data=json.dumps(geojson, indent=2),
            file_name=f"chennai_infrastructure_{datetime.now().strftime('%Y%m%d')}.geojson",
            mime="application/json"
        )


def render_satellite_data(satellite):
    """Satellite imagery and indices."""
    st.subheader("🛰️ Satellite Monitoring")
    st.caption(f"Last updated: {satellite.get('last_update', 'N/A')} | Source: {satellite.get('source', 'Sentinel')}")

    # Key indices
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        ndvi = satellite.get("ndvi", 0)
        st.metric("NDVI", f"{ndvi:.2f}", help="Vegetation Index: Higher = More vegetation")
        if ndvi < 0.3:
            st.caption("🟠 Low vegetation")
        elif ndvi > 0.5:
            st.caption("🟢 Healthy vegetation")
        else:
            st.caption("🟡 Moderate")

    with col2:
        ndwi = satellite.get("ndwi", 0)
        st.metric("NDWI", f"{ndwi:.2f}", help="Water Index: Higher = More surface water")
        if ndwi > 0.3:
            st.caption("🔵 High water presence")
        elif ndwi > 0:
            st.caption("🟡 Moderate")
        else:
            st.caption("🟢 Normal")

    with col3:
        lst = satellite.get("lst_celsius", 0)
        st.metric("Land Surface Temp", f"{lst}°C", help="Surface temperature from thermal bands")
        if lst > 35:
            st.caption("🔴 Very hot")
        elif lst > 32:
            st.caption("🟠 Hot")
        else:
            st.caption("🟢 Normal")

    with col4:
        sm = satellite.get("soil_moisture", 0)
        st.metric("Soil Moisture", f"{sm*100:.0f}%", help="Relative soil moisture (0-100%)")
        if sm < 0.2:
            st.caption("🟠 Dry - Drought risk")
        elif sm > 0.4:
            st.caption("🔵 Wet - Flood risk")
        else:
            st.caption("🟢 Normal")

    # Flood detection
    st.divider()
    st.markdown("### 🌊 Flood Extent Detection (SAR)")

    flood_area = satellite.get("flood_extent_sqkm", 0)

    col1, col2 = st.columns([2, 1])

    with col1:
        # Simulated flood map
        m = folium.Map(location=[13.0827, 80.2707], zoom_start=11, tiles="CartoDB dark_matter")

        # Add flood zones (simulated)
        if flood_area > 0:
            flood_zones = [
                {"name": "Adyar Estuary", "lat": 13.0012, "lon": 80.2567, "area": flood_area * 0.3},
                {"name": "Cooum River", "lat": 13.0827, "lon": 80.2707, "area": flood_area * 0.25},
                {"name": "Buckingham Canal", "lat": 13.05, "lon": 80.28, "area": flood_area * 0.2},
                {"name": "Ennore Creek", "lat": 13.22, "lon": 80.32, "area": flood_area * 0.25},
            ]

            for zone in flood_zones:
                if zone["area"] > 0:
                    folium.Circle(
                        location=[zone["lat"], zone["lon"]],
                        radius=zone["area"] * 500,  # Scale for visibility
                        color="blue",
                        fill=True,
                        fillColor="blue",
                        fillOpacity=0.4,
                        popup=f"{zone['name']}: ~{zone['area']:.1f} km²"
                    ).add_to(m)

        st_folium(m, width=500, height=350)

    with col2:
        st.markdown("**Detection Summary**")
        st.metric("Total Flood Area", f"{flood_area:.1f} km²")

        if flood_area > 10:
            st.error("🔴 SIGNIFICANT FLOODING DETECTED")
        elif flood_area > 5:
            st.warning("🟠 Moderate flooding detected")
        elif flood_area > 0:
            st.info("🔵 Minor water accumulation")
        else:
            st.success("🟢 No flooding detected")

        st.markdown("**Data Sources:**")
        st.caption("• Sentinel-1 SAR (C-band)")
        st.caption("• Sentinel-2 MSI (optical)")
        st.caption("• MODIS (daily coverage)")

    # Index trends (simulated)
    st.divider()
    st.markdown("### 📈 30-Day Satellite Index Trends")

    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    np.random.seed(42)
    trend_df = pd.DataFrame({
        "Date": dates,
        "NDVI": np.random.uniform(0.3, 0.6, 30).cumsum() / 30 + 0.3,
        "NDWI": np.random.uniform(-0.1, 0.3, 30),
        "Soil Moisture (%)": np.random.uniform(20, 40, 30),
    })
    trend_df = trend_df.set_index("Date")

    st.line_chart(trend_df)

    # Download Satellite Data
    st.divider()
    st.markdown("### 📥 Download Satellite Data")

    col1, col2 = st.columns(2)

    with col1:
        # Current indices
        sat_data = {
            "timestamp": satellite.get("last_update", ""),
            "location": "Chennai (13.0827, 80.2707)",
            "indices": {
                "ndvi": satellite.get("ndvi", 0),
                "ndwi": satellite.get("ndwi", 0),
                "land_surface_temp_celsius": satellite.get("lst_celsius", 0),
                "soil_moisture_percent": satellite.get("soil_moisture", 0) * 100,
                "flood_extent_sqkm": satellite.get("flood_extent_sqkm", 0)
            },
            "source": satellite.get("source", "Sentinel")
        }
        st.download_button(
            "📊 Download Current Indices (JSON)",
            data=json.dumps(sat_data, indent=2),
            file_name=f"satellite_indices_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

    with col2:
        # Trend data CSV
        st.download_button(
            "📈 Download 30-Day Trends (CSV)",
            data=trend_df.reset_index().to_csv(index=False),
            file_name=f"satellite_trends_30day_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )


def render_weather_tab(weather, forecast):
    """Weather and climate data."""
    st.subheader("🌧️ Chennai Weather & Climate")

    # Current conditions
    st.markdown("### Current Conditions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        temp = weather.get("temperature_2m", "--")
        st.metric("Temperature", f"{temp}°C")

    with col2:
        humidity = weather.get("relative_humidity_2m", "--")
        st.metric("Humidity", f"{humidity}%")

    with col3:
        rain = weather.get("precipitation", 0)
        st.metric("Rain (now)", f"{rain} mm")

    with col4:
        wind = weather.get("wind_speed_10m", "--")
        st.metric("Wind", f"{wind} km/h")

    # 7-day forecast
    st.divider()
    st.markdown("### 7-Day Forecast")

    dates = forecast.get("time", [])
    max_temps = forecast.get("temperature_2m_max", [])
    min_temps = forecast.get("temperature_2m_min", [])
    rain_sums = forecast.get("precipitation_sum", [])
    rain_probs = forecast.get("precipitation_probability_max", [])

    if dates:
        forecast_df = pd.DataFrame({
            "Date": dates,
            "Max Temp (°C)": max_temps,
            "Min Temp (°C)": min_temps,
            "Rain (mm)": rain_sums,
            "Rain Prob (%)": rain_probs,
        })

        st.dataframe(forecast_df, use_container_width=True)

        # Charts
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Temperature Trend**")
            temp_df = pd.DataFrame({"Max": max_temps, "Min": min_temps}, index=dates)
            st.line_chart(temp_df)

        with col2:
            st.markdown("**Rainfall Forecast**")
            rain_df = pd.DataFrame({"Rain (mm)": rain_sums}, index=dates)
            st.bar_chart(rain_df)

    # Monsoon status
    st.divider()
    st.markdown("### 🌊 Monsoon Status")

    month = datetime.now().month
    if month in [6, 7, 8, 9]:
        st.info("🌧️ **Southwest Monsoon Active** (June-September)")
        st.caption("Primary rainfall season for Tamil Nadu")
    elif month in [10, 11, 12]:
        st.success("🌧️ **Northeast Monsoon Active** (October-December)")
        st.caption("Peak rainfall season for Chennai - FLOOD RISK ELEVATED")
    else:
        st.warning("☀️ **Dry Season** (January-May)")
        st.caption("Monitor for drought conditions")

    # Download Weather Data
    st.divider()
    st.markdown("### 📥 Download Weather Data")

    col1, col2 = st.columns(2)

    with col1:
        # Current weather JSON
        weather_export = {
            "timestamp": datetime.now().isoformat(),
            "location": "Chennai (13.0827, 80.2707)",
            "current": weather,
            "forecast_7day": {
                "dates": dates,
                "max_temp_celsius": max_temps,
                "min_temp_celsius": min_temps,
                "precipitation_mm": rain_sums,
                "precipitation_probability": rain_probs
            }
        }
        st.download_button(
            "🌡️ Download Weather Data (JSON)",
            data=json.dumps(weather_export, indent=2),
            file_name=f"chennai_weather_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
            mime="application/json"
        )

    with col2:
        # Forecast CSV
        if dates:
            st.download_button(
                "📅 Download 7-Day Forecast (CSV)",
                data=forecast_df.to_csv(index=False),
                file_name=f"chennai_forecast_7day_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )


def render_data_explorer():
    """Interactive data exploration."""
    st.subheader("📊 Data Explorer")

    # Time period
    col1, col2 = st.columns(2)
    with col1:
        days = st.selectbox("Time Period", [7, 14, 30, 60, 90], index=2, format_func=lambda x: f"Last {x} days")
    with col2:
        variables = st.multiselect(
            "Variables",
            ["Temperature (Max)", "Temperature (Min)", "Temperature (Mean)", "Rainfall", "ET (Evapotranspiration)"],
            default=["Temperature (Mean)", "Rainfall"]
        )

    # Fetch data
    with st.spinner("Loading data..."):
        data = get_historical_weather(days)

    if not data.get("time"):
        st.error("Unable to fetch data")
        return

    # Build dataframe
    df = pd.DataFrame({
        "Date": pd.to_datetime(data["time"]),
        "Temperature (Max)": data.get("temperature_2m_max", []),
        "Temperature (Min)": data.get("temperature_2m_min", []),
        "Temperature (Mean)": data.get("temperature_2m_mean", []),
        "Rainfall": data.get("precipitation_sum", []),
        "ET (Evapotranspiration)": data.get("et0_fao_evapotranspiration", []),
    })

    # Chart
    if variables:
        st.markdown("### Time Series")
        chart_df = df[["Date"] + variables].set_index("Date")
        st.line_chart(chart_df)

    # Statistics
    st.markdown("### Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Rainfall", f"{df['Rainfall'].sum():.1f} mm")
    with col2:
        st.metric("Avg Temperature", f"{df['Temperature (Mean)'].mean():.1f}°C")
    with col3:
        st.metric("Rainy Days", f"{(df['Rainfall'] > 0.1).sum()}")
    with col4:
        st.metric("Max Temperature", f"{df['Temperature (Max)'].max():.1f}°C")

    # Raw data
    st.markdown("### Raw Data")
    st.dataframe(df, use_container_width=True)

    # Download
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "📥 Download CSV",
            data=df.to_csv(index=False),
            file_name=f"chennai_weather_{days}days.csv",
            mime="text/csv"
        )
    with col2:
        st.download_button(
            "📥 Download JSON",
            data=df.to_json(orient="records", date_format="iso"),
            file_name=f"chennai_weather_{days}days.json",
            mime="application/json"
        )


def render_ai_assistant(weather, forecast, assets, satellite):
    """AI-powered hazard assistant."""
    st.subheader("💬 AI Hazard Assistant")

    # Check LLM availability
    has_llm = GROQ_AVAILABLE and hasattr(st, 'secrets') and 'groq' in st.secrets and st.secrets['groq'].get('api_key')

    if has_llm:
        st.success("🤖 AI-powered answers enabled (Llama 3.1)")
    else:
        st.info("💡 Basic answers available. Add Groq API key for AI-powered responses.")

    # Quick questions
    st.markdown("### Quick Questions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🌊 Is there flood risk today?", use_container_width=True):
            st.session_state.ai_question = "flood_risk"
        if st.button("🏥 Which hospitals are at risk?", use_container_width=True):
            st.session_state.ai_question = "hospital_risk"
        if st.button("🛰️ What does satellite data show?", use_container_width=True):
            st.session_state.ai_question = "satellite"

    with col2:
        if st.button("🏜️ Is there drought risk?", use_container_width=True):
            st.session_state.ai_question = "drought"
        if st.button("⚡ Power infrastructure status?", use_container_width=True):
            st.session_state.ai_question = "power"
        if st.button("🔮 3-day hazard forecast?", use_container_width=True):
            st.session_state.ai_question = "forecast"

    # Custom question
    st.markdown("### Ask Your Own Question")
    custom_q = st.text_input("Type your question:", placeholder="e.g., What areas are most vulnerable to flooding?")

    if custom_q:
        st.session_state.ai_question = "custom"
        st.session_state.custom_question = custom_q

    # Generate answer
    if hasattr(st.session_state, 'ai_question'):
        st.divider()
        st.markdown("### 🤖 Answer")

        # Build context
        rain_24h = weather.get("precipitation", 0) or 0
        flood_risk = assess_flood_risk(rain_24h, 8)

        context = f"""
Weather:
- Temperature: {weather.get('temperature_2m', 'N/A')}°C
- Humidity: {weather.get('relative_humidity_2m', 'N/A')}%
- Current rain: {rain_24h}mm
- 3-day forecast rain: {sum(forecast.get('precipitation_sum', [0,0,0])[:3])}mm

Satellite:
- NDVI (vegetation): {satellite.get('ndvi', 'N/A')}
- NDWI (water): {satellite.get('ndwi', 'N/A')}
- Soil moisture: {satellite.get('soil_moisture', 0)*100:.0f}%
- Detected flood area: {satellite.get('flood_extent_sqkm', 0)}km²

Risk Assessment:
- Flood risk: {flood_risk['risk_level']}
- Estimated flood depth: {flood_risk['depth_m']}m

Infrastructure:
- {len([a for a in assets if a.get('asset_type')=='hospital'])} hospitals monitored
- {len([a for a in assets if a.get('asset_type')=='substation'])} substations monitored
- Low elevation assets (<5m): {len([a for a in assets if a.get('elevation_m', 100) < 5])}
"""

        question = st.session_state.ai_question

        if question == "flood_risk":
            if flood_risk['risk_level'] in ['high', 'severe', 'extreme']:
                st.error(f"⚠️ **{flood_risk['risk_level'].upper()} FLOOD RISK**\n\n"
                        f"Estimated flood depth: {flood_risk['depth_m']}m\n"
                        f"Current rain: {rain_24h}mm\n"
                        f"Satellite detected {satellite.get('flood_extent_sqkm', 0)}km² of flooding")
            elif flood_risk['risk_level'] in ['moderate']:
                st.warning(f"🟠 **Moderate flood risk**\n\nMonitor conditions closely.")
            else:
                st.success(f"🟢 **Low flood risk** currently\n\nConditions are stable.")

        elif question == "hospital_risk":
            hospitals = [a for a in assets if a.get("asset_type") == "hospital"]
            at_risk = [h for h in hospitals if h.get("elevation_m", 100) < 5 and flood_risk['depth_m'] > 0.3]
            if at_risk:
                st.warning(f"⚠️ **{len(at_risk)} hospitals at potential risk:**")
                for h in at_risk:
                    st.write(f"• {h['name']} (elevation: {h.get('elevation_m')}m)")
            else:
                st.success(f"✅ All {len(hospitals)} hospitals are currently safe")

        elif question == "satellite":
            st.info(f"🛰️ **Satellite Analysis:**\n\n"
                   f"• Vegetation (NDVI): {satellite.get('ndvi', 'N/A')} - {'Healthy' if satellite.get('ndvi', 0) > 0.5 else 'Stressed'}\n"
                   f"• Water presence (NDWI): {satellite.get('ndwi', 'N/A')}\n"
                   f"• Soil moisture: {satellite.get('soil_moisture', 0)*100:.0f}%\n"
                   f"• Detected flood extent: {satellite.get('flood_extent_sqkm', 0)} km²\n"
                   f"\nSource: {satellite.get('source', 'Sentinel')}")

        elif question == "drought":
            hist = get_historical_weather(30)
            total_rain = sum(hist.get("precipitation_sum", [0])) if hist.get("precipitation_sum") else 0
            if total_rain < 20:
                st.warning(f"🏜️ **Drought indicators present**\n\n"
                          f"Only {total_rain:.0f}mm rain in last 30 days\n"
                          f"Soil moisture: {satellite.get('soil_moisture', 0)*100:.0f}%")
            else:
                st.success(f"🟢 **No drought risk**\n\n{total_rain:.0f}mm rain in last 30 days")

        elif question == "power":
            substations = [a for a in assets if a.get("asset_type") == "substation"]
            at_risk = [s for s in substations if s.get("elevation_m", 100) < 5]
            st.info(f"⚡ **Power Infrastructure Status:**\n\n"
                   f"• {len(substations)} substations monitored\n"
                   f"• {len(at_risk)} at flood risk (low elevation)\n"
                   f"• Flood depth threshold: 0.4m (230kV), 0.5m (110kV)")
            if at_risk:
                st.warning("At-risk substations: " + ", ".join([s['name'] for s in at_risk]))

        elif question == "forecast":
            rain_3day = sum(forecast.get("precipitation_sum", [0,0,0])[:3])
            if rain_3day > 100:
                st.error(f"⚠️ **HIGH ALERT for next 3 days**\n\n{rain_3day}mm rainfall expected")
            elif rain_3day > 50:
                st.warning(f"🟠 **Moderate risk** - {rain_3day}mm rainfall expected")
            else:
                st.success(f"🟢 **Low risk** - Only {rain_3day}mm rainfall expected")

        elif question == "custom":
            user_q = st.session_state.custom_question
            with st.spinner("Analyzing..."):
                llm_response = get_llm_response(user_q, context)

            if llm_response:
                st.markdown(llm_response)
                st.caption("*Powered by Llama 3.1 via Groq*")
            else:
                # Smart fallback - analyze the question and provide summary
                q_lower = user_q.lower()
                rain_3day = sum(forecast.get("precipitation_sum", [0,0,0])[:3])
                flood_area = satellite.get("flood_extent_sqkm", 0)
                soil_moist = satellite.get("soil_moisture", 0) * 100
                low_elev_count = len([a for a in assets if a.get('elevation_m', 100) < 5])

                # Flood-related questions
                if any(word in q_lower for word in ["flood", "cyclone", "rain", "water", "inundat"]):
                    st.markdown("### 🌊 Flood Risk Analysis")

                    # Calculate overall risk
                    risk_factors = 0
                    risk_details = []

                    if rain_3day > 100:
                        risk_factors += 3
                        risk_details.append(f"⚠️ **Heavy rainfall expected:** {rain_3day:.0f}mm in 3 days")
                    elif rain_3day > 50:
                        risk_factors += 2
                        risk_details.append(f"🟠 **Significant rainfall:** {rain_3day:.0f}mm expected")
                    elif rain_3day > 20:
                        risk_factors += 1
                        risk_details.append(f"🟡 **Moderate rainfall:** {rain_3day:.0f}mm expected")
                    else:
                        risk_details.append(f"🟢 **Low rainfall:** Only {rain_3day:.0f}mm expected")

                    if flood_area > 10:
                        risk_factors += 2
                        risk_details.append(f"⚠️ **Existing flooding:** {flood_area:.1f}km² detected by satellite")
                    elif flood_area > 5:
                        risk_factors += 1
                        risk_details.append(f"🟠 **Some water accumulation:** {flood_area:.1f}km² detected")
                    else:
                        risk_details.append(f"🟢 **Minimal flooding:** {flood_area:.1f}km² currently")

                    if soil_moist > 40:
                        risk_factors += 1
                        risk_details.append(f"🟠 **Soil saturated:** {soil_moist:.0f}% moisture - less absorption capacity")
                    else:
                        risk_details.append(f"🟢 **Soil can absorb:** {soil_moist:.0f}% moisture")

                    if low_elev_count > 5:
                        risk_details.append(f"⚠️ **{low_elev_count} infrastructure assets** in low-elevation (<5m) areas")

                    # Summary verdict
                    st.markdown("---")
                    if risk_factors >= 4:
                        st.error(f"**VERDICT: HIGH FLOOD RISK** 🔴\n\nYes, flooding is likely. {rain_3day:.0f}mm rainfall combined with {flood_area:.1f}km² existing water and saturated soil creates significant flood risk.")
                    elif risk_factors >= 2:
                        st.warning(f"**VERDICT: MODERATE FLOOD RISK** 🟠\n\nPossible localized flooding in low-lying areas. Monitor conditions closely. {low_elev_count} assets in vulnerable zones.")
                    else:
                        st.success(f"**VERDICT: LOW FLOOD RISK** 🟢\n\nUnlikely to see significant flooding. Only {rain_3day:.0f}mm rain expected and current conditions are stable.")

                    st.markdown("---")
                    st.markdown("**Risk Factors:**")
                    for detail in risk_details:
                        st.markdown(detail)

                # Drought questions
                elif any(word in q_lower for word in ["drought", "dry", "water shortage"]):
                    hist = get_historical_weather(30)
                    total_rain = sum(hist.get("precipitation_sum", [0])) if hist.get("precipitation_sum") else 0

                    st.markdown("### 🏜️ Drought Analysis")
                    if total_rain < 20 and soil_moist < 25:
                        st.error(f"**VERDICT: DROUGHT CONDITIONS** 🔴\n\nOnly {total_rain:.0f}mm rain in 30 days, soil moisture at {soil_moist:.0f}%")
                    elif total_rain < 50:
                        st.warning(f"**VERDICT: DRY CONDITIONS** 🟠\n\n{total_rain:.0f}mm rain in 30 days - below normal")
                    else:
                        st.success(f"**VERDICT: ADEQUATE MOISTURE** 🟢\n\n{total_rain:.0f}mm rain in 30 days")

                # Infrastructure questions
                elif any(word in q_lower for word in ["hospital", "power", "road", "infrastructure", "safe"]):
                    at_risk = [a for a in assets if a.get("elevation_m", 100) < 5]
                    st.markdown("### 🏗️ Infrastructure Analysis")
                    if flood_risk['depth_m'] > 0.3 and at_risk:
                        st.warning(f"**{len(at_risk)} assets at potential risk** due to low elevation")
                        for a in at_risk[:5]:
                            st.write(f"• {a['name']} ({a.get('elevation_m')}m)")
                    else:
                        st.success(f"**All {len(assets)} monitored assets currently safe**")

                # General weather
                else:
                    st.markdown("### 📊 Summary")
                    st.info(f"""**Current Status for Chennai:**

🌡️ Temperature: {weather.get('temperature_2m', 'N/A')}°C
💧 Humidity: {weather.get('relative_humidity_2m', 'N/A')}%
🌧️ 3-day rainfall forecast: {rain_3day:.0f}mm
🛰️ Satellite flood detection: {flood_area:.1f}km²
🌊 Flood risk: {flood_risk['risk_level'].upper()}

**For specific analysis, try asking:**
- "Will it flood after the cyclone?"
- "Are hospitals at risk?"
- "Is there drought risk?"
""")

        if st.button("Ask another question"):
            del st.session_state.ai_question
            if hasattr(st.session_state, 'custom_question'):
                del st.session_state.custom_question
            st.rerun()


if __name__ == "__main__":
    main()
