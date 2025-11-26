"""Enhanced Streamlit dashboard for Cascade Scanner MVP - Local Version."""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timezone, timedelta
import json
import pandas as pd
import io

st.set_page_config(
    page_title="Cascade Scanner",
    page_icon="🌊",
    layout="wide",
)

# Import local modules
try:
    from src.core import CascadeScanner, query_parser, format_output
    from src.graph import get_connection
    from src.ml import anomaly_detector, temporal_analyzer
    FULL_MODE = True
except Exception as e:
    st.error(f"Failed to import modules: {e}")
    FULL_MODE = False

# ============ INITIALIZE ============
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "scanner" not in st.session_state and FULL_MODE:
    try:
        st.session_state.scanner = CascadeScanner()
    except Exception as e:
        st.warning(f"Scanner init failed: {e}")
        FULL_MODE = False


def main():
    st.title("🌊 Cascade Scanner")
    st.markdown("*Flood-cascade risk assessment for Chennai with temporal analysis*")

    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs([
        "🗺️ Live Scanner",
        "📈 Temporal Analysis",
        "📥 Download Center",
        "📊 Historical Data"
    ])

    with tab1:
        render_scanner_tab()

    with tab2:
        render_temporal_tab()

    with tab3:
        render_download_tab()

    with tab4:
        render_history_tab()


def render_scanner_tab():
    """Main scanner interface."""
    # Sidebar for scanner
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

    # Main content
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📍 Chennai Infrastructure Map")
        render_map()

    with col2:
        st.subheader("📊 Current Status")
        render_status()

    # Results
    if st.session_state.last_result:
        render_results()


def run_scan(query: str, time_mode: str, stakeholder: str):
    """Execute scan and update state."""
    with st.spinner("Scanning Chennai for flood risks..."):
        try:
            if FULL_MODE and hasattr(st.session_state, 'scanner'):
                if query:
                    parsed = query_parser.parse(query)
                    time_mode = parsed.time_mode
                    stakeholder = parsed.stakeholder

                result = st.session_state.scanner.execute_scan(
                    location="chennai",
                    time_mode=time_mode,
                    stakeholder=stakeholder,
                )
                st.session_state.last_result = result

                # Record for temporal analysis
                temporal_analyzer.record_scan(result)

                # Check for anomalies
                anomaly = anomaly_detector.detect(result)
                st.session_state.anomaly = anomaly

                # Add to history
                st.session_state.scan_history.append({
                    "timestamp": datetime.now().isoformat(),
                    "event_id": result.event_id,
                    "alerts": len(result.alerts),
                    "risk_level": result.hazard_state.flood_risk_level
                })

                st.success(f"Scan complete: {len(result.alerts)} alerts")
            else:
                st.error("Scanner not available")
        except Exception as e:
            st.error(f"Scan failed: {e}")


def render_map():
    """Render Folium map with assets."""
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

    # Load assets from Neo4j
    assets = []
    if FULL_MODE:
        try:
            conn = get_connection()
            result = conn.execute_query("MATCH (a:Asset) RETURN a")
            assets = [dict(r["a"]) for r in result]
        except Exception as e:
            st.warning(f"Could not load assets: {e}")

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

    # Add alerts if available
    result = st.session_state.last_result
    if result and hasattr(result, 'alerts') and result.alerts:
        for alert in result.alerts[:10]:
            for a in assets:
                if a.get("asset_id") == alert.asset_id:
                    folium.Marker(
                        location=[a.get("lat", 0), a.get("lon", 0)],
                        popup=f"⚠️ {alert.asset_name}<br>Risk: {alert.risk_score:.0f}%",
                        icon=folium.Icon(color="red", icon="exclamation-sign"),
                    ).add_to(m)
                    break

    # Legend
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
    """Render status metrics."""
    result = st.session_state.last_result

    if result:
        hazard = result.hazard_state

        col1, col2 = st.columns(2)
        with col1:
            st.metric("24h Rainfall", f"{hazard.rainfall_24h_mm:.1f} mm")
            st.metric("Est. Depth", f"{hazard.depth_m:.2f} m")
        with col2:
            st.metric("Risk Level", hazard.flood_risk_level.upper())
            st.metric("Alerts", len(result.alerts) if hasattr(result, 'alerts') else 0)

        # Anomaly indicator
        if hasattr(st.session_state, "anomaly"):
            anomaly = st.session_state.anomaly
            if anomaly.is_anomaly:
                st.warning(f"⚠️ Anomaly: {anomaly.explanation}")

        st.caption(f"Event: {result.event_id}")
        st.caption(f"Scan time: {result.scan_duration_seconds:.2f}s")
    else:
        st.info("👆 Click **Run Scan** to check current conditions")
        st.metric("Current Time", datetime.now().strftime("%H:%M IST"))
        st.metric("Location", "Chennai, India")


def render_results():
    """Render scan results."""
    result = st.session_state.last_result

    st.divider()
    st.subheader("📋 Scan Results")

    tab1, tab2, tab3 = st.tabs(["📝 Action List", "📊 Raw Data", "📋 Alerts Table"])

    with tab1:
        output = format_output(result, "emergency_manager") if FULL_MODE else "No output available"
        st.markdown(output)

    with tab2:
        data = format_output(result, "researcher") if FULL_MODE else "{}"
        try:
            st.json(json.loads(data))
        except:
            st.code(data)

    with tab3:
        if hasattr(result, 'alerts') and result.alerts:
            df = pd.DataFrame([
                {
                    "Asset": a.asset_name,
                    "Risk %": f"{a.risk_score:.0f}",
                    "Type": a.cascade_type,
                    "Severity": a.consequence_severity,
                    "ETA (h)": f"{a.eta_failure_hours:.1f}",
                    "Action": a.recommended_action,
                }
                for a in result.alerts[:20]
            ])
            st.dataframe(df, use_container_width=True)
        else:
            st.success("✅ No active alerts - all infrastructure safe")


def render_temporal_tab():
    """Temporal analysis interface."""
    st.subheader("📈 Temporal Analysis")
    st.markdown("Analyze flood risk trends over time")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Analysis Settings")
        period = st.selectbox(
            "Analysis Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "Last 365 days"],
            index=1
        )

        days_map = {
            "Last 7 days": 7,
            "Last 30 days": 30,
            "Last 90 days": 90,
            "Last 365 days": 365
        }
        days_back = days_map[period]

        if st.button("🔄 Analyze Trends", type="primary"):
            with st.spinner("Analyzing temporal patterns..."):
                analysis = temporal_analyzer.analyze_period(days_back)
                st.session_state.temporal_analysis = analysis

    with col2:
        if hasattr(st.session_state, 'temporal_analysis'):
            analysis = st.session_state.temporal_analysis
            st.markdown("### Trend Summary")

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Rainfall", f"{analysis.total_rainfall_mm:.1f} mm")
            with m2:
                st.metric("Avg Daily", f"{analysis.avg_daily_rainfall_mm:.1f} mm")
            with m3:
                st.metric("High Risk Days", analysis.high_risk_days)
            with m4:
                st.metric("Total Alerts", analysis.total_alerts)

            # Trend indicators
            st.markdown("### Trend Indicators")
            col_a, col_b = st.columns(2)
            with col_a:
                trend_emoji = {"increasing": "📈", "decreasing": "📉", "stable": "➡️"}.get(analysis.rainfall_trend, "❓")
                st.info(f"**Rainfall Trend:** {trend_emoji} {analysis.rainfall_trend.title()}")
            with col_b:
                risk_emoji = {"elevated": "🔴", "moderate": "🟡", "stable": "🟢"}.get(analysis.risk_trend, "⚪")
                st.info(f"**Risk Trend:** {risk_emoji} {analysis.risk_trend.title()}")

            # Monsoon info
            if analysis.monsoon_intensity:
                st.markdown(f"**Monsoon Intensity:** {analysis.monsoon_intensity.replace('_', ' ').title()}")
            if analysis.peak_risk_month:
                st.markdown(f"**Peak Risk Month:** {analysis.peak_risk_month}")

            # Most affected assets
            if analysis.most_affected_assets:
                st.markdown("### Most Affected Assets")
                for asset in analysis.most_affected_assets[:5]:
                    st.markdown(f"- {asset}")

    # Time series visualization
    st.divider()
    st.markdown("### Historical Time Series")

    series_data = temporal_analyzer.get_historical_series(days_back=30)
    if series_data:
        df = pd.DataFrame(series_data)
        df['date'] = pd.to_datetime(df['date'])

        # Create line chart
        st.line_chart(df.set_index('date')['value'], use_container_width=True)

        # Risk level distribution
        st.markdown("### Risk Level Distribution")
        risk_counts = df['risk_level'].value_counts()
        st.bar_chart(risk_counts)


def render_download_tab():
    """Download center for exporting data."""
    st.subheader("📥 Download Center")
    st.markdown("Export scan results and analysis data")

    result = st.session_state.last_result

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Current Scan Results")

        if result:
            # JSON export
            json_data = {
                "event_id": result.event_id,
                "timestamp": result.timestamp.isoformat() if hasattr(result, 'timestamp') else datetime.now().isoformat(),
                "location": result.location,
                "time_mode": result.time_mode,
                "hazard_state": {
                    "type": result.hazard_state.hazard_type,
                    "depth_m": result.hazard_state.depth_m,
                    "rainfall_24h_mm": result.hazard_state.rainfall_24h_mm,
                    "risk_level": result.hazard_state.flood_risk_level,
                },
                "alerts_count": len(result.alerts) if hasattr(result, 'alerts') else 0,
                "safe_assets_count": len(result.safe_assets) if hasattr(result, 'safe_assets') else 0,
            }

            st.download_button(
                "📄 Download JSON",
                data=json.dumps(json_data, indent=2),
                file_name=f"cascade_scan_{result.event_id}.json",
                mime="application/json",
                use_container_width=True
            )

            # CSV export for alerts
            if hasattr(result, 'alerts') and result.alerts:
                alerts_df = pd.DataFrame([
                    {
                        "asset_id": a.asset_id,
                        "asset_name": a.asset_name,
                        "risk_score": a.risk_score,
                        "cascade_type": a.cascade_type,
                        "consequence_severity": a.consequence_severity,
                        "eta_failure_hours": a.eta_failure_hours,
                        "recommended_action": a.recommended_action,
                    }
                    for a in result.alerts
                ])

                csv_buffer = io.StringIO()
                alerts_df.to_csv(csv_buffer, index=False)

                st.download_button(
                    "📊 Download Alerts CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"cascade_alerts_{result.event_id}.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            # Markdown report
            report = f"""# Cascade Scanner Report
## Event: {result.event_id}
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Hazard Conditions
- **Location:** Chennai, India
- **24h Rainfall:** {result.hazard_state.rainfall_24h_mm:.1f} mm
- **Flood Depth:** {result.hazard_state.depth_m:.2f} m
- **Risk Level:** {result.hazard_state.flood_risk_level.upper()}

## Summary
{result.summary}

## Alerts
Total: {len(result.alerts) if hasattr(result, 'alerts') else 0}

---
*Generated by Cascade Scanner MVP*
"""
            st.download_button(
                "📝 Download Report (MD)",
                data=report,
                file_name=f"cascade_report_{result.event_id}.md",
                mime="text/markdown",
                use_container_width=True
            )

        else:
            st.info("Run a scan first to enable downloads")

    with col2:
        st.markdown("### Temporal Analysis Data")

        if hasattr(st.session_state, 'temporal_analysis'):
            analysis = st.session_state.temporal_analysis

            # Export temporal analysis
            temporal_json = {
                "period": {
                    "start": analysis.period_start.isoformat(),
                    "end": analysis.period_end.isoformat(),
                    "days": analysis.total_days
                },
                "rainfall": {
                    "total_mm": analysis.total_rainfall_mm,
                    "avg_daily_mm": analysis.avg_daily_rainfall_mm,
                    "max_daily_mm": analysis.max_daily_rainfall_mm,
                    "rainy_days": analysis.rainy_days
                },
                "risk": {
                    "high_risk_days": analysis.high_risk_days,
                    "moderate_risk_days": analysis.moderate_risk_days,
                    "low_risk_days": analysis.low_risk_days,
                    "total_alerts": analysis.total_alerts
                },
                "trends": {
                    "rainfall_trend": analysis.rainfall_trend,
                    "risk_trend": analysis.risk_trend,
                    "monsoon_intensity": analysis.monsoon_intensity
                },
                "most_affected_assets": analysis.most_affected_assets
            }

            st.download_button(
                "📈 Download Temporal Analysis JSON",
                data=json.dumps(temporal_json, indent=2),
                file_name=f"temporal_analysis_{analysis.period_end.strftime('%Y%m%d')}.json",
                mime="application/json",
                use_container_width=True
            )

            # Time series CSV
            series_data = temporal_analyzer.get_historical_series(days_back=analysis.total_days)
            if series_data:
                series_df = pd.DataFrame(series_data)
                csv_buffer = io.StringIO()
                series_df.to_csv(csv_buffer, index=False)

                st.download_button(
                    "📊 Download Time Series CSV",
                    data=csv_buffer.getvalue(),
                    file_name=f"timeseries_{analysis.total_days}days.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        else:
            st.info("Run temporal analysis first to enable downloads")

        st.divider()
        st.markdown("### Bulk Data Export")

        # Export all scan history
        if st.session_state.scan_history:
            history_json = json.dumps(st.session_state.scan_history, indent=2)
            st.download_button(
                "📜 Download Scan History",
                data=history_json,
                file_name="scan_history.json",
                mime="application/json",
                use_container_width=True
            )


def render_history_tab():
    """Historical data and comparison view."""
    st.subheader("📊 Historical Data")
    st.markdown("View and compare historical flood events")

    # Date range selector
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=30)
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now()
        )

    if st.button("📊 Load Historical Data"):
        # Get historical data
        days = (end_date - start_date).days
        series = temporal_analyzer.get_historical_series(days_back=days)

        if series:
            df = pd.DataFrame(series)
            df['date'] = pd.to_datetime(df['date'])
            st.session_state.historical_df = df

    # Display historical data
    if hasattr(st.session_state, 'historical_df'):
        df = st.session_state.historical_df

        st.markdown("### Rainfall Time Series")
        st.line_chart(df.set_index('date')['value'])

        st.markdown("### Data Table")
        st.dataframe(df, use_container_width=True)

        # Statistics
        st.markdown("### Period Statistics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rainfall", f"{df['value'].sum():.1f} mm")
        with col2:
            st.metric("Average", f"{df['value'].mean():.1f} mm")
        with col3:
            st.metric("Maximum", f"{df['value'].max():.1f} mm")
        with col4:
            rainy = len(df[df['value'] > 0.1])
            st.metric("Rainy Days", rainy)

    # Period comparison
    st.divider()
    st.markdown("### Compare Two Periods")

    comp_col1, comp_col2 = st.columns(2)
    with comp_col1:
        st.markdown("**Period 1**")
        p1_start = st.date_input("P1 Start", key="p1_start", value=datetime.now() - timedelta(days=60))
        p1_end = st.date_input("P1 End", key="p1_end", value=datetime.now() - timedelta(days=30))

    with comp_col2:
        st.markdown("**Period 2**")
        p2_start = st.date_input("P2 Start", key="p2_start", value=datetime.now() - timedelta(days=30))
        p2_end = st.date_input("P2 End", key="p2_end", value=datetime.now())

    if st.button("🔄 Compare Periods"):
        comparison = temporal_analyzer.compare_periods(
            datetime.combine(p1_start, datetime.min.time()),
            datetime.combine(p1_end, datetime.min.time()),
            datetime.combine(p2_start, datetime.min.time()),
            datetime.combine(p2_end, datetime.min.time())
        )

        st.markdown("### Comparison Results")

        comp_df = pd.DataFrame([
            {
                "Metric": "Total Rainfall (mm)",
                "Period 1": f"{comparison['period1']['total_rainfall_mm']:.1f}",
                "Period 2": f"{comparison['period2']['total_rainfall_mm']:.1f}",
                "Change": f"{comparison['comparison']['rainfall_change_pct']:.1f}%"
            },
            {
                "Metric": "High Risk Days",
                "Period 1": comparison['period1']['high_risk_days'],
                "Period 2": comparison['period2']['high_risk_days'],
                "Change": comparison['comparison']['risk_change']
            },
            {
                "Metric": "Total Alerts",
                "Period 1": comparison['period1']['total_alerts'],
                "Period 2": comparison['period2']['total_alerts'],
                "Change": comparison['comparison']['alerts_change']
            }
        ])

        st.dataframe(comp_df, use_container_width=True)

    # Session history
    st.divider()
    st.markdown("### Current Session Scan History")

    if st.session_state.scan_history:
        history_df = pd.DataFrame(st.session_state.scan_history)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("No scans recorded in this session yet")


if __name__ == "__main__":
    main()
