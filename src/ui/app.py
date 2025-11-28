"""Chennai Weather & Environment Dashboard - Simple, Practical, For Everyone."""

import streamlit as st
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import requests
import json

# Optional LLM support
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False

st.set_page_config(
    page_title="Chennai Weather Dashboard",
    page_icon="🌤️",
    layout="wide",
)

# ============ WEATHER DATA FETCHING ============

@st.cache_data(ttl=1800)  # Cache for 30 minutes
def get_current_weather():
    """Fetch current weather from Open-Meteo (free, no API key)."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "current": "temperature_2m,relative_humidity_2m,precipitation,rain,weather_code,wind_speed_10m,wind_direction_10m,apparent_temperature,cloud_cover",
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
    """Fetch 7-day forecast from Open-Meteo."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "daily": "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,precipitation_probability_max,wind_speed_10m_max,sunrise,sunset",
        "timezone": "Asia/Kolkata"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("daily", {})
    except:
        pass
    return {}


@st.cache_data(ttl=1800)
def get_hourly_forecast():
    """Fetch hourly forecast for next 24 hours."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "hourly": "temperature_2m,relative_humidity_2m,precipitation_probability,precipitation,rain,weather_code,wind_speed_10m,cloud_cover",
        "forecast_days": 2,
        "timezone": "Asia/Kolkata"
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        if resp.status_code == 200:
            return resp.json().get("hourly", {})
    except:
        pass
    return {}


@st.cache_data(ttl=3600)
def get_historical_data(days_back=30):
    """Fetch historical weather data."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": 13.0827,
        "longitude": 80.2707,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,rain_sum,wind_speed_10m_max",
        "timezone": "Asia/Kolkata"
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        if resp.status_code == 200:
            return resp.json().get("daily", {})
    except:
        pass
    return {}


def weather_code_to_emoji(code):
    """Convert WMO weather code to emoji and description."""
    codes = {
        0: ("☀️", "Clear sky"),
        1: ("🌤️", "Mainly clear"),
        2: ("⛅", "Partly cloudy"),
        3: ("☁️", "Overcast"),
        45: ("🌫️", "Foggy"),
        48: ("🌫️", "Depositing rime fog"),
        51: ("🌧️", "Light drizzle"),
        53: ("🌧️", "Moderate drizzle"),
        55: ("🌧️", "Dense drizzle"),
        61: ("🌧️", "Slight rain"),
        63: ("🌧️", "Moderate rain"),
        65: ("🌧️", "Heavy rain"),
        71: ("🌨️", "Slight snow"),
        73: ("🌨️", "Moderate snow"),
        75: ("🌨️", "Heavy snow"),
        80: ("🌦️", "Slight rain showers"),
        81: ("🌦️", "Moderate rain showers"),
        82: ("⛈️", "Violent rain showers"),
        95: ("⛈️", "Thunderstorm"),
        96: ("⛈️", "Thunderstorm with hail"),
        99: ("⛈️", "Thunderstorm with heavy hail"),
    }
    return codes.get(code, ("❓", "Unknown"))


# ============ AI ASSISTANT ============

def get_llm_response(question: str, weather_context: str) -> str:
    """Get response from Groq LLM (free tier) with weather context."""
    try:
        # Check for API key in Streamlit secrets
        api_key = None
        if hasattr(st, 'secrets') and 'groq' in st.secrets:
            api_key = st.secrets['groq'].get('api_key')

        if not api_key or not GROQ_AVAILABLE:
            return None

        client = Groq(api_key=api_key)

        system_prompt = f"""You are a helpful weather assistant for Chennai, India.
Answer questions clearly and practically based on the current weather data provided.
Be concise but informative. Give actionable advice when relevant.

Current Weather Data:
{weather_context}

Guidelines:
- Be practical and specific to Chennai
- Consider local conditions (tropical climate, monsoons, humidity)
- Give clear yes/no answers when appropriate
- Include relevant numbers from the data
- Keep responses under 150 words"""

        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",  # Fast, free model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question}
            ],
            max_tokens=300,
            temperature=0.7
        )

        return response.choices[0].message.content
    except Exception as e:
        return None


def get_weather_advice(current, forecast, question=""):
    """Generate advice based on weather data."""
    temp = current.get("temperature_2m", 30)
    humidity = current.get("relative_humidity_2m", 70)
    rain = current.get("rain", 0)
    wind = current.get("wind_speed_10m", 10)
    clouds = current.get("cloud_cover", 50)

    # Get today's forecast
    today_rain_prob = 0
    today_rain_sum = 0
    if forecast.get("precipitation_probability_max"):
        today_rain_prob = forecast["precipitation_probability_max"][0] or 0
    if forecast.get("precipitation_sum"):
        today_rain_sum = forecast["precipitation_sum"][0] or 0

    advice = []

    # Umbrella advice
    if rain > 0 or today_rain_prob > 50:
        advice.append(f"🌂 **Take an umbrella!** Rain probability: {today_rain_prob}%")
    elif today_rain_prob > 30:
        advice.append(f"🌂 **Maybe carry an umbrella** - {today_rain_prob}% chance of rain")
    else:
        advice.append("☀️ **No umbrella needed** - Low rain probability")

    # Temperature advice
    if temp > 35:
        advice.append(f"🥵 **Very hot ({temp}°C)** - Stay hydrated, avoid afternoon sun")
    elif temp > 30:
        advice.append(f"🌡️ **Warm ({temp}°C)** - Light clothes recommended")
    elif temp < 25:
        advice.append(f"😊 **Pleasant ({temp}°C)** - Great weather!")

    # Humidity advice
    if humidity > 80:
        advice.append(f"💧 **Very humid ({humidity}%)** - May feel uncomfortable outdoors")
    elif humidity > 60:
        advice.append(f"💧 **Moderately humid ({humidity}%)**")

    # Wind advice
    if wind > 30:
        advice.append(f"💨 **Windy ({wind} km/h)** - Hold onto your hat!")

    # Travel advice
    if rain > 5 or today_rain_sum > 20:
        advice.append("🚗 **Travel:** Roads may be wet, drive carefully")
    elif today_rain_prob < 20 and temp < 35:
        advice.append("✈️ **Great day for outdoor activities!**")

    return advice


# ============ MAIN APP ============

def main():
    st.title("🌤️ Chennai Weather Dashboard")
    st.markdown("*Real-time weather, forecasts, and data for everyday decisions*")

    # Create tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🏠 Today",
        "📅 7-Day Forecast",
        "📊 Data Explorer",
        "🗺️ Weather Map",
        "💬 Ask AI"
    ])

    with tab1:
        render_today_tab()

    with tab2:
        render_forecast_tab()

    with tab3:
        render_data_explorer()

    with tab4:
        render_weather_map()

    with tab5:
        render_ai_chat()


def render_today_tab():
    """Today's weather - simple and practical."""
    st.subheader("Today's Weather in Chennai")

    # Fetch data
    current = get_current_weather()
    forecast = get_weather_forecast()
    hourly = get_hourly_forecast()

    if not current:
        st.error("Unable to fetch weather data. Please try again later.")
        return

    # Current conditions - big and clear
    weather_code = current.get("weather_code", 0)
    emoji, desc = weather_code_to_emoji(weather_code)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        st.markdown(f"# {emoji}")
        st.markdown(f"**{desc}**")

    with col2:
        temp = current.get("temperature_2m", "--")
        feels = current.get("apparent_temperature", temp)
        st.metric("Temperature", f"{temp}°C", f"Feels like {feels}°C")

    with col3:
        humidity = current.get("relative_humidity_2m", "--")
        st.metric("Humidity", f"{humidity}%")

    # More details
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        rain = current.get("rain", 0)
        st.metric("Rain (now)", f"{rain} mm")

    with col2:
        wind = current.get("wind_speed_10m", 0)
        st.metric("Wind", f"{wind} km/h")

    with col3:
        clouds = current.get("cloud_cover", 0)
        st.metric("Cloud Cover", f"{clouds}%")

    with col4:
        if forecast.get("precipitation_probability_max"):
            rain_prob = forecast["precipitation_probability_max"][0] or 0
            st.metric("Rain Chance Today", f"{rain_prob}%")

    # Quick advice
    st.divider()
    st.markdown("### 🎯 Quick Advice")

    advice = get_weather_advice(current, forecast)
    for a in advice:
        st.markdown(a)

    # Hourly forecast chart
    st.divider()
    st.markdown("### ⏰ Next 24 Hours")

    if hourly.get("time"):
        # Get next 24 hours
        times = hourly["time"][:24]
        temps = hourly.get("temperature_2m", [])[:24]
        rain_probs = hourly.get("precipitation_probability", [])[:24]

        hourly_df = pd.DataFrame({
            "Time": [t.split("T")[1][:5] for t in times],
            "Temperature (°C)": temps,
            "Rain Probability (%)": rain_probs
        })

        st.line_chart(hourly_df.set_index("Time"))

    # Download current data
    st.divider()
    st.download_button(
        "📥 Download Current Weather (JSON)",
        data=json.dumps(current, indent=2),
        file_name=f"chennai_weather_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
        mime="application/json"
    )


def render_forecast_tab():
    """7-day weather forecast."""
    st.subheader("📅 7-Day Forecast")

    forecast = get_weather_forecast()

    if not forecast.get("time"):
        st.error("Unable to fetch forecast data.")
        return

    # Forecast cards
    dates = forecast["time"]
    max_temps = forecast.get("temperature_2m_max", [])
    min_temps = forecast.get("temperature_2m_min", [])
    rain_sums = forecast.get("precipitation_sum", [])
    rain_probs = forecast.get("precipitation_probability_max", [])
    weather_codes = forecast.get("weather_code", [])

    cols = st.columns(7)

    for i, col in enumerate(cols):
        if i < len(dates):
            with col:
                date = datetime.strptime(dates[i], "%Y-%m-%d")
                day_name = date.strftime("%a")
                day_num = date.strftime("%d")

                emoji, desc = weather_code_to_emoji(weather_codes[i] if i < len(weather_codes) else 0)
                max_t = max_temps[i] if i < len(max_temps) else "--"
                min_t = min_temps[i] if i < len(min_temps) else "--"
                rain_p = rain_probs[i] if i < len(rain_probs) else 0

                st.markdown(f"**{day_name}**")
                st.markdown(f"{day_num}")
                st.markdown(f"# {emoji}")
                st.markdown(f"**{max_t}°** / {min_t}°")
                if rain_p and rain_p > 20:
                    st.markdown(f"🌧️ {rain_p}%")

    # Detailed table
    st.divider()
    st.markdown("### Detailed Forecast")

    forecast_df = pd.DataFrame({
        "Date": dates,
        "Max Temp (°C)": max_temps,
        "Min Temp (°C)": min_temps,
        "Rain (mm)": rain_sums,
        "Rain Probability (%)": rain_probs,
    })

    st.dataframe(forecast_df, use_container_width=True)

    # Charts
    st.markdown("### Temperature Trend")
    temp_df = pd.DataFrame({
        "Date": dates,
        "Max": max_temps,
        "Min": min_temps
    })
    st.line_chart(temp_df.set_index("Date"))

    st.markdown("### Rain Forecast")
    rain_df = pd.DataFrame({
        "Date": dates,
        "Rain (mm)": rain_sums,
        "Probability (%)": rain_probs
    })
    st.bar_chart(rain_df.set_index("Date")["Rain (mm)"])


def render_data_explorer():
    """Interactive data exploration."""
    st.subheader("📊 Data Explorer")
    st.markdown("*Explore historical weather data*")

    # Time range selector
    col1, col2 = st.columns(2)
    with col1:
        days_back = st.selectbox(
            "Time Period",
            [7, 14, 30, 60, 90],
            index=2,
            format_func=lambda x: f"Last {x} days"
        )

    with col2:
        variables = st.multiselect(
            "Variables to Show",
            ["Temperature (Max)", "Temperature (Min)", "Temperature (Mean)", "Rainfall", "Wind Speed"],
            default=["Temperature (Mean)", "Rainfall"]
        )

    # Fetch data
    with st.spinner("Loading historical data..."):
        data = get_historical_data(days_back)

    if not data.get("time"):
        st.error("Unable to fetch historical data.")
        return

    # Build dataframe
    df = pd.DataFrame({
        "Date": pd.to_datetime(data["time"]),
        "Temperature (Max)": data.get("temperature_2m_max", []),
        "Temperature (Min)": data.get("temperature_2m_min", []),
        "Temperature (Mean)": data.get("temperature_2m_mean", []),
        "Rainfall": data.get("precipitation_sum", []),
        "Wind Speed": data.get("wind_speed_10m_max", [])
    })

    # Charts
    if variables:
        st.markdown("### Time Series")
        chart_df = df[["Date"] + variables].set_index("Date")
        st.line_chart(chart_df)

    # Statistics
    st.markdown("### Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Avg Temperature", f"{df['Temperature (Mean)'].mean():.1f}°C")
    with col2:
        st.metric("Total Rainfall", f"{df['Rainfall'].sum():.1f} mm")
    with col3:
        st.metric("Rainy Days", f"{(df['Rainfall'] > 0.1).sum()}")
    with col4:
        st.metric("Max Temperature", f"{df['Temperature (Max)'].max():.1f}°C")

    # Data table
    st.markdown("### Raw Data")
    st.dataframe(df, use_container_width=True)

    # Download
    st.download_button(
        "📥 Download Data (CSV)",
        data=df.to_csv(index=False),
        file_name=f"chennai_weather_{days_back}days.csv",
        mime="text/csv"
    )


def render_weather_map():
    """Weather map visualization."""
    st.subheader("🗺️ Chennai Weather Map")

    current = get_current_weather()

    # Create map
    m = folium.Map(location=[13.0827, 80.2707], zoom_start=11, tiles="CartoDB positron")

    # Weather stations / points of interest
    locations = [
        {"name": "Chennai Central", "lat": 13.0827, "lon": 80.2707, "type": "city_center"},
        {"name": "Chennai Airport", "lat": 12.9941, "lon": 80.1709, "type": "airport"},
        {"name": "Marina Beach", "lat": 13.0500, "lon": 80.2824, "type": "beach"},
        {"name": "Guindy", "lat": 13.0067, "lon": 80.2206, "type": "suburb"},
        {"name": "Tambaram", "lat": 12.9249, "lon": 80.1000, "type": "suburb"},
        {"name": "Avadi", "lat": 13.1067, "lon": 80.0970, "type": "suburb"},
        {"name": "Velachery", "lat": 12.9815, "lon": 80.2180, "type": "suburb"},
        {"name": "Anna Nagar", "lat": 13.0850, "lon": 80.2101, "type": "suburb"},
    ]

    temp = current.get("temperature_2m", 30)
    rain = current.get("rain", 0)

    for loc in locations:
        # Simulate slight variations
        np.random.seed(hash(loc["name"]) % 2**32)
        loc_temp = temp + np.random.uniform(-1.5, 1.5)
        loc_rain = max(0, rain + np.random.uniform(-0.5, 0.5))

        # Color based on temperature
        if loc_temp > 35:
            color = "red"
        elif loc_temp > 32:
            color = "orange"
        elif loc_temp > 28:
            color = "yellow"
        else:
            color = "green"

        popup_html = f"""
        <b>{loc['name']}</b><br>
        🌡️ {loc_temp:.1f}°C<br>
        🌧️ {loc_rain:.1f} mm
        """

        folium.CircleMarker(
            location=[loc["lat"], loc["lon"]],
            radius=15,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(popup_html, max_width=200)
        ).add_to(m)

    # Add legend
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background: white; padding: 10px; border-radius: 5px; border: 1px solid gray;">
        <b>Temperature</b><br>
        <span style="color: green;">●</span> &lt; 28°C (Cool)<br>
        <span style="color: yellow;">●</span> 28-32°C (Warm)<br>
        <span style="color: orange;">●</span> 32-35°C (Hot)<br>
        <span style="color: red;">●</span> &gt; 35°C (Very Hot)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    st_folium(m, width=800, height=500)

    # Current conditions summary
    st.markdown("### Current Conditions")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Temperature", f"{current.get('temperature_2m', '--')}°C")
    with col2:
        st.metric("Humidity", f"{current.get('relative_humidity_2m', '--')}%")
    with col3:
        st.metric("Rain", f"{current.get('rain', 0)} mm")
    with col4:
        st.metric("Wind", f"{current.get('wind_speed_10m', '--')} km/h")


def render_ai_chat():
    """AI assistant for weather questions."""
    st.subheader("💬 Ask About Weather")
    st.markdown("*Ask any question about Chennai weather*")

    # Check if LLM is available
    has_llm = False
    if GROQ_AVAILABLE and hasattr(st, 'secrets') and 'groq' in st.secrets:
        has_llm = st.secrets['groq'].get('api_key') is not None

    if has_llm:
        st.success("🤖 AI-powered answers enabled (Llama 3.1)")
    else:
        st.info("💡 Quick answers available. For AI-powered responses, add Groq API key to secrets.")

    # Common questions
    st.markdown("### Quick Questions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🌂 Should I take an umbrella?", use_container_width=True):
            st.session_state.ai_question = "umbrella"
        if st.button("👨‍👩‍👧‍👦 Good time to visit Chennai with kids?", use_container_width=True):
            st.session_state.ai_question = "visit"
        if st.button("🏖️ Can I go to the beach today?", use_container_width=True):
            st.session_state.ai_question = "beach"

    with col2:
        if st.button("🚗 How are driving conditions?", use_container_width=True):
            st.session_state.ai_question = "driving"
        if st.button("🏃 Good for outdoor exercise?", use_container_width=True):
            st.session_state.ai_question = "exercise"
        if st.button("📅 Best day this week?", use_container_width=True):
            st.session_state.ai_question = "best_day"

    # Custom question
    st.markdown("### Or Ask Your Own Question")
    custom_q = st.text_input("Type your question:", placeholder="e.g., Will it rain tomorrow?")

    if custom_q:
        st.session_state.ai_question = "custom"
        st.session_state.custom_question = custom_q

    # Generate answer
    if hasattr(st.session_state, 'ai_question'):
        st.divider()
        st.markdown("### 🤖 Answer")

        # Fetch data
        current = get_current_weather()
        forecast = get_weather_forecast()

        question = st.session_state.ai_question

        temp = current.get("temperature_2m", 30)
        humidity = current.get("relative_humidity_2m", 70)
        rain = current.get("rain", 0)
        wind = current.get("wind_speed_10m", 10)

        rain_probs = forecast.get("precipitation_probability_max", [0]*7)
        max_temps = forecast.get("temperature_2m_max", [30]*7)
        rain_sums = forecast.get("precipitation_sum", [0]*7)
        dates = forecast.get("time", [])

        if question == "umbrella":
            today_rain = rain_probs[0] if rain_probs else 0
            if today_rain > 60:
                st.success(f"**YES, definitely take an umbrella!** 🌂\n\nRain probability today is {today_rain}%. Current rain: {rain}mm")
            elif today_rain > 30:
                st.warning(f"**Maybe carry one** - {today_rain}% chance of rain today. Better safe than sorry!")
            else:
                st.info(f"**Probably not needed** - Only {today_rain}% chance of rain. Sky looks clear!")

        elif question == "visit":
            # Find best days
            best_days = []
            for i, (prob, temp, date) in enumerate(zip(rain_probs[:7], max_temps[:7], dates[:7])):
                if prob < 30 and temp < 35:
                    day_name = datetime.strptime(date, "%Y-%m-%d").strftime("%A, %b %d")
                    best_days.append((day_name, prob, temp))

            if len(best_days) >= 3:
                st.success(f"**Great time to visit!** 🎉\n\nBest days this week:")
                for day, prob, temp in best_days[:3]:
                    st.markdown(f"- **{day}**: {temp}°C, {prob}% rain chance")
            elif best_days:
                st.warning(f"**Okay to visit**, but limited good days:\n")
                for day, prob, temp in best_days:
                    st.markdown(f"- **{day}**: {temp}°C, {prob}% rain chance")
            else:
                st.error("**Not ideal this week** - High rain chances or very hot temperatures expected.")

        elif question == "beach":
            if rain > 0 or rain_probs[0] > 50:
                st.error(f"**Not recommended today** 🌧️\n\nRain expected ({rain_probs[0]}% probability). Try another day!")
            elif wind > 25:
                st.warning(f"**Windy at the beach** ({wind} km/h). Sand might be uncomfortable.")
            elif temp > 38:
                st.warning(f"**Very hot ({temp}°C)** - Go early morning or evening, bring lots of water!")
            else:
                st.success(f"**Great beach day!** 🏖️\n\nTemperature: {temp}°C, Low rain chance, Moderate wind.")

        elif question == "driving":
            if rain > 2:
                st.error(f"**Wet roads** 🌧️ - Currently raining ({rain}mm). Drive carefully, reduce speed.")
            elif rain_probs[0] > 60:
                st.warning(f"**Expect rain** - {rain_probs[0]}% chance today. Keep wipers ready!")
            else:
                st.success(f"**Good driving conditions** 🚗\n\nDry roads, visibility good.")

        elif question == "exercise":
            if temp > 35:
                st.warning(f"**Too hot ({temp}°C)** - Exercise early morning (before 7am) or evening (after 6pm).")
            elif humidity > 85:
                st.warning(f"**Very humid ({humidity}%)** - Stay hydrated, take breaks.")
            elif rain > 0:
                st.info(f"**Indoor workout recommended** - It's raining!")
            else:
                st.success(f"**Good for outdoor exercise!** 🏃\n\nTemperature: {temp}°C, Humidity: {humidity}%")

        elif question == "best_day":
            # Score each day
            scores = []
            for i in range(min(7, len(rain_probs))):
                score = 100
                score -= rain_probs[i] * 0.5  # Penalize rain
                if max_temps[i] > 35:
                    score -= 20
                if max_temps[i] > 38:
                    score -= 20
                scores.append((i, score, dates[i], rain_probs[i], max_temps[i]))

            scores.sort(key=lambda x: x[1], reverse=True)
            best = scores[0]
            day_name = datetime.strptime(best[2], "%Y-%m-%d").strftime("%A, %B %d")

            st.success(f"**Best day this week: {day_name}** ⭐\n\n"
                      f"- Temperature: {best[4]}°C\n"
                      f"- Rain probability: {best[3]}%\n"
                      f"- Score: {best[1]:.0f}/100")

            st.markdown("**Other good days:**")
            for i, (idx, score, date, rain_p, temp) in enumerate(scores[1:4]):
                day = datetime.strptime(date, "%Y-%m-%d").strftime("%A")
                st.markdown(f"{i+2}. {day}: {temp}°C, {rain_p}% rain (Score: {score:.0f})")

        elif question == "custom":
            user_question = st.session_state.custom_question

            # Build weather context for LLM
            weather_context = f"""
Current Conditions (Chennai):
- Temperature: {temp}°C
- Humidity: {humidity}%
- Current rain: {rain}mm
- Wind: {wind} km/h

7-Day Forecast:
"""
            for i in range(min(7, len(dates))):
                day = datetime.strptime(dates[i], "%Y-%m-%d").strftime("%A")
                weather_context += f"- {day}: {max_temps[i]}°C, {rain_probs[i]}% rain chance, {rain_sums[i]}mm expected\n"

            # Try LLM first
            with st.spinner("Thinking..."):
                llm_response = get_llm_response(user_question, weather_context)

            if llm_response:
                st.markdown(f"🤖 **AI Response:**\n\n{llm_response}")
                st.caption("*Powered by Llama 3.1 via Groq (free tier)*")
            else:
                # Fallback to keyword matching
                q_lower = user_question.lower()
                if "rain" in q_lower or "umbrella" in q_lower:
                    st.info(f"**Rain forecast:**\n- Today: {rain_probs[0]}% chance\n- Tomorrow: {rain_probs[1] if len(rain_probs) > 1 else 'N/A'}%\n- Current rain: {rain}mm")
                elif "temp" in q_lower or "hot" in q_lower or "cold" in q_lower:
                    st.info(f"**Temperature:**\n- Now: {temp}°C\n- Today's max: {max_temps[0]}°C\n- Week range: {min(max_temps)}-{max(max_temps)}°C")
                elif "humid" in q_lower:
                    st.info(f"**Humidity:** Currently {humidity}%\n\n{'Very humid!' if humidity > 80 else 'Comfortable' if humidity < 60 else 'Moderately humid'}")
                elif "wind" in q_lower:
                    st.info(f"**Wind:** Currently {wind} km/h\n\n{'Windy!' if wind > 20 else 'Light breeze' if wind > 5 else 'Calm'}")
                elif "tomorrow" in q_lower:
                    if len(dates) > 1:
                        st.info(f"**Tomorrow's forecast:**\n- Max temp: {max_temps[1]}°C\n- Rain chance: {rain_probs[1]}%\n- Expected rain: {rain_sums[1]}mm")
                else:
                    st.info(f"**Current conditions:**\n- Temperature: {temp}°C\n- Humidity: {humidity}%\n- Rain: {rain}mm\n- Wind: {wind} km/h\n\n*To enable AI-powered answers, add a Groq API key in Streamlit secrets.*")

        # Clear question
        if st.button("Ask another question"):
            del st.session_state.ai_question
            if hasattr(st.session_state, 'custom_question'):
                del st.session_state.custom_question
            st.rerun()


if __name__ == "__main__":
    main()
