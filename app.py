from db import init_db, log_prediction
import streamlit as st
import requests
import numpy as np
import joblib
import os
import pandas as pd
import altair as alt
from dotenv import load_dotenv

# =====================================================
# STREAMLIT CONFIG (MUST BE FIRST)
# =====================================================
st.set_page_config(
    page_title="Real-time Weather Prediction",
    page_icon="🌤️",
    layout="centered"
)

# =====================================================
# LOAD ENV VARIABLES & INIT DB
# =====================================================
load_dotenv()
init_db()

# =====================================================
# LOAD OPENWEATHER API KEY (FINAL & SAFE)
# =====================================================
@st.cache_resource
def load_api_key():
    if "OPENWEATHER_API_KEY" in st.secrets:
        return st.secrets["OPENWEATHER_API_KEY"]
    if os.getenv("OPENWEATHER_API_KEY"):
        return os.getenv("OPENWEATHER_API_KEY")
    return None

API_KEY = load_api_key()

if not API_KEY:
    st.error("❌ OpenWeather API key not found. Add OPENWEATHER_API_KEY to secrets or .env")
    st.stop()

# =====================================================
# LOAD MODELS ONCE
# =====================================================
MODEL_PATHS = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Decision Tree": "models/decision_tree.pkl",
}

@st.cache_resource
def load_models():
    return {name: joblib.load(path) for name, path in MODEL_PATHS.items()}

MODELS = load_models()

# =====================================================
# FETCH WEATHER DATA (CACHED)
# =====================================================
@st.cache_data(ttl=300)
def get_weather(city):
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": API_KEY,
        "units": "metric"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.RequestException as e:
        return None, {"error": str(e)}

    data = response.json()

    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "weather_code": data["weather"][0]["id"],
    }, None

# =====================================================
# UI
# =====================================================
st.title("🌤️ Real-time Weather Prediction App")

city = st.text_input("Enter City Name", "Chennai")

model_choice = st.selectbox(
    "Choose Prediction Model",
    list(MODELS.keys())
)

# =====================================================
# PREDICTION
# =====================================================
if st.button("Get Weather & Predict"):
    weather, error = get_weather(city)

    if weather is None:
        st.error("❌ OpenWeather API Error")
        st.json(error)
        st.stop()

    st.subheader(f"🌍 Weather in {city}")

    st.metric("🌡️ Temperature (°C)", weather["temp"])
    st.metric("💧 Humidity (%)", weather["humidity"])
    st.metric("🧭 Pressure (hPa)", weather["pressure"])
    st.metric("🌬️ Wind Speed (m/s)", weather["wind_speed"])

    # -------------------------------------------------
    # WEATHER BAR CHART
    # -------------------------------------------------
    weather_df = pd.DataFrame({
        "Feature": ["Temperature", "Humidity", "Pressure", "Wind Speed"],
        "Value": [
            weather["temp"],
            weather["humidity"],
            weather["pressure"],
            weather["wind_speed"]
        ]
    })

    bar_chart = alt.Chart(weather_df).mark_bar().encode(
        x=alt.X("Feature", sort=None),
        y="Value",
        color="Feature"
    )

    st.altair_chart(bar_chart, use_container_width=True)

    # -------------------------------------------------
    # MODEL PREDICTION
    # -------------------------------------------------
    features = np.array([[
        weather["temp"],
        weather["humidity"],
        weather["pressure"],
        weather["wind_speed"],
        weather["weather_code"]
    ]])

    prediction = MODELS[model_choice].predict(features)[0]

    log_prediction(city, weather, model_choice, prediction)

    st.success(
        f"🔮 Prediction using **{model_choice}**: **{prediction:.2f} °C**"
    )

    # -------------------------------------------------
    # MODEL COMPARISON
    # -------------------------------------------------
    comparison_df = pd.DataFrame({
        "Model": MODELS.keys(),
        "Prediction": [
            model.predict(features)[0] for model in MODELS.values()
        ]
    })

    line_chart = alt.Chart(comparison_df).mark_line(point=True).encode(
        x="Model",
        y="Prediction"
    )

    st.altair_chart(line_chart, use_container_width=True)
