import streamlit as st
import requests
import numpy as np
import joblib

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Real-time Weather Prediction",
    page_icon="🌤️",
    layout="centered"
)

# =============================
# LOAD API KEY
# =============================
API_KEY = st.secrets.get("WEATHER_API_KEY")

if not API_KEY:
    st.error("❌ WEATHER_API_KEY not found in Streamlit secrets.")
    st.stop()

# =============================
# MODEL PATHS
# =============================
MODEL_PATHS = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Decision Tree": "models/decision_tree.pkl",
}

@st.cache_resource
def load_model(path):
    return joblib.load(path)

# =============================
# FETCH WEATHER (WeatherAPI)
# =============================
def get_weather(city):
    url = "https://api.weatherapi.com/v1/current.json"
    params = {
        "key": API_KEY,
        "q": city,
        "aqi": "no"
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code != 200:
        st.error("❌ WeatherAPI Error")
        st.json(response.json())
        return None

    data = response.json()

    return {
        "temp": data["current"]["temp_c"],
        "humidity": data["current"]["humidity"],
        "pressure": data["current"]["pressure_mb"],
        "wind_speed": data["current"]["wind_kph"] / 3.6,  # kph → m/s
        "weather_code": data["current"]["condition"]["code"],
    }

# =============================
# UI
# =============================
st.title("🌤️ Real-time Weather Prediction App")

city = st.text_input("Enter City Name", "Chennai")

model_choice = st.selectbox(
    "Choose Prediction Model",
    list(MODEL_PATHS.keys())
)

# =============================
# PREDICTION
# =============================
if st.button("Get Weather & Predict"):
    weather = get_weather(city)

    if weather:
        st.subheader(f"🌍 Weather in {city}")

        st.write(f"**Temperature:** {weather['temp']} °C")
        st.write(f"**Humidity:** {weather['humidity']} %")
        st.write(f"**Pressure:** {weather['pressure']} hPa")
        st.write(f"**Wind Speed:** {weather['wind_speed']:.2f} m/s")
        st.write(f"**Weather Code:** {weather['weather_code']}")

        features = np.array([[
            weather["temp"],
            weather["humidity"],
            weather["pressure"],
            weather["wind_speed"],
            weather["weather_code"]
        ]])

        model = load_model(MODEL_PATHS[model_choice])
        prediction = model.predict(features)[0]

        st.success(
            f"🔮 Prediction using **{model_choice}**: **{prediction:.2f} °C**"
        )

