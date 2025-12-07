import streamlit as st
import requests
import numpy as np
import joblib

# -----------------------------
# Load API key & model paths
# -----------------------------
API_KEY = st.secrets["API_KEY"]  # Your OpenWeather API Key

MODEL_PATHS = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Decision Tree": "models/decision_tree.pkl"
}

# -----------------------------
# Function: Load model safely
# -----------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"❌ Could not load model: {path}")
        st.error(str(e))
        return None

# -----------------------------
# Function: Fetch weather data
# -----------------------------
def get_weather(city):
    url = (
        f"https://api.openweathermap.org/data/2.5/weather?"
        f"q={city}&appid={API_KEY}&units=metric"
    )
    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    return {
        "temp": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "weather_code": data["weather"][0]["id"]
    }

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌤️ Real-time Weather Prediction App")
st.markdown("### Select a Machine Learning Model for Prediction")

model_choice = st.selectbox(
    "Choose a Prediction Model:",
    list(MODEL_PATHS.keys())
)

city = st.text_input("Enter City Name", "Chennai")

if st.button("Get Weather & Predict"):
    weather = get_weather(city)

    if weather is None:
        st.error("City not found. Try another name.")
    else:
        st.subheader(f"🌍 Weather in {city}")

        st.write(f"**Temperature:** {weather['temp']} °C")
        st.write(f"**Humidity:** {weather['humidity']} %")
        st.write(f"**Pressure:** {weather['pressure']} hPa")
        st.write(f"**Wind Speed:** {weather['wind_speed']} m/s")
        st.write(f"**Weather Code:** {weather['weather_code']}")

        # -----------------------------
        # Prepare ML Input
        # -----------------------------
        features = np.array([[
            weather["temp"],
            weather["humidity"],
            weather["pressure"],
            weather["wind_speed"],
            weather["weather_code"]
        ]])

        # Load model
        model = load_model(MODEL_PATHS[model_choice])

        if model:
            prediction = model.predict(features)[0]

            st.success(
                f"🌈 **Prediction using {model_choice}: {prediction:.2f}**"
            )
