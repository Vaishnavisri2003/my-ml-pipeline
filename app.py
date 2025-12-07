import streamlit as st
import requests
import numpy as np
import joblib

# ---------- CONFIG ----------

# API key from Streamlit secrets (both local secrets.toml and cloud)
API_KEY = st.secrets["API_KEY"]

# Available models and their file paths
MODEL_FILES = {
    "Linear Regression": "models/linear_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "Decision Tree": "models/decision_tree.pkl",
}

@st.cache_resource
def load_model(path: str):
    """Cache model load so it doesn't reload on every interaction."""
    return joblib.load(path)

# ---------- UI ----------

st.title("🌤️ Real-time Weather Prediction App (Multiple Models)")

city = st.text_input("Enter City Name", "Chennai")

model_choice = st.selectbox(
    "Choose a prediction model",
    list(MODEL_FILES.keys())
)

st.caption("📌 The model predicts approx. next-hour temperature from current weather.")

if st.button("Get Weather & Predict"):
    if not API_KEY:
        st.error("API Key not found. Check Streamlit secrets.")
    else:
        # Call OpenWeather API
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            # Extract features
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            pressure = data["main"]["pressure"]
            wind_speed = data["wind"]["speed"]
            weather_code = data["weather"][0]["id"]  # numeric condition code

            st.subheader(f"🌍 Current Weather in {city}")
            st.write(f"🌡 **Temperature:** {temp} °C")
            st.write(f"💧 **Humidity:** {humidity} %")
            st.write(f"🔵 **Pressure:** {pressure} hPa")
            st.write(f"🌬 **Wind Speed:** {wind_speed} m/s")
            st.write(f"🌈 **Weather Code:** {weather_code}")

            # Prepare features in same order as training
            features = np.array([[temp, humidity, pressure, wind_speed, weather_code]])

            # Load selected model
            model_path = MODEL_FILES[model_choice]
            model = load_model(model_path)

            # Predict
            prediction = model.predict(features)[0]

            st.success(
                f"🔮 Model **{model_choice}** predicts next-hour temperature: "
                f"**{prediction:.2f} °C**"
            )
        else:
            st.error("City not found. Try again.")

