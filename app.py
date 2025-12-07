import streamlit as st
import requests
import numpy as np
import joblib

# Load API key securely from Streamlit Secrets
API_KEY = st.secrets["API_KEY"]

# Load the trained ML model
model = joblib.load("models/weather_model.pkl")

st.title("🌤️ Advanced Weather Prediction App (5 Features)")

city = st.text_input("Enter City Name", "Chennai")

if st.button("Get Weather & Predict"):
    if not API_KEY:
        st.error("API Key not found.")
    else:
        # API call
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            # Extract real-time features
            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            pressure = data["main"]["pressure"]
            wind_speed = data["wind"]["speed"]
            weather_code = data["weather"][0]["id"]  # condition code

            st.subheader(f"🌍 Current Weather in {city}")
            st.write(f"🌡 Temperature: **{temp}°C**")
            st.write(f"💧 Humidity: **{humidity}%**")
            st.write(f"🔵 Pressure: **{pressure} hPa**")
            st.write(f"🌬 Wind Speed: **{wind_speed} m/s**")
            st.write(f"🌈 Weather Code: **{weather_code}**")

            # Put features into the correct order for model
            features = np.array([[temp, humidity, pressure, wind_speed, weather_code]])

            # Make prediction
            prediction = model.predict(features)[0]

            st.success(f"🔮 Predicted Temperature Next Hour: **{prediction:.2f}°C**")

        else:
            st.error("City not found. Try again.")
