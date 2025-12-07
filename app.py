import streamlit as st
import requests
import json
import numpy as np
import joblib

API_KEY = st.secrets["API_KEY"]
# Load trained model
model = joblib.load("models/weather_model.pkl")

st.title("🌤️ Real-time Weather Prediction App")

city = st.text_input("Enter City Name", "Chennai")

if st.button("Get Weather & Predict"):
    if not API_KEY:
        st.error("API Key not found. Check the .env file.")
    else:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()

            temp = data["main"]["temp"]
            humidity = data["main"]["humidity"]
            pressure = data["main"]["pressure"]
            wind_speed = data["wind"]["speed"]

            st.write(f"### 🌍 Weather in **{city}**")
            st.write(f"Temperature: **{temp}°C**")
            st.write(f"Humidity: **{humidity}%**")
            st.write(f"Pressure: **{pressure} hPa**")
            st.write(f"Wind Speed: **{wind_speed} m/s**")

            # Model uses only 1 feature (temperature)
            features = [[temp]]
            prediction = model.predict(features)[0]

            st.success(f"🌈 Predicted Value from Model: **{prediction:.2f}**")

        else:
            st.error("City not found. Try again.")
