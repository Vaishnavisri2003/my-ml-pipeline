import requests
import json
import os

RAW_DATA_PATH = "data/raw/weather.json"

def fetch_weather_data():
    url = "https://api.open-meteo.com/v1/forecast?latitude=12.97&longitude=77.59&hourly=temperature_2m,relativehumidity_2m"
    response = requests.get(url)

    if response.status_code == 200:
        os.makedirs("data/raw", exist_ok=True)
        with open(RAW_DATA_PATH, "w") as f:
            json.dump(response.json(), f, indent=4)
        print("✔ Data fetched and saved to data/raw/weather.json")
    else:
        print("❌ API request failed:", response.status_code)

if __name__ == "__main__":
    fetch_weather_data()
