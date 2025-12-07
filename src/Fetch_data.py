import requests
import json

RAW_DATA_PATH = "data/raw/weather.json"

def fetch_weather_data():
    url = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=13.0827&longitude=80.2707&hourly="
        "temperature_2m,relativehumidity_2m,surface_pressure,wind_speed_10m,weather_code"
    )

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()

        with open(RAW_DATA_PATH, "w") as f:
            json.dump(data, f, indent=4)

        print("✔ Weather data saved to data/raw/weather.json")
    else:
        print("✖ Failed to fetch data:", response.status_code)

if __name__ == "__main__":
    fetch_weather_data()
