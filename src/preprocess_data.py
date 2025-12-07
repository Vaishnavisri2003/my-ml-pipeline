import json
import pandas as pd

RAW_DATA_PATH = "data/raw/weather.json"
PROCESSED_DATA_PATH = "data/processed/weather_clean.csv"

def preprocess_weather_data():
    with open(RAW_DATA_PATH, "r") as f:
        data = json.load(f)

    hourly = data["hourly"]

    df = pd.DataFrame({
        "time": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relativehumidity_2m"],
        "pressure": hourly["surface_pressure"],
        "wind_speed": hourly["wind_speed_10m"],
        "weather_code": hourly["weather_code"]
    })

    df["time"] = pd.to_datetime(df["time"])

    df = df.dropna()

    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("✔ Clean dataset saved with all features!")

if __name__ == "__main__":
    preprocess_weather_data()
