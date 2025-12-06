import json
import pandas as pd

RAW_DATA_PATH = "data/raw/weather.json"
PROCESSED_DATA_PATH = "data/processed/weather_clean.csv"

def preprocess_weather_data():
    with open(RAW_DATA_PATH, "r") as f:
        data = json.load(f)

    # Extract hourly weather data
    hourly = data["hourly"]
    df = pd.DataFrame({
        "time": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relativehumidity_2m"]
    })

    # Convert time to datetime format
    df["time"] = pd.to_datetime(df["time"])

    # Drop missing values (good practice for ML)
    df = df.dropna()

    # Save to CSV
    df.to_csv(PROCESSED_DATA_PATH, index=False)
    print("✔ Clean dataset saved to data/processed/weather_clean.csv")

if __name__ == "__main__":
    preprocess_weather_data()
