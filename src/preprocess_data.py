import json
import pandas as pd
from sklearn.preprocessing import LabelEncoder

RAW_DATA_PATH = "data/raw/weather.json"
PROCESSED_DATA_PATH = "data/processed/weather_clean.csv"

def preprocess_weather_data():
    with open(RAW_DATA_PATH, "r") as f:
        data = json.load(f)

    hourly = data["hourly"]

    # Build DataFrame with NEW FEATURES
    df = pd.DataFrame({
        "time": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relativehumidity_2m"],
        "pressure": hourly["surface_pressure"],           # NEW
        "wind_speed": hourly["windspeed_10m"],            # NEW
        "weather_code": hourly["weathercode"]             # NEW numeric code
    })

    # Convert time to datetime
    df["time"] = pd.to_datetime(df["time"])

    # Drop missing values
    df = df.dropna()

    # Encode weather code (just to ensure ML compatibility)
    label_encoder = LabelEncoder()
    df["weather_code"] = label_encoder.fit_transform(df["weather_code"])

    # Save clean dataset
    df.to_csv(PROCESSED_DATA_PATH, index=False)

    print("✔ Enhanced dataset saved to data/processed/weather_clean.csv with 5 features!")

if __name__ == "__main__":
    preprocess_weather_data()

