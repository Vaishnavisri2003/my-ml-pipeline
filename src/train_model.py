import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

DATA_PATH = "data/processed/weather_clean.csv"
MODEL_PATH = "models/weather_model.pkl"

def train_weather_model():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # FEATURES (X) — using 5 features
    X = df[[
        "temperature",
        "humidity",
        "pressure",
        "wind_speed",
        "weather_code"
    ]]

    # TARGET (y) — you can choose what to predict
    # For now, we predict temperature of next hour
    # So we shift temperature by -1
    df["target_temp"] = df["temperature"].shift(-1)

    # Remove last row with NaN target
    df = df.dropna()

    X = df[["temperature", "humidity", "pressure", "wind_speed", "weather_code"]]
    y = df["target_temp"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"✔ Model trained successfully — MSE: {mse:.2f}")

    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✔ Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    train_weather_model()
