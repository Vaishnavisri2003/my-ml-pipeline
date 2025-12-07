import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

DATA_PATH = "data/processed/weather_clean.csv"

def train_weather_models():
    df = pd.read_csv(DATA_PATH)

    X = df[["temperature", "humidity", "pressure", "wind_speed", "weather_code"]]
    y = df["target_temp"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeRegressor(random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        save_path = f"models/{name}.pkl"
        joblib.dump(model, save_path)
        print(f"✔ Saved model: {save_path}")

    print("🎉 All models trained and saved successfully!")

if __name__ == "__main__":
    train_weather_models()
