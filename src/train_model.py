import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

DATA_PATH = "data/processed/weather_clean.csv"
MODEL_DIR = "models/"

def train_weather_models():
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Prepare features and target
    X = df[["temperature", "humidity", "pressure", "wind_speed", "weather_code"]]
    df["target_temp"] = df["temperature"].shift(-1)
    df = df.dropna()

    X = df[["temperature", "humidity", "pressure", "wind_speed", "weather_code"]]
    y = df["target_temp"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "decision_tree": DecisionTreeRegressor(random_state=42)
    }

    results = {}

    for name, model in models.items():
        print(f"\n🔧 Training {name} model...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        results[name] = mse

        # Save the model
        joblib.dump(model, f"{MODEL_DIR}{name}.pkl")
        print(f"✔ {name} saved to {MODEL_DIR}{name}.pkl (MSE={mse:.2f})")

    print("\n📌 Final Model Comparison (Lower MSE = Better)")
    for name, mse in results.items():
        print(f"{name}: {mse:.2f}")

if __name__ == "__main__":
    train_weather_models()
