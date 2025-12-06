import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

DATA_PATH = "data/processed/weather_clean.csv"
MODEL_PATH = "models/weather_model.pkl"

def train_model():
    df = pd.read_csv(DATA_PATH)

    X = df[["humidity"]]       # input feature
    y = df["temperature"]      # target variable

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    
    joblib.dump(model, MODEL_PATH)
    print(f"✔ Model trained and saved to {MODEL_PATH}")
    print(f"📌 Model Test Score (R²): {score:.4f}")

if __name__ == "__main__":
    train_model()
