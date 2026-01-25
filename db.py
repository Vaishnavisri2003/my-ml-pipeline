import sqlite3
from datetime import datetime

DB_NAME = "predictions.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            city TEXT,
            temperature REAL,
            humidity REAL,
            pressure REAL,
            wind_speed REAL,
            model_used TEXT,
            prediction REAL
        )
    """)

    conn.commit()
    conn.close()

def log_prediction(city, weather, model_used, prediction):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO predictions (
            timestamp, city, temperature, humidity,
            pressure, wind_speed, model_used, prediction
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        city,
        weather["temp"],
        weather["humidity"],
        weather["pressure"],
        weather["wind_speed"],
        model_used,
        prediction
    ))

    conn.commit()
    conn.close()
