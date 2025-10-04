# ------------------------------
# Imports
# ------------------------------
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import requests
from datetime import datetime

# ------------------------------
# Flask app
# ------------------------------
app = Flask(__name__)
CORS(app)

# ------------------------------
# Cities in Kerala with lat/lon
# ------------------------------
cities_kerala = {
    "Kochi": (9.9312, 76.2673),
    "Thiruvananthapuram": (8.5241, 76.9366),
    "Kozhikode": (11.2588, 75.7804),
    "Thrissur": (10.5276, 76.2144),
    "Alappuzha": (9.4981, 76.3388),
    "Kollam": (8.8932, 76.6141),
    "Palakkad": (10.7867, 76.6548),
    "Malappuram": (11.0732, 76.0748),
    "Kannur": (11.8745, 75.3704),
    "Pathanamthitta": (9.2641, 76.7878)
}

# ------------------------------
# ML Model download from Google Drive
# ------------------------------
MODEL_URL = "https://drive.google.com/uc?export=download&id=1W21f5fyx-5KMoJaDxuZtV0C6Qw_7GcGd"
MODEL_FILENAME = "weather_model.pkl"

def download_model(url, filename):
    if not os.path.exists(filename):
        print(f"Downloading {filename}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(filename, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"{filename} downloaded successfully!")

# Download model if missing
download_model(MODEL_URL, MODEL_FILENAME)

# Load model and encoder (encoder is already in folder)
model = joblib.load(MODEL_FILENAME)
le = joblib.load("label_encoder.pkl")

# ------------------------------
# Features
# ------------------------------
features = ['latitude', 'longitude', 'hour', 'month', 'dayofweek']

# ------------------------------
# Prediction function
# ------------------------------
def predict_weather(lat, lon, date_time):
    hour = date_time.hour
    month = date_time.month
    dayofweek = date_time.weekday()

    X_new = pd.DataFrame([[lat, lon, hour, month, dayofweek]], columns=features)
    pred = model.predict(X_new)
    label = le.inverse_transform(pred)[0]
    return label

# ------------------------------
# Forecast endpoint
# ------------------------------
@app.route("/forecast", methods=["GET"])
def forecast():
    city = request.args.get("city")
    date_str = request.args.get("date")

    if not city or not date_str:
        return jsonify({"error": "Please provide 'city' and 'date' in YYYY-MM-DD format"}), 400

    city = city.strip()
    date_str = date_str.strip()

    if city not in cities_kerala:
        return jsonify({"error": f"City '{city}' not found"}), 404

    try:
        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return jsonify({"error": "Invalid date format. Use YYYY-MM-DD"}), 400

    latitude, longitude = cities_kerala[city]
    predicted_label = predict_weather(latitude, longitude, date_obj)

    return jsonify({
        "source": "ML Model",
        "city": city,
        "date": date_str,
        "predicted_weather": predicted_label
    })

# ------------------------------
# Run Flask app
# ------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
