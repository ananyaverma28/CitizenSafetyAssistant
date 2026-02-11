# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timezone
from math import radians, cos, sin, asin, sqrt

# -----------------------
# Config
# -----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "notebooks", "artifacts", "xgb_accident_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "..", "notebooks", "artifacts", "feature_columns.pkl")

OPENWEATHER_API_KEY = os.environ.get("OPENWEATHER_API_KEY")
if not OPENWEATHER_API_KEY:
    print("OPENWEATHER_API_KEY not set in environment. Set it and restart the app.")
else:
    print("OPENWEATHER_API_KEY loaded successfully.")

SAFE_DEFAULTS = {
    "Visibility(mi)": 10,
    "Temperature(F)": 70,
    "Humidity(%)": 70,
    "Pressure(in)": 29.92,
    "Wind_Speed(mph)": 8,
    "Precipitation(in)": 0.0
}

# -----------------------
# Helpers
# -----------------------
def haversine_miles(lat1, lon1, lat2, lon2):
    R = 3958.8
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

def geocode_city(city_name):
    """Get lat/lon for a city using OpenWeather free geocoding API"""
    if not OPENWEATHER_API_KEY:
        return None, None, None
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {"q": city_name, "limit": 1, "appid": OPENWEATHER_API_KEY}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        if not data:
            return None, None, None
        item = data[0]
        lat, lon = item.get("lat"), item.get("lon")
        display = ", ".join([p for p in [item.get("name"), item.get("state"), item.get("country")] if p])
        return lat, lon, display
    except Exception as e:
        print(f"Geocoding error for {city_name}: {e}")
        return None, None, None

def fetch_current_weather(lat, lon):
    """Fetch current weather using free API"""
    if not OPENWEATHER_API_KEY:
        return None
    url = "http://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": OPENWEATHER_API_KEY, "units": "imperial"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        temp_F = data.get("main", {}).get("temp")
        humidity = data.get("main", {}).get("humidity")
        pressure_in = data.get("main", {}).get("pressure") * 0.02953 if data.get("main", {}).get("pressure") else None
        wind_mph = data.get("wind", {}).get("speed")
        visibility_mi = data.get("visibility", None)
        if visibility_mi is not None:
            visibility_mi = visibility_mi / 1609.344  # meters → miles
        precip_in = 0.0
        if "rain" in data:
            precip_in = data["rain"].get("1h", 0) * 0.0393701
        elif "snow" in data:
            precip_in = data["snow"].get("1h", 0) * 0.0393701
        weather = data.get("weather", [{}])[0]
        return {
            "temp_F": temp_F,
            "humidity": humidity,
            "pressure_in": pressure_in,
            "wind_mph": wind_mph,
            "visibility_mi": visibility_mi,
            "precip_in": precip_in,
            "weather_main": weather.get("main", ""),
            "weather_desc": weather.get("description", ""),
        }
    except Exception as e:
        print(f"Weather fetch error for ({lat},{lon}): {e}")
        return None

# -----------------------
# Load model
# -----------------------
try:
    xgb_model = joblib.load(MODEL_PATH)
    feature_columns = joblib.load(FEATURES_PATH)
    print("✅ Loaded model and feature columns.")
except Exception as e:
    print("❌ Error loading model/features:", e)
    xgb_model = None
    feature_columns = []

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({"status": "ok", "message": "Citizen Safety Assistant running."})

@app.route("/predict_route", methods=["POST"])
def predict_route():
    if xgb_model is None:
        return jsonify({"error": "Model not loaded."}), 500

    payload = request.get_json(force=True)
    from_city = payload.get("from_city")
    to_city = payload.get("to_city")
    travel_date = payload.get("travel_date")
    travel_time = payload.get("travel_time", "12:00")

    if not from_city or not to_city or not travel_date:
        return jsonify({"error": "Please provide from_city, to_city, and travel_date."}), 400

    # Parse datetime
    try:
        travel_dt = datetime.fromisoformat(f"{travel_date}T{travel_time}")
    except Exception:
        return jsonify({"error": "travel_date or travel_time format invalid."}), 400

    # Geocode cities
    lat1, lon1, disp1 = geocode_city(from_city)
    lat2, lon2, disp2 = geocode_city(to_city)
    if lat1 is None or lat2 is None:
        return jsonify({"error": "Could not geocode one or both cities."}), 400

    # Fetch current weather (free API)
    weather1 = fetch_current_weather(lat1, lon1)
    weather2 = fetch_current_weather(lat2, lon2)
    if weather1 is None or weather2 is None:
        return jsonify({"error": "Failed to fetch weather. Check API key or network."}), 500

    # Build features
    def safe(v, key):
        return SAFE_DEFAULTS.get(key, 0) if v is None else v

    feature_input = {
        "Temperature(F)": float(np.nanmean([safe(weather1.get("temp_F"), "Temperature(F)"),
                                            safe(weather2.get("temp_F"), "Temperature(F)") ])),
        "Humidity(%)": float(np.nanmean([safe(weather1.get("humidity"), "Humidity(%)"),
                                         safe(weather2.get("humidity"), "Humidity(%)")])),
        "Pressure(in)": float(np.nanmean([safe(weather1.get("pressure_in"), "Pressure(in)"),
                                          safe(weather2.get("pressure_in"), "Pressure(in)")])),
        "Visibility(mi)": float(np.nanmean([safe(weather1.get("visibility_mi"), "Visibility(mi)"),
                                           safe(weather2.get("visibility_mi"), "Visibility(mi)")])),
        "Wind_Speed(mph)": float(np.nanmean([safe(weather1.get("wind_mph"), "Wind_Speed(mph)"),
                                            safe(weather2.get("wind_mph"), "Wind_Speed(mph)")])),
        "Precipitation(in)": float(np.nanmean([safe(weather1.get("precip_in"), "Precipitation(in)"),
                                              safe(weather2.get("precip_in"), "Precipitation(in)")])),
    }

    # Weather condition
    conditions = [weather1.get("weather_main", "").lower(), weather1.get("weather_desc", "").lower(),
                  weather2.get("weather_main", "").lower(), weather2.get("weather_desc", "").lower()]
    feature_input["Weather_Condition"] = "Clear"
    for w in conditions:
        if any(k in w for k in ["rain","snow","thunder","storm","fog","drizzle","sleet"]):
            feature_input["Weather_Condition"] = w
            break

    # Duration
    dist_miles = haversine_miles(lat1, lon1, lat2, lon2)
    duration_min = (dist_miles / 50.0) * 60.0
    feature_input["Duration(min)"] = float(duration_min)

    # Build DataFrame for model
    df = pd.DataFrame([feature_input])
    cat_cols = df.select_dtypes(include=["object"]).columns
    df[cat_cols] = df[cat_cols].astype("category")
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    df = df.reindex(columns=feature_columns, fill_value=0)

    # Predict
    try:
        pred = xgb_model.predict(df)
        severity_pred = int(pred[0])
    except Exception as e:
        return jsonify({"error": f"Model prediction failed: {e}"}), 500

    severity_levels = {
        1: "🟢 Low Risk – Conditions are safe.",
        2: "🟠 Moderate Risk – Mild hazard due to weather/visibility.",
        3: "🔴 High Risk – Strong influence of weather or congestion.",
        4: "⚫ Severe Risk – Very high chance of accident under given conditions."
    }

    response = {
        "route": f"{disp1} → {disp2}",
        "distance_miles": round(dist_miles, 2),
        "estimated_duration_min": round(duration_min, 1),
        "severity": severity_pred,
        "message": severity_levels.get(severity_pred, "Unknown"),
        "weather_from": weather1,
        "weather_to": weather2,
        "feature_input": feature_input
    }

    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
