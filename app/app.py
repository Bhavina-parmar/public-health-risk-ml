# app/app.py

from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import sys
from app.logger import logger
from app.alerts import (
    DRIFT_ALERT_THRESHOLD,
    LOW_CONFIDENCE_THRESHOLD
)
from app.logger import logger

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.schema import SCHEMA

app = Flask(__name__)

print("\n🚀 Starting Flask API...")

# ----------------------------
# Resolve paths
# ----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

with open(os.path.join(MODEL_DIR, "latest.txt"), "r", encoding="utf-8") as f:
    ACTIVE_VERSION = f.read().strip()

MODEL_PATH = os.path.join(MODEL_DIR, ACTIVE_VERSION, "model.pkl")
FEATURE_PATH = os.path.join(MODEL_DIR, ACTIVE_VERSION, "features.pkl")
STATS_PATH = os.path.join(MODEL_DIR, ACTIVE_VERSION, "stats.pkl")

print(f"📦 Loading model version: {ACTIVE_VERSION}")

# ----------------------------
# Load model + features + stats
# ----------------------------
model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURE_PATH)
TRAIN_STATS = joblib.load(STATS_PATH)

print("🎉 Model, features & stats loaded successfully")

# ----------------------------
# Drift detection
# ----------------------------
def detect_drift(input_data):
    drifted_features = []

    for feature in FEATURES:
        mean = TRAIN_STATS[feature]["mean"]
        std = TRAIN_STATS[feature]["std"]

        if std == 0:
            continue

        z_score = abs((input_data[feature] - mean) / std)

        if z_score > 3:  # 3-sigma rule
            drifted_features.append(feature)

    return drifted_features

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return jsonify({"message": "API running"})

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    logger.info(
        f"PREDICTION | result={pred} | confidence={confidence:.4f} | drift={drifted}"
    )
    # 🔒 Schema validation
    for field, rules in SCHEMA.items():
        if field not in data:
            logger.warning(f"VALIDATION_ERROR | missing_field={field}")
            return jsonify({"error": f"Missing field: {field}"}), 400
        try:
    # existing predict logic
            try:
                value = float(data[field])
            except:
                return jsonify({"error": f"{field} must be numeric"}), 400

            if "min" in rules and value < rules["min"]:
                return jsonify({"error": f"{field} below minimum"}), 400

            if "max" in rules and value > rules["max"]:
                return jsonify({"error": f"{field} above maximum"}), 400
        except Exception as e:
            logger.error(f"INTERNAL_ERROR | {str(e)}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

    


    # Feature order consistency
    X = np.array([[data[f] for f in FEATURES]], dtype=float)

    # Prediction + confidence
    proba = model.predict_proba(X)[0]
    pred = int(proba.argmax())
    confidence = float(proba[pred])

    # Local explanation
    contributions = []
    for i, feature in enumerate(FEATURES):
        score = abs(X[0][i] * model.feature_importances_[i])
        contributions.append((feature, round(score, 4)))

    top_features = sorted(contributions, key=lambda x: x[1], reverse=True)[:5]

    # 🔥 Drift detection
    input_dict = {f: float(data[f]) for f in FEATURES}
    drifted = detect_drift(input_dict)
    if drifted:
        logger.warning(f"DATA_DRIFT | features={drifted}")
    alerts = []

# 🚨 Drift alert
    if len(drifted) >= DRIFT_ALERT_THRESHOLD:
        msg = f"ALERT | High drift detected | features={drifted}"
        logger.error(msg)
        alerts.append("HIGH_DRIFT")

# 🚨 Low confidence alert
    if confidence < LOW_CONFIDENCE_THRESHOLD:
        msg = f"ALERT | Low confidence | confidence={confidence:.4f}"
        logger.warning(msg)
        alerts.append("LOW_CONFIDENCE")
    return jsonify({
        "prediction": pred,
        "confidence": round(confidence, 4),
        "top_features": top_features,
        "drift_detected": drifted,
        "alerts": alerts
    })



@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    importance = dict(zip(FEATURES, model.feature_importances_))
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    return jsonify({"feature_importance": sorted_imp})

# ----------------------------
# Start server
# ----------------------------
if __name__ == "__main__":
    print("🌍 Flask server running on http://127.0.0.1:5000\n")
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
