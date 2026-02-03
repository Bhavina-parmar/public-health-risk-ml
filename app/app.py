from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
import csv
import datetime

app = Flask(__name__)

print("\n🚀 Starting Flask API...")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "features.pkl")

print(f"📁 Model path: {MODEL_PATH}")
print(f"📁 Feature path: {FEATURE_PATH}")
print(f"📦 Model exists: {os.path.exists(MODEL_PATH)}")
print(f"📦 Features exists: {os.path.exists(FEATURE_PATH)}")

model = None
FEATURES = None

try:
    print("🔄 Loading model...")
    model = joblib.load(MODEL_PATH)
    print("🎉 Model object imported:", type(model))
except Exception as e:
    print("❌ Failed loading model:", e)

try:
    print("🔄 Loading feature list...")
    FEATURES = joblib.load(FEATURE_PATH)
    print("🎉 Features loaded:", FEATURES)
except Exception as e:
    print("❌ Failed loading features:", e)

print("🐍 Reached end of load block\n")
FEATURE_IMPORTANCE = None

if model is not None and hasattr(model, "feature_importances_"):
    FEATURE_IMPORTANCE = dict(
        zip(FEATURES, model.feature_importances_)
    )


@app.route("/")
def home():
    return jsonify({"message": "API running!"})

@app.route("/predict", methods=["POST"])
def predict():
    print("➡️ Endpoint called")
    data = request.get_json()
    print("📩 Data received:", data)

    if FEATURES is None or model is None:
        return jsonify({"error": "Model not ready"}), 500
    
    missing = [f for f in FEATURES if f not in data]
    if missing:
        print("⛔ Missing:", missing)
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    values = [data[f] for f in FEATURES]
    X = np.array([values], dtype=float)
    proba=model.predict_proba(X)[0]
    pred=int(proba.argmax())
    confidence=float(proba[pred])
    return jsonify({
        "prediction":pred,
        "confidence":round(confidence,4)
    })
    # pred = model.predict(X)[0]
    # return jsonify({"prediction": int(pred)})

@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    if FEATURE_IMPORTANCE is None:
        return jsonify({"error": "Feature importance not available"}), 500

    sorted_features = sorted(
        FEATURE_IMPORTANCE.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return jsonify({
        "feature_importance": sorted_features
    })


@app.route("/feedback", methods=["POST"])
def feedback():
    print("🔥 /feedback endpoint HIT")

    data = request.get_json()
    print("📩 Feedback data received:", data)

    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "correct": data["correct"],
        "prediction": data["prediction"],
        **data["input"]
    }

    os.makedirs("feedback", exist_ok=True)
    filepath = os.path.join("feedback", "feedback.csv")

    write_header = not os.path.exists(filepath)

    with open(filepath, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    print("📝 Feedback saved")
    return jsonify({"status": "saved"})


if __name__ == "__main__":
    print("🌍 Flask server starting...\n")
    app.run(debug=False, use_reloader=False)

