from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

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
    pred = model.predict(X)[0]
    return jsonify({"prediction": int(pred)})

if __name__ == "__main__":
    print("🌍 Flask server starting...\n")
    app.run(debug=False, use_reloader=False)

