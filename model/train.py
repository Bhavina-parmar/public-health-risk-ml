# model/train.py

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from datetime import datetime
from model.validate import validate_dataframe


# ----------------------------
# Paths
# ----------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "small.csv")
MODEL_DIR = BASE_DIR
LATEST_FILE = os.path.join(MODEL_DIR, "latest.txt")
EXPERIMENT_PATH = os.path.join(BASE_DIR, "..", "experiments", "experiments.csv")


# ----------------------------
# Load dataset
# ----------------------------
print("📊 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✔ Loaded: {df.shape}")

print("🔍 Validating dataset...")
valid, errors = validate_dataframe(df)

if not valid:
    print("❌ Data validation failed:")
    for e in errors:
        print(" -", e)
    raise ValueError("Training stopped due to invalid data")

print("✅ Data validation passed")

# ----------------------------
# Create target
# ----------------------------
risk_col = "Mortality Rate (%)"
df["risk"] = np.where(df[risk_col] > df[risk_col].median(), 1, 0)
print("🎯 Target created")

# ----------------------------
# Select features
# ----------------------------
feature_cols = df.select_dtypes(include=np.number).columns.tolist()
feature_cols.remove("risk")
print(f"🔢 Features: {len(feature_cols)}")

X = df[feature_cols]
y = df["risk"]

# ----------------------------
# Train / Test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train model
# ----------------------------
print("🚀 Training Random Forest...")
best_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
best_model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
preds = best_model.predict(X_test)
acc = accuracy_score(y_test, preds)
print(f"⭐ Accuracy: {acc:.4f}")



# ----------------------------
# Versioning logic
# ----------------------------
if os.path.exists(LATEST_FILE):
    with open(LATEST_FILE, "r", encoding="utf-8", errors="ignore") as f:
        current_version = f.read().strip()
    version_num = int(current_version.replace("v", "")) + 1
else:
    version_num = 1

version_name = f"v{version_num}"
version_path = os.path.join(MODEL_DIR, version_name)
os.makedirs(version_path, exist_ok=True)



# ----------------------------
# Save model + features
# ----------------------------
joblib.dump(best_model, os.path.join(version_path, "model.pkl"))
joblib.dump(feature_cols, os.path.join(version_path, "features.pkl"))

with open(LATEST_FILE, "w", encoding="utf-8") as f:
    f.write(version_name)



# ----------------------------
# Experiment tracking
# ----------------------------
experiment = {
    "version": version_name,
    "accuracy": round(acc, 4),
    "n_estimators": best_model.n_estimators,
    "max_depth": best_model.max_depth,
    "rows": len(df),
    "timestamp": datetime.utcnow().isoformat()
}

exp_df = pd.DataFrame([experiment])

if os.path.exists(EXPERIMENT_PATH):
    exp_df.to_csv(EXPERIMENT_PATH, mode="a", header=False, index=False)
else:
    exp_df.to_csv(EXPERIMENT_PATH, index=False)

print("🧪 Experiment logged")

print(f"💾 Model saved as {version_name}")
print("🎉 Training complete")
# ----------------------------
# Save training statistics (for drift detection)
# ----------------------------
stats = {}

for col in feature_cols:
    stats[col] = {
        "mean": float(df[col].mean()),
        "std": float(df[col].std())
    }

joblib.dump(stats, os.path.join(version_path, "stats.pkl"))
print("📈 Training statistics saved")