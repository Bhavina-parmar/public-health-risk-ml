import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "small.csv")
FEEDBACK_PATH = os.path.join(BASE_DIR, "feedback", "feedback.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_model.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "model", "features.pkl")

print("📂 Loading base training data...")
df = pd.read_csv(DATA_PATH)

# recreate target (same logic as train.py)
risk_col = "Mortality Rate (%)"
df["risk"] = (df[risk_col] > df[risk_col].median()).astype(int)

print("📂 Loading feedback data...")
df_feedback = pd.read_csv(FEEDBACK_PATH)

# Load feature list
features = joblib.load(FEATURE_PATH)

# 🔑 Apply feedback corrections
for _, row in df_feedback.iterrows():
    if row["correct"] is False:
        # flip the label
        mask = df["Year"] == row["Year"]
        df.loc[mask, "risk"] = 1 - df.loc[mask, "risk"]

X = df[features]
y = df["risk"]

print("🚀 Retraining model with corrected labels...")
model = RandomForestClassifier(
    n_estimators=20,
    max_depth=8,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, MODEL_PATH)
print("🎉 Model retrained and saved!")
