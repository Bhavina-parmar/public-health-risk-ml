# model/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

print("📊 Loading dataset...")
df = pd.read_csv("data/small.csv")

print("✔ Loaded:", df.shape)

# Target column
risk_col = "Mortality Rate (%)"
df['risk'] = np.where(df[risk_col] > df[risk_col].median(), 1, 0)
print("🎯 Target created")

# Select numeric features
feature_cols = df.select_dtypes(include=np.number).columns.tolist()
feature_cols.remove('risk')

X = df[feature_cols]
y = df['risk']
print("🔢 Features:", len(feature_cols))

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train a simple, small model
model = RandomForestClassifier(
    n_estimators=10,
    max_depth=6,
    random_state=42
)

print("🚀 Training Random Forest...")
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("⭐ Accuracy:", accuracy_score(y_test, y_pred))

# Save both files
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_model.pkl")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "features.pkl")

print("💾 Saving model:", MODEL_PATH)
joblib.dump(model, MODEL_PATH)

print("💾 Saving features:", FEATURE_PATH)
joblib.dump(feature_cols, FEATURE_PATH)

print("🎉 Done! Model + features saved.")
