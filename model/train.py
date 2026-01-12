# model/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ----------------------------
# 1. Load Dataset
# ----------------------------
df = pd.read_csv("data/raw.csv")  # adjust path if needed
print("Dataset loaded. Shape:", df.shape)

# ----------------------------
# 2. Define Target Variable
# ----------------------------
risk_col = "Mortality Rate (%)"  # update according to your dataset
df['risk'] = np.where(df[risk_col] > df[risk_col].median(), 1, 0)
print("Target variable 'risk' created. Distribution:\n", df['risk'].value_counts())

# ----------------------------
# 3. Select Features
# ----------------------------
feature_cols = df.select_dtypes(include=np.number).columns.tolist()
feature_cols.remove('risk')

# ----------------------------
# 4. Sample smaller subset for local testing (memory-friendly)
# ----------------------------
df_sample = df.sample(n=50000, random_state=42)  # 50k rows
X = df_sample[feature_cols]
y = df_sample['risk']
print("Using subset for training. Features shape:", X.shape, "Target shape:", y.shape)

# ----------------------------
# 5. Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 6. Scale Features for Logistic Regression
# ----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------
# 7. Define Models (Memory-friendly)
# ----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=20, n_jobs=1, random_state=42)
}

# ----------------------------
# 8. Train and Evaluate Models
# ----------------------------
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for Logistic Regression
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    
    print(f"{name} Accuracy:", accuracy_score(y_test, y_pred))
    print(f"{name} Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print(f"{name} Classification Report:\n", classification_report(y_test, y_pred))

# ----------------------------
# 9. Save Best Model (Random Forest)
# ----------------------------
best_model = models["Random Forest"]
joblib.dump(best_model, "model/best_model.pkl")
print("\nBest model saved at model/best_model.pkl")
