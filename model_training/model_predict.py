import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Simulate Dataset (MFCCs + tabular) ---
n_samples = 100

# 13 MFCCs (mean values) — dummy audio features
mfcc_features = np.random.normal(size=(n_samples, 13))

# Tabular features
caffeine = np.random.randint(0, 400, size=(n_samples, 1))  # mg
tiredness = np.random.randint(0, 11, size=(n_samples, 1))  # 0-10 scale
prev_sleep = np.random.uniform(3, 9, size=(n_samples, 1))  # hours

X = np.hstack([mfcc_features, caffeine, tiredness, prev_sleep])

# Target sleep quality: higher is better (0–10 scale)
# Synthetic rule: quality drops with more caffeine + tiredness, improves with prior sleep
y = 10 - (caffeine.flatten()/400)*3 - tiredness.flatten()*0.5 + prev_sleep.flatten()*0.8
y = np.clip(y + np.random.normal(0, 0.5, size=y.shape), 0, 10)

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# --- Train Model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# --- Evaluate ---
preds = model.predict(X_test)
rmse = mean_squared_error(y_test, preds, squared=False)
print(f"RMSE: {rmse:.2f}")

# --- Save Model ---
joblib.dump(model, "models/sleep_quality_model.pkl")
print("Model saved to models/sleep_quality_model.pkl")
