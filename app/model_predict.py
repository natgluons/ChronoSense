import joblib

# Load scikit-learn model for sleep quality prediction
model = joblib.load("models/sleep_quality_model.pkl")

def predict_sleep_quality(features):
    return model.predict([features])[0]

def suggest_bedtime(wake_time, quality_score):
    # Simple logic for now: earlier bed if poor quality
    if quality_score < 5:
        bedtime = wake_time - 8.5
    else:
        bedtime = wake_time - 7.5
    if bedtime < 0:
        bedtime += 24
    return f"{bedtime:.2f}", f"{wake_time:.2f}"
