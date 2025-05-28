import pandas as pd
from datetime import datetime
import os

def log_user_session(log_path, caffeine_mg, tiredness, previous_sleep_hrs, wake_time, predicted_quality):
    new_entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "caffeine_mg": caffeine_mg,
        "tiredness": tiredness,
        "previous_sleep_hrs": previous_sleep_hrs,
        "wake_time": wake_time,
        "sleep_quality": predicted_quality
    }

    # Load or create the CSV
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        df = pd.DataFrame([new_entry])

    df.to_csv(log_path, index=False)
    print(f"Session logged to {log_path}")


def calculate_sleep_duration(prev_sleep_hours, tiredness, caffeine_mg):
    """Estimate needed sleep duration."""
    base = 7.5
    if tiredness > 7:
        base += 1
    if prev_sleep_hours < 6:
        base += 1
    if caffeine_mg > 200:
        base += 0.5
    return min(base, 9.0)

def suggest_bedtime(wake_time, needed_duration):
    """Suggest bedtime given desired wake time and needed sleep duration."""
    bedtime = wake_time - needed_duration
    if bedtime < 0:
        bedtime += 24
    return round(bedtime, 2)

def suggest_times(wake_time, prev_sleep, tiredness, caffeine):
    duration = calculate_sleep_duration(prev_sleep, tiredness, caffeine)
    bedtime = suggest_bedtime(wake_time, duration)
    return bedtime, wake_time
