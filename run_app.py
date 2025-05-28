from app.gradio_ui import run_app
from app.logic import log_user_session

log_user_session(
    log_path="data/user_logs.csv",
    caffeine_mg=user_caffeine,
    tiredness=user_tiredness,
    previous_sleep_hrs=user_prev_sleep,
    wake_time=user_wake_time,
    predicted_quality=predicted_quality
)

if __name__ == "__main__":
    run_app()
