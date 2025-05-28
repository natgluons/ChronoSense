import gradio as gr
import datetime
import random

# Basic rule-based bedtime recommender
def recommend_bedtime(wake_time, caffeine_intake, tiredness_level, sleep_time):
    # Parse wake time and sleep time
    wake_dt = datetime.datetime.strptime(wake_time, "%H:%M")
    sleep_dt = datetime.datetime.strptime(sleep_time, "%H:%M")
    
    # Adjust sleep need based on caffeine and tiredness
    base_sleep_hours = 8
    if caffeine_intake == "High":
        base_sleep_hours += 0.5
    elif caffeine_intake == "Low":
        base_sleep_hours -= 0.5

    if tiredness_level == "Very Tired":
        base_sleep_hours += 1
    elif tiredness_level == "Slightly Tired":
        base_sleep_hours += 0.5

    # Suggest ideal bedtime
    ideal_bedtime = wake_dt - datetime.timedelta(hours=base_sleep_hours)
    return ideal_bedtime.strftime("%H:%M")

iface = gr.Interface(
    fn=recommend_bedtime,
    inputs=[
        gr.Textbox(label="Desired Wake-up Time (HH:MM, 24h format)", value="07:00"),
        gr.Radio(["None", "Low", "Medium", "High"], label="Caffeine Intake Today", value="Medium"),
        gr.Radio(["Not Tired", "Slightly Tired", "Very Tired"], label="Tiredness Level", value="Slightly Tired"),
        gr.Textbox(label="Last Night Sleep Time (HH:MM, 24h format)", value="23:00")
    ],
    outputs=gr.Textbox(label="Recommended Bedtime"),
    title="ChronoSense MVP",
    description="Enter your daily info to get a personalized bedtime recommendation."
)

iface.launch()
