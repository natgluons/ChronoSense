import gradio as gr
import numpy as np
from app.audio_preprocessing import extract_features
from app.model_predict import predict_sleep_quality, suggest_bedtime

def run_app():
    def process_input(audio, wake_time, caffeine_mg, tiredness, prev_sleep_hrs):
        # Extract audio features
        features = extract_features(audio.name)  # audio is a TemporaryFile

        # Merge with tabular input
        input_vector = np.concatenate([
            features,
            [float(caffeine_mg), float(tiredness), float(prev_sleep_hrs)]
        ])
        
        quality_score = predict_sleep_quality(input_vector)
        bedtime, wake_suggest = suggest_bedtime(float(wake_time), quality_score)

        return f"Predicted Sleep Quality: {quality_score:.2f}", f"Suggested Bedtime: {bedtime}", f"Suggested Wake Time: {wake_suggest}"

    demo = gr.Interface(
        fn=process_input,
        inputs=[
            gr.Audio(source="upload", type="file", label="Sleep Audio"),
            gr.Number(label="Planned Wake-Up Time (24h, e.g., 7.0 for 7 AM)"),
            gr.Number(label="Caffeine Intake (mg)"),
            gr.Slider(0, 10, step=1, label="Tiredness (0 = alert, 10 = very tired)"),
            gr.Number(label="Hours Slept Last Night")
        ],
        outputs=["text", "text", "text"],
        title="ChronoSense MVP",
        description="Upload sleep audio and log your state to get a personalized sleep recommendation."
    )

    demo.launch()
