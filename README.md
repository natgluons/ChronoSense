# ChronoSense: ML-Driven Sleep Audio Analyzer & Optimizer
ChronoSense is an AI-powered tool that analyzes sleep-related audio to detect disturbances (snoring, movement, ambient noise), correlate them with self-reported sleep quality, and recommend personalized sleep strategies using machine learning.

This project focuses on:
* Audio classification of sleep sounds (snoring, coughing, environment noise)
* Sleep quality prediction based on audio features and user logs (e.g., caffeine intake, tiredness)
* Chronotype profiling and adaptive bedtime/wake-up suggestions
* Optional noise playback feedback (e.g., brown noise for light sleepers)
* Pure code. No wearables. Just audio + logs + ML.

## Features
* Sleep disturbance detection from nighttime audio (CNN or pretrained audio models)
* Audio preprocessing with librosa/torchaudio for spectrogram & MFCC extraction
* Lightweight ML pipeline using scikit-learn or PyTorch to predict sleep quality
* Chronobiology-inspired recommendations (circadian-aligned sleep schedule)
* Optional user inputs (caffeine, stress, hours slept) to enrich predictions
* Experimental feedback mode: play relaxing sounds if audio triggers detected

## Tech Stack
- Python 3.10+
- librosa, torchaudio, PyTorch / TensorFlow
- scikit-learn, pandas, numpy
- matplotlib, seaborn for visualization
- Optional UI: Streamlit or Gradio for interaction

## Use Cases
- DIY sleep tracking without expensive wearables
- Research on chronotype/sleep noise relationships
- Training custom audio classification models
- Prototyping personalized wellness or sleep assistant apps