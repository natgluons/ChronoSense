import librosa
import numpy as np

def extract_features(audio_path, sr=22050, duration=30):
    y, _ = librosa.load(audio_path, sr=sr, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean  # shape (13,)
