# app.py
import os
import numpy as np
import pandas as pd
import librosa
import tempfile
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from sklearn.preprocessing import OneHotEncoder

app = Flask(__name__)

# ─── Startup: load model + build encoder ───────────────────────────
MODEL_PATH    = 'Emotion_Voice_Detection_Model.h5'
FEATURES_CSV  = 'features.csv'           # place this next to app.py
EMOTION_MODEL = load_model(MODEL_PATH)

# Read the CSV and fit the encoder one time
df          = pd.read_csv(FEATURES_CSV)
labels_raw  = df['labels'].values.reshape(-1, 1)
encoder     = OneHotEncoder()
encoder.fit(labels_raw)

# You can still keep a Python list of labels if you want quick index lookups:
EMOTION_LABELS = [lab[0] for lab in encoder.categories_[0]]
# ────────────────────────────────────────────────────────────────────

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    audio_file = request.files['file']
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    # Save upload to a real temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    audio_file.save(tmp_path)

    try:
        # Load + preprocess audio
        y, sr = librosa.load(tmp_path,
                             sr=22050*2,
                             duration=2.5,
                             offset=0.5,
                             res_type='kaiser_fast')
        mfccs     = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features  = np.mean(mfccs, axis=1).reshape(1, -1)

        # Predict and invert one-hot
        pred       = EMOTION_MODEL.predict(features)
        label_full = encoder.inverse_transform(pred)[0][0]  
        gender, emotion = label_full.split('_')

        return jsonify({
            "gender":     gender,
            "emotion":    emotion,
            "label_raw":  label_full,
            "probabilities": pred[0].tolist()
        })
    finally:
        os.remove(tmp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
