import os
import tempfile

import streamlit as st
import numpy as np
import pandas as pd
import librosa

from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import load_model

# ─── Startup: load model & build encoder ─────────────────────────────
@st.cache_resource
def load_resources():
    # 1) Load model
    model = load_model('Emotion_Voice_Detection_Model.h5')

    # 2) Build encoder
    df       = pd.read_csv('features.csv')
    labels   = df['labels'].values.reshape(-1,1)
    encoder  = OneHotEncoder()
    encoder.fit(labels)

    return model, encoder

model, encoder = load_resources()

# Streamlit UI
st.title("🎤 Emotion Detection from Voice")

uploaded_file = st.file_uploader(
    "Upload a WAV file (2.5s) →", type=['wav']
)

if uploaded_file:
    # 1) Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded_file.getbuffer())
        tmp_path = tmp.name

    # 2) Load & preprocess
    y, sr = librosa.load(
        tmp_path,
        sr=22050*2,
        duration=2.5,
        offset=0.5,
        res_type='kaiser_fast'
    )
    mfccs   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features = np.mean(mfccs, axis=1).reshape(1, -1)

    # 3) Predict
    pred      = model.predict(features)[0]
    label_full = encoder.inverse_transform([pred])[0][0]
    gender, emotion = label_full.split('_')

    # 4) Display results
    st.audio(tmp_path, format='audio/wav')
    st.markdown(f"**Detected Gender:** {gender.title()}")
    st.markdown(f"**Detected Emotion:** {emotion.title()}")

    # Optional: show probability bar chart
    labels = encoder.categories_[0]
    probs  = pd.DataFrame({
        'Emotion': labels,
        'Probability': pred
    }).sort_values('Probability', ascending=False)

    st.bar_chart(probs.set_index('Emotion'))

    # 5) Clean up
    os.remove(tmp_path)
