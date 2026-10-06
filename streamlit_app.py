import streamlit as st
import joblib, re

model = joblib.load('models/best_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def clean_text(t):
    t = str(t).lower()
    t = re.sub(r'[^a-z0-9\s]', '', t)
    return re.sub(r'\s+', ' ', t).strip()

st.title("🕵️ Scam & Fake News Detector")
text = st.text_area("Paste a news article, SMS, or email:")

if st.button("Analyze"):
    if text.strip():
        vec = vectorizer.transform([clean_text(text)])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.error("⚠️ Flagged as FAKE / SCAM")
        else:
            st.success("✅ Looks REAL / SAFE")
    else:
        st.warning("Please enter some text.")
