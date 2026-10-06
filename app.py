# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import re

app = FastAPI(title="Scam & Fake News Detector")

model = joblib.load('models/best_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

class TextInput(BaseModel):
    text: str

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

@app.get("/")
def root():
    return {"status": "API running"}

@app.post("/predict")
def predict(input: TextInput):
    cleaned = clean_text(input.text)
    vec = vectorizer.transform([cleaned])
    pred = int(model.predict(vec)[0])
    label = "FAKE / SCAM" if pred == 1 else "REAL / SAFE"
    return {"label": label, "raw": pred}
