# predict.py
import joblib
import re

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load saved artifacts
model = joblib.load('models/best_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')

def predict(text):
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    # Confidence score
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(vec)[0]
        confidence = max(prob)
    else:
        confidence = None  # PassiveAggressive doesn't support this
    
    label = "FAKE / SCAM" if pred == 1 else "REAL / SAFE"
    return {"label": label, "confidence": confidence}

if __name__ == "__main__":
    while True:
        text = input("\nEnter text (or 'quit'): ")
        if text.lower() == 'quit':
            break
        result = predict(text)
        print(f"→ {result['label']} (confidence: {result['confidence']})")
