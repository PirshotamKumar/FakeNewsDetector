# train.py
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# --- 1. Text Cleaning Function ---
def clean_text(text):
    """Lowercase, remove special characters and extra whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- 2. Load and Prepare Data ---
# Assume your CSV has columns: 'text' and 'label' (0 = real, 1 = fake)
# Or for scam data: 'text' and 'label' (0 = safe, 1 = scam)
def load_and_prepare(filepath):
    df = pd.read_csv(filepath)
    
    # Drop nulls
    df = df.dropna(subset=['text', 'label'])
    
    # Clean text
    df['text'] = df['text'].apply(clean_text)
    
    return df

# --- 3. Train Multiple Models ---
def train_models(X_train, y_train, X_test, y_test):
    # Vectorize: TF-IDF converts text to numerical features
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_df=0.7,
        min_df=5,
        ngram_range=(1, 2)  # unigrams + bigrams
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Define candidate models
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'PassiveAggressive': PassiveAggressiveClassifier(random_state=42),
        'MultinomialNB': MultinomialNB()
    }
    
    results = {}
    best_model = None
    best_acc = 0
    
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)
        acc = accuracy_score(y_test, y_pred)
        
        results[name] = acc
        print(f"\n--- {name} ---")
        print(f"Accuracy: {acc:.4f}")
        print(classification_report(y_test, y_pred))
        
        if acc > best_acc:
            best_acc = acc
            best_model = model
    
    return best_model, vectorizer, results

# --- 4. Main Execution ---
if __name__ == "__main__":
    # Point this to your dataset
    DATA_PATH = "data/news.csv"  # Or "data/scam_messages.csv"
    
    print("Loading data...")
    df = load_and_prepare(DATA_PATH)
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'], df['label'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['label']
    )
    
    print("\nTraining models...")
    best_model, vectorizer, results = train_models(X_train, y_train, X_test, y_test)
    
    # Save artifacts
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    
    print(f"\n✅ Best model saved. All results: {results}")
