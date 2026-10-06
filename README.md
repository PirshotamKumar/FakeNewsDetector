 
# Scam & Fake News Detector

ML system that classifies news articles and scam messages using TF-IDF + Logistic Regression.

## Results
| Model | Accuracy |
|-------|----------|
| Logistic Regression | 98.7% |
| Passive Aggressive | 99.1% |
| Naive Bayes | 93.2% |

## Quick Start
pip install -r requirements.txt
python train.py          # train the model
uvicorn app:app --reload # start API
streamlit run streamlit_app.py  # launch UI

## Project Structure
data/       → datasets
models/     → saved .pkl files
notebooks/  → EDA and experiments
src/        → training & inference code
