from flask import Flask, render_template, request, jsonify
import joblib
import re
import string
import spacy
import nltk
from nltk.corpus import stopwords
import os
# -----------------------------
# Download required NLTK data
# -----------------------------
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# -----------------------------
# Load spaCy model
# -----------------------------
try:
    nlp = spacy.load('en_core_web_sm')
except:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load('en_core_web_sm')

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Preprocessing functions (match training)
# -----------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)                  # Remove text in square brackets
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = re.sub(r'\S*\d\S*', '', text).strip()        # Remove words containing numbers
    return text.strip()

stop_words = nlp.Defaults.stop_words
def lemmatizer(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words]
    return ' '.join(tokens)

def extract_pos_tags(text):
    doc = nlp(text)
    tokens = [token.text for token in doc if token.tag_ == 'NN']  # Extract nouns only
    return ' '.join(tokens)

# -----------------------------
# Load model and vectorizer
# -----------------------------
model = joblib.load('best_model.pkl')
tfidf = joblib.load('tfidf_vectorizer.pkl')

# -----------------------------
# Correct topic mapping (match training)
# -----------------------------
topic_mapping = {
    0: 'Bank Account services',
    1: 'Credit card or prepaid card',
    2: 'Others',
    3: 'Theft/Dispute Reporting',
    4: 'Mortgage/Loan'
}

# -----------------------------
# Flask routes
# -----------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        complaint = request.form['complaint']
        
        # -----------------------------
        # Preprocessing (match training)
        # -----------------------------
        cleaned_text = clean_text(complaint)
        lemmatized_text = lemmatizer(cleaned_text)
        processed_text = extract_pos_tags(lemmatized_text)
        
        # Transform and predict
        X = tfidf.transform([processed_text])
        prediction = model.predict(X)[0]
        topic = topic_mapping.get(prediction, "Unknown")
        
        return jsonify({
            'success': True,
            'topic': topic,
            'processed_text': processed_text
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    # The important part is host='0.0.0.0'
    # Also, use the PORT environment variable provided by Railway
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)