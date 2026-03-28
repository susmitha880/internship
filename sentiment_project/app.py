# app.py

from flask import Flask, render_template, request
from textblob import TextBlob
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

app = Flask(__name__)

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

stop_words = set(stopwords.words('english'))

# 🔹 Clean text function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# 🔹 Home Page
@app.route('/')
def home():
    return render_template('index.html')

# 🔹 Prediction Route
@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']

    analysis = TextBlob(text)
    polarity = analysis.sentiment.polarity


    if polarity < 0:
        result = "Negative 😞"
    elif polarity == 0:
        result = "Neutral 😐"
    else:
        result = "Positive 😊"

    return render_template('index.html', prediction_text=result)

# 🔹 Run App
if __name__ == "__main__":
    app.run(debug=True)