from flask import Flask, request, render_template
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

# Load stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Clean text function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['news']
    cleaned = clean_text(user_input)

    vect = vectorizer.transform([cleaned])
    prediction = model.predict(vect)

    if prediction[0] == 1:
        result = "🛑 Fake News"
    else:
        result = "✅ Real News"

    return render_template('index.html', prediction_text=result)

# Run app
if __name__ == "__main__":
    app.run(debug=True)